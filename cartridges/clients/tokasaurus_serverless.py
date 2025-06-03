from __future__ import annotations
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union
import math
from typing import List
from dataclasses import dataclass, replace
import time

from pydantic import field_validator, validator
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from pydrantic import BaseConfig, ObjectConfig

from tokasaurus.
from tokasaurus.allocation import BlockAllocator
from tokasaurus.server_structs import Sequence
from tokasaurus.models.llama import LlamaForCausalLM
from tokasaurus.server_structs import BatchState, BatchSamplingParams, ExtraModelConfig, AttentionInfo
from tokasaurus.server_structs import AppendPageInformationBuilder, PageInformationBuilder, BatchSamplingParamsBuilder
from tokasaurus.allocation import NoSpaceException



from capsules.clients.base import Client, ClientConfig, ClientResponse, Sample, Usage


class TokasaurusServerlessClient(Client):
    """
    SE (01/16/25): The usage returned by this client does NOT take into consider prefix
    sharing. It is simply a sum of the number of tokens in the prompt and completion 
    across the batch.
    """
    
    class Config(ClientConfig):
        _pass_as_config: bool = True

        model_name: str
        max_completion_tokens_per_forward: int = 1024
        max_batch_size: Optional[int] = None
        scheduling_steps_ahead: int = 4
        stop_string_num_token_lookback: int = 5
        page_size: int = 16
        kv_cache_num_tokens: int = 1024
        pp_size: int = 1
        pp_one_at_a_time: bool = False
        pp_disable_microbatching: bool = False
        track_early_stopping: bool = False
        early_stopping_buffer_size: int = 128
        early_stopping_initial_wait: int = 8
        use_hydragen: bool = False

        def kv_cache_num_blocks(self):
            assert self.kv_cache_num_tokens % self.page_size == 0
            return self.kv_cache_num_tokens // self.page_size

        def max_batch_index(self):
            # fudge factor on the total number of sequences running at any time
            return self.max_completion_tokens_per_forward * 2
        

    def __init__(
        self, 
        config: Config,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
    ):
        self.config = config
        self.allocator = BlockAllocator(
            page_size=config.page_size,
            num_blocks=config.kv_cache_num_blocks(),
        )
        self.dtype = getattr(torch, dtype)
        self.device = device
        self._prepare_model(config)
    
    def _split_stop_strings_into_tokens(self, stop: List[str]) -> List[int]:
        stop_strings, stop_tokens = [], []
        for stop_string in stop:
            if stop_string in self.tokenizer.vocab:
                stop_tokens.append(self.tokenizer.convert_tokens_to_ids(stop_string))
            else:
                stop_strings.append(stop_string)
        return stop_strings, stop_tokens

    def chat(
        self, 
        chats: List[List[Dict[str, Any]]], 
        temperature: float = 0.6, 
        stop: Optional[List[str]] = None, 
        max_completion_tokens: Optional[int] = None,

        # client-specific args
        pbar: bool = False,
        log_time: bool = False,
    ) -> ClientResponse:
        assert max_completion_tokens is not None, "max_completion_tokens must be provided"
        prompts = [self.tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
        stop_strings, stop_tokens = self._split_stop_strings_into_tokens(stop)

        max_batch_size = len(prompts) if self.config.max_batch_size is None else self.config.max_batch_size
        responses = []
        for batch_start_idx in range(0, len(prompts), max_batch_size):
            batch_prompts = prompts[batch_start_idx:batch_start_idx + max_batch_size]
            batch_responses, batch_usage, timings = self._sample(
                prompts=batch_prompts, 
                temperature=temperature, 
                stop_strings=stop_strings, 
                stop_tokens=stop_tokens,
                max_completion_tokens=max_completion_tokens,
                pbar=pbar,
                log_time=log_time,
            )
            responses.extend(batch_responses)
        return ClientResponse(samples=responses, timings=timings)
    
    def complete(
        self, 
        prompts: List[str], 
        temperature: float = 0.6, 
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,

        # client-specific args
        pbar: bool = False,
        log_time: bool = False,
    ) -> ClientResponse:
        assert max_completion_tokens is not None
        max_batch_size = len(prompts) if self.config.max_batch_size is None else self.config.max_batch_size
        responses = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        for batch_start_idx in range(0, len(prompts), max_batch_size):
            batch_prompts = prompts[batch_start_idx:batch_start_idx + max_batch_size]
            batch_responses, batch_usage, timings = self._sample(
                prompts=batch_prompts, 
                temperature=temperature, 
                stop_strings=stop,
                max_completion_tokens=max_completion_tokens,
                pbar=pbar,
                log_time=log_time,
            )
            responses.extend(batch_responses)
            usage += batch_usage
        return ClientResponse(samples=responses, usage=usage, timings=timings)

    def _prepare_model(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)        
        self.model = LlamaForCausalLM.from_pretrained(
            config.model_name,
            dtype=self.dtype,
            device=self.device,
            extra_config=ExtraModelConfig(
                pp_size=config.pp_size,
                pp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        )
        self.model.setup_caches(
            num_pages=config.kv_cache_num_blocks(), 
            page_size=config.page_size
        )  
    
    def clear_cache(self):
        self.allocator = BlockAllocator(self.config)
        # self.model.setup_caches(
        #     num_pages=self.config.kv_cache_num_blocks(), 
        #     page_size=self.config.page_size
        # )  

        
    def _sample(
        self, 
        prompts: List[Union[str, List[int]]], 
        temperature: float = 0.6, 
        stop_strings: List[str] = [], 
        stop_tokens: List[Union[str, int]] = [],
        max_completion_tokens: int = 1,
        pbar: bool = False,
        log_time: bool = False,
    ) -> List[Sample]:        
        # need to get the token id corresponding to string stop tokens
        if hasattr(self.tokenizer, "eos_token_id"):
            stop_tokens.append(self.tokenizer.eos_token_id)
        stop_tokens = [
            self.tokenizer.convert_tokens_to_ids(token) if isinstance(token, str) else token 
            for token in stop_tokens
        ]
        sequences = self._prepare_sequences(prompts, max_completion_tokens=max_completion_tokens)

        do_sample = temperature is not None and temperature > 0
        sampling_params = BatchSamplingParams(
            temperature=None if temperature is None else torch.tensor([temperature] * len(sequences), device=self.model.device),
            greedy_mask=torch.tensor([not do_sample] * len(sequences), device=self.model.device),
        )

        timings = []
        completed: List[Sequence] = []
        for token_idx in tqdm(range(max_completion_tokens), desc="Sampling", disable=not pbar):
            t0 = time.time()
            batch_state: BatchState = self._build_batch(
                sequences=sequences, 
                device=self.model.device, 
                page_size=self.allocator.page_size,
                sampling_params=sampling_params,
            )
            t1 = time.time()
            
            batch_state.sampling_params = sampling_params
            
            t2 = time.time()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.dtype):
                self.model.plan(batch_state=batch_state)
                out = self.model(batch_state=batch_state)
            t3 = time.time()
            
            self._update_sequences(sequences=sequences, batch_state=out)
            t4 = time.time()
            if len(stop_strings) > 0:
                newly_completed, sequences = self._check_stop_strings(sequences=sequences, stop_strings=stop_strings)
                completed.extend(newly_completed)
            if len(stop_tokens) > 0:
                newly_completed, sequences = self._check_stop_tokens(sequences=sequences, stop_tokens=stop_tokens)
                completed.extend(newly_completed)
            t5 = time.time()

            timings.append({"build": t1-t0, "run": t3-t2, "update": t4-t3, "check": t5-t4, "token_idx": token_idx})
            if log_time:
                print(timings[-1])

            if len(sequences) == 0:
                break
        # need to sort by id to match the order of the input prompts
        completed.extend(sequences)
        completed = sorted(completed, key=lambda x: x.id)
        for seq in completed:
            self.allocator.free_and_update(seq)
    
        responses = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        for seq in completed:
            tokens = [self.tokenizer.decode(token_id) for token_id in seq.completion_ids]
            responses.append(
                Sample(
                    text="".join(tokens),
                    token_ids=seq.completion_ids, 
                    tokens=tokens,
                    log_prob=seq.logprobs,
                    stop_reason="max_completion_tokens" if seq.completion_total == len(seq.completion_ids) else "stop_string",
                )
            )
            usage += Usage(
                prompt_tokens=len(seq.input_ids),
                completion_tokens=len(seq.completion_ids),
            )
        return responses, usage, timings

    def _split_stop_strings_into_tokens(self, stop: Optional[List[str]]) -> Tuple[List[str], List[int]]:
        stop_strings, stop_tokens = [], []
        if stop is None:
            return stop_strings, stop_tokens
        for stop_string in stop:
            if stop_string in self.tokenizer.vocab:
                stop_tokens.append(self.tokenizer.convert_tokens_to_ids(stop_string))
            else:
                stop_strings.append(stop_string)
        return stop_strings, stop_tokens

    def _check_stop_strings(self, sequences: List[Sequence], stop_strings: List[str]):
        completed, active = [], []
        completions = self.tokenizer.batch_decode(
            [seq.completion_ids for seq in sequences],
        )
        for seq, completion in zip(sequences, completions):
            for stop_string in stop_strings:
                if stop_string in completion:
                    completed.append(seq)
                    break
            else:
                active.append(seq)
        return completed, active


    def _check_stop_tokens(self, sequences: List[Sequence], stop_tokens: List[int]):
        completed, active = [], []
        for seq in sequences:
            if seq.completion_ids[-1] in stop_tokens:
                completed.append(seq)
            else:
                active.append(seq)
        return completed, active


    def _update_sequences(self, sequences: List[Sequence], batch_state: BatchState):
        output_ids = batch_state.output_ids.tolist()
        logprobs = batch_state.logprobs.tolist()
        if len(sequences) > 1:
            breakpoint()
        for seq in sequences:
            seq.completion_ids.append(output_ids[seq.batch_index])
            seq.logprobs.append(logprobs[seq.batch_index])
            if seq.prompt_scheduled < seq.prompt_total():
                seq.prompt_scheduled = seq.prompt_total()
            seq.completion_scheduled += 1

    def _prepare_sequences(
        self, 
        prompts: List[Union[str, int]],
        max_completion_tokens: int
    ) -> List[Sequence]:
        if isinstance(prompts[0], str):
            batch_ids = self.tokenizer(prompts).input_ids   
        else:
            batch_ids = prompts

        sequences = []
        for i, token_ids in enumerate(batch_ids):
            seq = Sequence(
                id=i,
                input_ids=token_ids,
                completion_total=max_completion_tokens,
                completion_scheduled=0,
                request=None,
                output=None,
                # prompt_total=len(token_ids),
                batch_index=i
            )
            
            self.allocator.allocate_prefill(
                seq, 

                # NOTE: We will update the interface of allocator to be simple allocate
                # this will not require the extra step of determining of the overhang 
                # from the prefill, which we are currently not doing below!
                num_blocks_reserved_for_decode=math.ceil(max_completion_tokens / self.allocator.page_size)
            )
            
            # we need to implement this ourselves because the 
            pages_needed_for_decode = math.ceil((
                max_completion_tokens - 
                len(seq.kv_indices) * self.config.page_size + # number of tokens allocated so far
                (len(token_ids) - seq.num_cached_prompt_tokens)  # number of tokens needed for prefill
            ) / self.config.page_size)

            if pages_needed_for_decode > self.allocator.num_free_blocks:
                raise NoSpaceException()            
            new_blocks = self.allocator.request_blocks_for_seq(seq, pages_needed_for_decode)
            seq.kv_indices.extend([block.idx for block in new_blocks])

            # TODO: determine if we should consider the cached prompt tokens in the sequence length. 
            assert len(seq.kv_indices) * self.config.page_size >= len(seq.input_ids) + max_completion_tokens

            # result: BlockAllocationResult = self.allocator.allocate(seq)
            # seq.kv_indices = result.block_indices
            # seq.prompt_scheduled = result.num_cached_prompt_tokens
            sequences.append(seq)

        return sequences
        
    def _build_batch(
        self, 
        sequences: list[Sequence],
        page_size: int,
        sampling_params: BatchSamplingParams,
        starting_prefill_offset: int = 0,
        embedding: bool = False,
        device: str = "cuda:0",
    ) -> BatchState:
        prefill_seqs = [
            (seq, seq.prompt_total() - seq.prompt_scheduled) 
            for seq in sequences if seq.prompt_scheduled < seq.prompt_total()
        ]
        decoding_seqs = [
            seq for seq in sequences 
            if (seq.prompt_scheduled == seq.prompt_total()) and (seq.completion_total > 0 or embedding)
        ]
        # NOTE: important that we do not include sequences as decoding if they are not requesting any completions
        # this arises sometimes when we're doing 
        if embedding and len(prefill_seqs) > 0:
            raise ValueError("Cannot build a batch that mixes prefill and decoding sequences when embedding")
            

        append_builder = AppendPageInformationBuilder()
        prefill_builder = PageInformationBuilder()
        decode_builder = PageInformationBuilder()

        sampling_builder = BatchSamplingParamsBuilder()

        position_ids = []
        lm_head_indices = []
        prefill_input_ids_list = []

        for i, (seq, slen) in enumerate(prefill_seqs):
            assert seq.completion_scheduled == 0
            start_position = seq.prompt_scheduled
            if i == 0:
                start_position += starting_prefill_offset
            end_position = start_position + slen

            prefill_ids = seq.input_ids[start_position:end_position]
            assert len(prefill_ids) == slen

            prefill_input_ids_list.extend(prefill_ids)

            seq_pos_ids = list(range(start_position, end_position))
            position_ids.extend(seq_pos_ids)

            for builder in [prefill_builder, append_builder]:
                builder: AppendPageInformationBuilder | PageInformationBuilder
                assert seq.kv_indices is not None
                builder.add_sequence(
                    kv_indices=seq.kv_indices,
                    kv_seq_len=start_position + slen,
                    num_qtokens=slen,
                    page_size=page_size,
                )

            if end_position == seq.prompt_total():
                lm_head_indices.append(len(position_ids) - 1)

                sampling_builder.add_sequence(
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                )

        if self.config.use_hydragen:
            assert prefix_tree is not None
            decoding_sids = [seq.id for seq in decoding_seqs]

            hydragen_groups = group_for_hydragen(
                prefix_tree,
                decoding_sids,
                min_group_size=config.hydragen_min_group_size,
                min_prefix_len=config.hydragen_min_prefix_len,
            )
            hydragen_builder = PageInformationBuilder()

            sid_to_decoding_seq = {seq.id: seq for seq in decoding_seqs}

            # we re-order the decoding sequences so that seqs in the same hydragen group
            # are adjacent to each other, with ungrouped seqs at the end.
            ordered_decoding_seqs = []

            grouped_sids: set[str] = set()
            sid_to_group: dict[str, HydragenGroup] = {}
            for group in hydragen_groups:
                hydragen_builder.add_sequence(
                    kv_indices=group.block_ids,
                    kv_seq_len=len(group.block_ids) * page_size,
                    num_qtokens=len(group.seq_ids),
                    page_size=page_size,
                )
                for sid in group.seq_ids:
                    sid_to_group[sid] = group
                    grouped_sids.add(sid)
                    ordered_decoding_seqs.append(sid_to_decoding_seq[sid])

            # ungrouped seqs at the end
            for seq in decoding_seqs:
                if seq.id not in grouped_sids:
                    ordered_decoding_seqs.append(seq)

            assert len(ordered_decoding_seqs) == len(decoding_seqs)
            decoding_seqs = ordered_decoding_seqs

        decoding_input_ids = []
        for seq in decoding_seqs:
            # NOTE: off by one since last prefill token produces first
            # decode token.
            position_of_current_token = seq.total_scheduled() - 1
            position_ids.append(position_of_current_token)

            if self.config.use_hydragen and seq.id in sid_to_group:
                group = sid_to_group[seq.id]
                starting_block = len(group.block_ids)
            else:
                starting_block = 0

            assert seq.kv_indices is not None
            decode_builder.add_sequence(
                kv_indices=seq.kv_indices,
                kv_seq_len=position_of_current_token,
                num_qtokens=1,
                page_size=page_size,
                starting_block=starting_block,
            )
            # starting block of 0 needed for append
            append_builder.add_sequence(
                kv_indices=seq.kv_indices,
                kv_seq_len=position_of_current_token,
                num_qtokens=1,
                page_size=page_size,
            )

            sampling_builder.add_sequence(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
            )

            decoding_input_ids.append(seq.completion_ids[-1])

        prefill_lengths = [slen for _, slen in prefill_seqs]
        start_of_decode = sum(prefill_lengths)

        lm_head_indices.extend(
            list(range(start_of_decode, start_of_decode + len(decoding_seqs)))
        )

        batch_indices = []
        
        for i, (seq, num_tokens) in enumerate(prefill_seqs):
            seq.batch_index = i
            batch_indices.extend([seq.batch_index] * num_tokens)

        for i, seq in enumerate(decoding_seqs):
            seq.batch_index = i + len(prefill_seqs)
            batch_indices.extend([seq.batch_index] * 1)

        attention_info = AttentionInfo(
            page_size=page_size,
            append_info=append_builder.build(),
            prefill_info=prefill_builder.build(),
            decode_info=decode_builder.build(),
            hydragen_info=hydragen_builder.build() if self.config.use_hydragen else None,
        )

        input_ids = prefill_input_ids_list + decoding_input_ids
        to_tensor = lambda x: torch.tensor(x, dtype=torch.long, device=device)
        batch_state = BatchState(
            input_ids=to_tensor(input_ids),
            position_ids=to_tensor(position_ids),
            sampling_params=sampling_params.to(device),
            attention_info=attention_info.to(device),
            lm_head_indices=to_tensor(lm_head_indices),
        )
        if (batch_state.input_ids.shape[0] != batch_state.position_ids.shape[0]) and (batch_state.input_ids.shape[0] != 0):
            breakpoint()
        return batch_state







        # # We are building up these tensors for the page table layout of FlashInfer
        # # See here: https://docs.flashinfer.ai/tutorials/kv_layout.html 
        # # NOTE: the last page len INCLUDES the tokens currently being processed
        # qo_indptr, kv_indices, kv_indptr, kv_last_page_len = [0], [], [0], []
        # position_ids, lm_head_indices, batch_indices = [], [], []
        # input_ids = []

        # def add_sequence(
        #     seq: Sequence,
        #     num_tokens: int, 
        #     starting_pos: int,
        #     batch_index: int,
        # ):
        #     num_tokens_including_current = starting_pos + num_tokens
        #     seq_pos_ids = list(range(starting_pos, starting_pos + num_tokens))
        #     position_ids.extend(seq_pos_ids)

        #     qo_indptr.append(qo_indptr[-1] + num_tokens)

        #     seq_kv_indices = seq.kv_indices[
        #         : math.ceil(num_tokens_including_current / page_size)
        #     ]

        #     kv_indices.extend(seq_kv_indices)
        #     kv_indptr.append(kv_indptr[-1] + len(seq_kv_indices))

        #     last_page_len = num_tokens_including_current % page_size
        #     if last_page_len == 0:
        #         last_page_len = page_size

        #     kv_last_page_len.append(last_page_len)

        #     batch_indices.extend([batch_index] * num_tokens)

        # for i, (seq, slen) in enumerate(prefill_seqs):
        #     assert seq.completion_scheduled == 0
        #     start_position = seq.prompt_scheduled
        #     if i == 0:
        #         start_position += starting_prefill_offset
        #     end_position = start_position + slen

        #     prefill_ids = seq.input_ids[start_position:end_position]
        #     assert len(prefill_ids) == slen

        #     input_ids.extend(prefill_ids)
        #     add_sequence(num_tokens=slen, starting_pos=start_position, seq=seq, batch_index=i) 
        #     seq.batch_index = i

        #     if (end_position == seq.prompt_total()) and (seq.completion_total > 0):
        #         lm_head_indices.append(len(position_ids) - 1)

        # decoding_index = qo_indptr[-1]

        # lm_head_indices.extend(
        #     list(range(decoding_index, decoding_index + len(decoding_seqs)))
        # )

        # for i, seq in enumerate(decoding_seqs):
        #     # NOTE: off by one since last prefill token produces first
        #     # decode token.
        #     position_of_current_token = seq.current_num_scheduled() - 1
        #     add_sequence(num_tokens=1, starting_pos=position_of_current_token, seq=seq, batch_index=i + len(prefill_seqs))
        #     seq.batch_index = i + len(prefill_seqs)

        #     # This is to enforce that we don't include input_ids for the embeddings
        #     if not embedding:
        #         input_ids.append(seq.completion_ids[-1])

        # assert all(0 < x <= page_size for x in kv_last_page_len)

        # to_tensor = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
        # page_information = PageInformation(
        #     num_tokens=0,  # FIXME: Changed this from num_decode_tokens to num_tokens
        #     # page_size=page_size,
        #     num_seqs=len(sequences),  # FIXME: Added this but not sure it's correct
        #     qo_indptr=to_tensor(qo_indptr),
        #     kv_last_page_len=to_tensor(kv_last_page_len),
        #     kv_indptr=to_tensor(kv_indptr),
        #     kv_indices=to_tensor(kv_indices),
        # )
        # # FIXME: Need to create prefill and decode tensors, not sure where it is now
        # # page_information.create_prefill_decode_tensors()

