from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from transformers import AutoTokenizer

from cartridges.clients.base import (
    Client,
    ClientSample,
    ClientConfig,
    ClientResponse,
    TopLogprobs,
)
from cartridges.clients.usage import Usage
from cartridges.clients.mixin import ServerMixin
from cartridges.utils import get_logger

if TYPE_CHECKING:
    from sglang import ProgramState


class SGLangClient(Client, ServerMixin):
    """
    SE (01/16/25): The usage returned by this client does NOT take into consider prefix
    sharing. It is simply a sum of the number of tokens in the prompt and completion 
    across the batch.
    """

    class Config(ClientConfig):
        _pass_as_config: bool = True
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct"

        url: Optional[str] = None

        mem_fraction_static: float = 0.8

        timeout: int = 100

    def __init__(self, config: Config):
        self.config = config

        if config.url is None:
            port = self.find_free_port()
            self.url = f"http://localhost:{port}"
            launch_command = f"""python -m sglang.launch_server \
            --port {port} \
            --model-path {config.model_name} \
            --mem-fraction-static {config.mem_fraction_static} \
            """
            self.launch_server(launch_command, port, capture_output=config.capture_output)
        else:
            self.url = config.url
        
        from sglang import RuntimeEndpoint
        self.backend = RuntimeEndpoint(self.url)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    def complete(
        self,
        prompts: List[Union[str, List[int]]],
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        max_completion_tokens: int = 1,
        **kwargs,
    ) -> ClientSample:
        """
        Tokasaurus does not directly support a `complete` API.
        """
        raise NotImplementedError(
            "The `complete` method is not yet supported by Tokasuarus Client."
        )
        
        
    def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        logprobs_start_message: Optional[int] = None,
        **kwargs
    ) -> ClientResponse:
        
        import sglang as sgl
        from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
        from sglang.lang.interpreter import ProgramState

        stop_strings, stop_token_ids = self._split_stop_strings_into_tokens(stop)
        
        @function
        def parallel_chats(s: ProgramState, messages: List[str]):
            for message in messages:
                s += getattr(sgl, message["role"])(message["content"])
            
            # Compute the number of tokens in the first `logprobs_start_message` 
            # messages. This is useful for excluding the system prompt (which can be
            # very long) from the logprobs that need to be sent back. 
            if logprobs_start_message is None or logprobs_start_message == 0:
                logprob_start_len = 0
            else:
                logprob_start_len = len(self.tokenizer.apply_chat_template(
                    messages[:logprobs_start_message],
                    add_generation_prompt=True,
                    tokenize=True
                ))
            print(logprob_start_len)
            s += assistant(gen(
                "answer", 
                max_tokens=max_completion_tokens, 
                return_logprob=True,
                temperature=temperature,
                top_logprobs_num=top_logprobs,
                logprob_start_len=logprob_start_len,
                stop=stop_strings,
                stop_token_ids=stop_token_ids,

            ))
        
        print(f"Running parallel chats with {len(chats)} chats")
        states: List[ProgramState] = parallel_chats.run_batch(
            [
                {"messages": messages}
                for messages in chats
            ], 
            backend=self.backend
        )
        print(f"Finished running parallel chats with {len(states)} states")
        return self._parse_states(states)
    

    def _parse_logprobs(self, answer: Dict[str, Any]) -> TopLogprobs:
        top_logprobs, top_ids = [], []

        num_input_tokens = 0
        for key in ["input", "output"]:
            if f"{key}_top_logprobs" not in answer:
                
                continue
            _, token_ids, _ = zip(*answer[f"{key}_token_logprobs"])

            if key == "input":  
                num_input_tokens += len(token_ids)

            for logprobs in answer[f"{key}_top_logprobs"]:
                if logprobs is None:
                    continue
                logprob, token_ids, _ = zip(*logprobs)
                top_logprobs.append(logprob)
                top_ids.append(token_ids)

        top_logprobs = TopLogprobs(
            num_input_tokens=num_input_tokens,
            token_ids=token_ids,
            top_logprobs=np.array(top_logprobs),
            top_ids=np.array(top_ids),
        )
        return top_logprobs

    def _parse_states(self, states: List[any]) -> List[any]:

        responses = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        for state in states:
            if state.error() is not None:
                print(f"[SGLang] Error: {state.error()}")
                responses.append(ClientSample(
                    text=state.messages()[-1]["content"],
                    tokens=[],
                    log_prob=0,
                    token_ids=[],
                    stop_reason="error"
                ))
                continue
            
            answer = state.stream_executor.meta_info["answer"]
            top_logprobs = self._parse_logprobs(answer)

            responses.append(
                ClientSample(
                    text=state.messages()[-1]["content"],
                    num_output_tokens=answer["completion_tokens"],
                    top_logprobs=top_logprobs,
                )
            )
            usage += Usage(
                prompt_tokens=answer["prompt_tokens"],
                completion_tokens=answer["completion_tokens"]
            )

        return ClientResponse(samples=responses, usage=usage)

    
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


