import pickle
from typing import List
import numpy as np
from transformers import AutoTokenizer

from cartridges.structs import TrainingExample
from cartridges.clients.base import FlatTopLogprobs


def convert_messages_to_llama2(messages: List[TrainingExample.Message], llama2_tokenizer, llama3_tokenizer) -> List[TrainingExample.Message]:
    """Convert TrainingExample messages from Llama3 to Llama2 format"""
    converted_messages = []
    
    for message in messages:
        # Create conversation format for tokenization
        llama2_token_ids = llama2_tokenizer.tokenizer(message.content)

        if message.role == "user":
            llama2_token_ids = llama2_tokenizer.tokenizer(message.content)
        else:
            llama2_token_ids = llama2_tokenizer.tokenizer(message.content)
        
        # Tokenize with Llama2 tokenizer
        # llama2_token_ids = llama2_tokenizer.tokenize(message.content)
        breakpoint()
        
        # Create top_logprobs with probability 1.0 for each token
        num_tokens = len(llama2_token_ids)
        
        # Create FlatTopLogprobs with probability 1.0 for correct tokens
        token_idx = np.arange(num_tokens, dtype=np.int32)
        token_id = np.array(llama2_token_ids, dtype=np.int32)
        logprobs = np.zeros(num_tokens, dtype=np.float32)  # log(1.0) = 0.0
        
        flat_top_logprobs = FlatTopLogprobs(
            token_idx=token_idx,
            token_id=token_id,
            logprobs=logprobs,
            shape=(num_tokens, 1)  # 1 top logprob per token
        )
        
        # Create new message with Llama2 tokenization
        converted_message = TrainingExample.Message(
            content=message.content,
            role=message.role,
            token_ids=llama2_token_ids,
            top_logprobs=flat_top_logprobs
        )
        
        converted_messages.append(converted_message)
    
    return converted_messages


def convert_training_examples_to_llama2(examples: List[TrainingExample]) -> List[TrainingExample]:
    """Convert a list of TrainingExamples from Llama3 to Llama2"""
    # Load tokenizers
    llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    converted_examples = []
    
    for example in examples:
        # Convert messages
        converted_messages = convert_messages_to_llama2(
            example.messages, 
            llama2_tokenizer, 
            llama3_tokenizer
        )
        
        # Create new TrainingExample with converted messages
        converted_example = TrainingExample(
            messages=converted_messages,
            system_prompt=example.system_prompt,
            type=example.type,
            metadata=example.metadata
        )
        
        converted_examples.append(converted_example)
    
    return converted_examples


if __name__ == "__main__":
    path = "/data/sabri/cartridges/2025-07-26-12-21-32-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0/artifact/dataset.pkl"
    data: List[TrainingExample] = pickle.load(open(path, "rb"))["rows"]

    llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") 

    out = data[0]
    from cartridges.datasets import llama_messages_to_element
    element = llama_messages_to_element(out.messages)
    input_text = llama3_tokenizer.decode(element.input_ids) 
    apply_text = llama3_tokenizer.apply_chat_template(
        [dict(
            role=msg.role,
            content=msg.content,
        ) for msg in out.messages], tokenize=False
    )

    breakpoint()
    print(input_text)
    print(apply_text)
    
    print(f"Converting {len(data)} training examples from Llama3 to Llama2...")
    converted_data = convert_training_examples_to_llama2(data)
        
    # Save converted data
    output_path = path.replace(".pkl", "_llama2.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"rows": converted_data}, f)
    
    print(f"Saved converted data to {output_path}")
    print(f"Original examples: {len(data)}")
    print(f"Converted examples: {len(converted_data)}")
    
    # Example comparison
    if data and converted_data:
        orig_msg = data[0].messages[0]
        conv_msg = converted_data[0].messages[0]
        print(f"\nExample comparison:")
        print(f"Original token_ids: {orig_msg.token_ids[:10] if orig_msg.token_ids else None}...")
        print(f"Converted token_ids: {conv_msg.token_ids[:10]}...")
        print(f"Converted has top_logprobs: {conv_msg.top_logprobs is not None}")

