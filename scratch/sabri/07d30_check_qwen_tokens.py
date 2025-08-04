import pickle
from typing import List
import numpy as np
from transformers import AutoTokenizer

from cartridges.structs import TrainingExample
from cartridges.clients.base import FlatTopLogprobs

if __name__ == "__main__":
    path = "/data/sabri/cartridges/2025-07-27-14-11-52-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl"
    data: List[TrainingExample] = pickle.load(open(path, "rb"))["rows"]

    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")

    out = data[5445]
    from cartridges.datasets import qwen_messages_to_element
    element = qwen_messages_to_element(out.messages)
    input_text = qwen_tokenizer.decode(element.input_ids) 
    apply_text = qwen_tokenizer.apply_chat_template(
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

