import os
import pickle
from datasets import load_dataset
from cartridges.structs import TrainingExample

def create_gsm8k_training_examples():
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    
    training_examples = []
    
    for example in train_data:
        question = example["question"]
        answer = example["answer"]
        
        messages = [
            TrainingExample.Message(
                content=question,
                role="user",
                token_ids=None
            ),
            TrainingExample.Message(
                content=answer,
                role="assistant",
                token_ids=None
            )
        ]
        
        training_example = TrainingExample(
            messages=messages,
            system_prompt="You are a helpful assistant that solves math problems step by step.",
            type="math_reasoning",
            metadata={"source": "gsm8k", "split": "train"}
        )
        
        training_examples.append(training_example)
    
    return training_examples

if __name__ == "__main__":
    print("Loading GSM8K dataset and creating training examples...")
    examples = create_gsm8k_training_examples()
    
    print(f"Created {len(examples)} training examples")
    
    output_dir = os.path.join(
        os.environ["CARTRIDGES_DIR"], "outputs"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gsm8k_training_examples.pkl")
    with open(output_file, "wb") as f:
        pickle.dump({"rows": examples}, f)
    
    print(f"Saved training examples to {output_file}")