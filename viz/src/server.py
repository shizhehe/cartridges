#!/usr/bin/env python3
"""
Backend API server for the dataset visualization app.
Provides endpoints for dataset discovery and loading.
"""

import os
import pickle
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Dataset Visualization API", version="1.0.0")


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

def load_dataset_from_pickle(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats
        if isinstance(data, dict):
            if 'rows' in data:
                return data['rows']
            elif 'examples' in data:
                return data['examples']
            elif 'data' in data:
                return data['data']
            else:
                # Try to extract first list value
                for value in data.values():
                    if isinstance(value, list):
                        return value
                return []
        elif isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        return []

def serialize_training_example(example) -> Dict[str, Any]:
    """Convert TrainingExample to JSON-serializable format."""
    try:
        messages = []
        for msg in example.messages:
            print(msg)
            message_data = {
                'content': msg.content,
                'role': msg.role,
                'token_ids': msg.token_ids,
                'top_logprobs': None
            }
            
            # Handle logprobs if they exist
            if hasattr(msg, 'top_logprobs') and msg.top_logprobs is not None:
                message_data['top_logprobs'] = {
                    'logprobs': msg.top_logprobs.logprobs.tolist() if hasattr(msg.top_logprobs.logprobs, 'tolist') else msg.top_logprobs.logprobs,
                    'token_ids': msg.top_logprobs.token_ids.tolist() if hasattr(msg.top_logprobs.token_ids, 'tolist') else msg.top_logprobs.token_ids
                }
            
            messages.append(message_data)
        
        return {
            'messages': messages,
            'system_prompt': example.system_prompt,
            'type': example.type,
            'metadata': example.metadata
        }
    except Exception as e:
        print(f"Error serializing example: {e}")
        return {
            'messages': [],
            'system_prompt': '',
            'type': 'unknown',
            'metadata': {}
        }

@app.get("/api/datasets")
def get_datasets(output_dir: Optional[str] = Query(None)):
    """Discover and return available datasets."""
    
    # If no output_dir specified, try common locations
    search_paths = []
    if output_dir:
        search_paths.append(output_dir)
    
    # Add some common search paths
    search_paths.extend([
        os.path.expanduser('~/code/cartridges/outputs'),
        os.path.expanduser('~/outputs'),
        '/tmp/cartridges_output',
        './outputs'
    ])
    
    # Also check environment variables
    env_output_dir = os.environ.get('CARTRIDGES_OUTPUT_DIR')
    if env_output_dir:
        search_paths.insert(0, env_output_dir)
    
    datasets = []
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        # Find all .pkl files recursively
        pkl_files = glob.glob(os.path.join(search_path, '**/*.pkl'), recursive=True)
        
        for pkl_file in pkl_files:
            try:
                # Quick check if it's a valid dataset
                examples = load_dataset_from_pickle(pkl_file)
                if examples:
                    file_path = Path(pkl_file)
                    dataset_name = file_path.stem
                    
                    # Calculate relative path from search_path
                    try:
                        relative_path = str(file_path.relative_to(search_path))
                    except ValueError:
                        # If relative_to fails, just use the filename
                        relative_path = file_path.name
                    
                    datasets.append({
                        'name': dataset_name,
                        'path': pkl_file,
                        'relative_path': relative_path,
                        'size': len(examples),
                        'directory': str(file_path.parent)
                    })
            except Exception as e:
                print(f"Error checking {pkl_file}: {e}")
                continue
    print(len(datasets))
    
    return datasets

@app.get("/api/dataset/{dataset_path:path}")
def get_dataset(dataset_path: str):
    """Load and return a specific dataset."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        examples = load_dataset_from_pickle(dataset_path)
        
        # Serialize examples
        serialized_examples = []
        for example in examples:
            serialized_examples.append(serialize_training_example(example))
        
        return {
            'examples': serialized_examples,
            'count': len(serialized_examples),
            'path': dataset_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    print("Health check called!")
    return {'status': 'healthy'}

@app.post("/api/decode-tokens")
def decode_tokens(request: Dict[str, Any]):
    """Decode token IDs to text using the specified tokenizer."""
    try:
        tokenizer_name = request.get('tokenizer_name', 'meta-llama/Llama-3.2-3B-Instruct')
        token_ids = request.get('token_ids', [])
        
        # Try to load the tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception as e:
            # Fallback to a default tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
            except Exception as e2:
                return {'error': f'Failed to load any tokenizer: {str(e2)}'}
        
        # Decode tokens
        decoded_tokens = []
        for token_id in token_ids:
            try:
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                decoded_tokens.append(decoded)
            except Exception:
                decoded_tokens.append(f"[ID:{token_id},ERR]")
        
        return {'decoded_tokens': decoded_tokens}
    
    except Exception as e:
        return {'error': str(e)}

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8000))
#     uvicorn.run("server:app", host='0.0.0.0', port=port, reload=True)