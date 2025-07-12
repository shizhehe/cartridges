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
            message_data = {
                'content': msg.content,
                'role': msg.role,
                'token_ids': msg.token_ids,
                'top_logprobs': None
            }
            
            # Handle logprobs if they exist
            if hasattr(msg, 'top_logprobs') and msg.top_logprobs is not None:
                top_logprobs = msg.top_logprobs.reconstruct()
                print(top_logprobs)
                message_data['top_logprobs'] = {
                    'logprobs': top_logprobs.logprobs.tolist() if hasattr(top_logprobs.logprobs, 'tolist') else top_logprobs.logprobs,
                    'token_ids': top_logprobs.token_ids.tolist() if hasattr(top_logprobs.token_ids, 'tolist') else top_logprobs.token_ids
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

def quick_check_dataset(file_path: str) -> Optional[int]:
    """Quickly check if a pickle file is a valid dataset and return approximate size."""
    return 16
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data formats and get count without loading all examples
        if isinstance(data, dict):
            if 'rows' in data and isinstance(data['rows'], list):
                return len(data['rows'])
            elif 'examples' in data and isinstance(data['examples'], list):
                return len(data['examples'])
            elif 'data' in data and isinstance(data['data'], list):
                return len(data['data'])
            else:
                # Try to find first list value
                for value in data.values():
                    if isinstance(value, list):
                        return len(value)
                return 0
        elif isinstance(data, list):
            return len(data)
        else:
            return 0
    except Exception as e:
        print(f"Error quick-checking dataset {file_path}: {e}")
        return None

@app.get("/api/datasets")
def discover_datasets(output_dir: Optional[str] = Query(None)):
    """Discover and return available datasets without loading full content."""
    
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
                size_bytes = os.path.getsize(pkl_file)
                size_gb = size_bytes / (1024 ** 3) if size_bytes is not None else None
                if size_gb is not None and size_gb > 0:
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
                        'size': size_gb,
                        'directory': str(file_path.parent)
                    })
            except Exception as e:
                print(f"Error checking {pkl_file}: {e}")
                continue
    
    return datasets

@app.get("/api/dataset/{dataset_path:path}")
def get_dataset_page(dataset_path: str, page: int = Query(0), page_size: int = Query(12)):
    """Load and return a specific page of a dataset."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load all examples (we'll optimize this further if needed)
        examples = load_dataset_from_pickle(dataset_path)
        total_count = len(examples)
        
        # Calculate pagination
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        # Only serialize the requested page
        page_examples = examples[start_idx:end_idx]
        serialized_examples = []
        for example in page_examples:
            serialized_examples.append(serialize_training_example(example))
        
        return {
            'examples': serialized_examples,
            'total_count': total_count,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_count + page_size - 1) // page_size,
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