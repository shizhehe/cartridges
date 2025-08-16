#!/usr/bin/env python3
"""
Backend API server for the dataset visualization app.
Provides endpoints for dataset discovery and loading.
"""

import os
import pickle
import glob
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from cartridges.structs import read_conversations

app = FastAPI(title="Dataset Visualization API", version="1.0.0")

# Configuration from environment variables
CORS_ENABLED = os.getenv('CORS_ENABLED', 'true').lower() == 'true'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
RELOAD = os.getenv('RELOAD', 'false').lower() == 'true'

# CORS middleware configuration
if CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from pickle or parquet file."""
    try:
        # Use the new read_conversations function that handles both formats
        conversations = read_conversations(file_path)
        return conversations
    except ImportError as e:
        print(f"Missing dependency for {file_path}: {e}")
        print("Please install required dependencies: pip install pyarrow pandas")
        return []
    except Exception as e:
        print(f"Error loading dataset {file_path}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        # Fallback to old pickle loading for backwards compatibility
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
        except Exception as e2:
            print(f"Error loading dataset with fallback {file_path}: {e2}")
            return []

def serialize_training_example(example) -> Dict[str, Any]:
    """Convert TrainingExample to JSON-serializable format."""
    try:
        messages = []
        for msg in example.messages:
            token_ids = msg.token_ids.tolist() if hasattr(msg.token_ids, "tolist") else msg.token_ids
            message_data = {
                'content': msg.content,
                'role': msg.role,
                'token_ids': token_ids,
                'top_logprobs': None
            }
            
            # Handle logprobs if they exist
            # if hasattr(msg, 'top_logprobs') and msg.top_logprobs is not None:
            #     top_logprobs = msg.top_logprobs.reconstruct()
            #     message_data['top_logprobs'] = {
            #         'logprobs': top_logprobs.logprobs.tolist() if hasattr(top_logprobs.logprobs, 'tolist') else top_logprobs.logprobs,
            #         'token_ids': top_logprobs.token_ids.tolist() if hasattr(top_logprobs.token_ids, 'tolist') else top_logprobs.token_ids
            #     }
            
            messages.append(message_data)
        
        # Serialize metadata to handle numpy arrays and other non-serializable objects
        serialized_metadata = {}
        if example.metadata:
            for key, value in example.metadata.items():
                try:
                    # Handle numpy arrays
                    if hasattr(value, 'tolist'):
                        serialized_metadata[key] = value.tolist()
                    # Handle other numpy types
                    elif hasattr(value, 'item'):
                        serialized_metadata[key] = value.item()
                    # Handle regular serializable objects
                    else:
                        # Test if it's JSON serializable
                        import json
                        json.dumps(value)
                        serialized_metadata[key] = value
                except (TypeError, ValueError):
                    # Convert to string if not serializable
                    serialized_metadata[key] = str(value)
        
        
        return {
            'messages': messages,
            'system_prompt': example.system_prompt,
            'type': example.type,
            'metadata': serialized_metadata
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
    """Quickly check if a file is a valid dataset and return approximate size."""
    try:
        # For parquet files, we can get row count without loading all data
        if file_path.endswith('.parquet'):
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                return table.num_rows
            except ImportError:
                print(f"Warning: pyarrow not available, falling back to loading full dataset for {file_path}")
                # Fallback to loading full dataset
                conversations = load_dataset(file_path)
                return len(conversations)
        
        # For pickle files, we need to load to check
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
            
        # Find all .pkl and .parquet files recursively
        pkl_files = glob.glob(os.path.join(search_path, '**/*.pkl'), recursive=True)
        parquet_files = glob.glob(os.path.join(search_path, '**/*.parquet'), recursive=True)
        all_files = pkl_files + parquet_files
        
        for file_path in all_files:
            try:
                # Quick check if it's a valid dataset
                size_bytes = os.path.getsize(file_path)
                size_gb = size_bytes / (1024 ** 3) if size_bytes is not None else None
                if size_gb is not None and size_gb > 0:
                    file_obj = Path(file_path)
                    dataset_name = file_obj.stem
                    
                    # Calculate relative path from search_path
                    try:
                        relative_path = str(file_obj.relative_to(search_path))
                    except ValueError:
                        # If relative_to fails, just use the filename
                        relative_path = file_obj.name
                    
                    datasets.append({
                        'name': dataset_name,
                        'path': file_path,
                        'relative_path': relative_path,
                        'size': size_gb,
                        'directory': str(file_obj.parent)
                    })
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
                continue
    
    # Sort datasets by relative path for consistent ordering
    datasets.sort(key=lambda d: d['relative_path'])
    
    return datasets

@app.get("/api/dataset/{dataset_path:path}/info")
def get_dataset_info(dataset_path: str):
    """Get dataset metadata without loading examples."""
    try:
        # Decode the path
        import urllib.parse
        dataset_path = urllib.parse.unquote(dataset_path)
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get total count efficiently
        total_count = quick_check_dataset(dataset_path)
        if total_count is None:
            # Fallback: load dataset to get count
            examples = load_dataset(dataset_path)
            total_count = len(examples)
        
        return {
            'path': dataset_path,
            'total_count': total_count,
            'file_size': os.path.getsize(dataset_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        t0 = time.time()
        examples = load_dataset(dataset_path)
        print(f"Loaded dataset in {time.time() - t0} seconds")
        total_count = len(examples)
        
        # Calculate pagination
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        # Only serialize the requested page
        t0 = time.time()
        page_examples = examples[start_idx:end_idx]
        serialized_examples = []
        for example in page_examples:
            serialized_examples.append(serialize_training_example(example))
        print(f"Serialized examples in {time.time() - t0} seconds")
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

@app.post("/api/dataset/config")
def get_dataset_config(request: Dict[str, Any]):
    """Get the SynthesizeConfig for a dataset if it exists."""
    try:
        dataset_path = request.get('dataset_path')
        if not dataset_path:
            raise HTTPException(status_code=400, detail="dataset_path is required")
        
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Look for config.yaml in the same directory as the dataset
        dataset_dir = os.path.dirname(dataset_path)
        config_path = os.path.join(dataset_dir, 'config.yaml')
        
        if not os.path.exists(config_path):
            # Also try the parent directory (common pattern)
            parent_config_path = os.path.join(os.path.dirname(dataset_dir), 'config.yaml')
            if os.path.exists(parent_config_path):
                config_path = parent_config_path
            else:
                return {'config': None, 'path': None}
        
        # Load the YAML config
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return {
            'config': config_data,
            'path': config_path,
            'exists': True
        }
    
    except Exception as e:
        print(f"Error loading config: {e}")
        return {'config': None, 'path': None, 'exists': False, 'error': str(e)}

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

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Visualization Server')
    parser.add_argument('--host', default=HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=PORT, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', default=RELOAD, help='Enable auto-reload')
    parser.add_argument('--no-cors', action='store_true', help='Disable CORS')
    parser.add_argument('--cors-origins', default=','.join(CORS_ORIGINS), 
                       help='Comma-separated list of allowed CORS origins')
    
    args = parser.parse_args()
    
    # Override configuration with CLI args
    cors_enabled = not args.no_cors and CORS_ENABLED
    cors_origins = args.cors_origins.split(',') if args.cors_origins != ','.join(CORS_ORIGINS) else CORS_ORIGINS
    
    # Configure CORS middleware
    if cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"CORS enabled: {cors_enabled}")
    if cors_enabled:
        print(f"CORS origins: {cors_origins}")
    
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)