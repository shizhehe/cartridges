"""
Streamlit app for visualizing Code Memory synthesis results.

Run with: streamlit run scratch/sabri/m06d22_viz.py

This visualizer works with or without plotly/streamlit dependencies by providing fallback HTML output.
"""

import pickle
import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import structures
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from cartridges.structs import TrainingExample

# Try to import streamlit and plotly, fall back to basic HTML if not available
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

def load_synthesis_datasets(output_dir: str = "output") -> Dict[str, Any]:
    """Load all synthesis dataset.pkl files from output directories.
    Only returns datasets that have non-empty rows."""
    datasets = {}
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return datasets
    
    print(f"Scanning output directory: {output_dir}")
    try:
        dir_entries = os.listdir(output_dir)
        print(f"Found {len(dir_entries)} entries in output directory")
    except Exception as e:
        print(f"Error reading output directory: {e}")
        return datasets
    
    for dir_name in dir_entries:
        dir_path = os.path.join(output_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        print(f"Processing directory: {dir_name}")
        
        try:
            # Look for dataset.pkl files in various locations
            potential_paths = [
                # Direct path: output/dir_name/dataset.pkl
                os.path.join(dir_path, "dataset.pkl"),
                # Artifact path: output/dir_name/artifact/dataset.pkl  
                os.path.join(dir_path, "artifact", "dataset.pkl"),
            ]
            
            # Also check subfolders for artifact/dataset.pkl pattern
            try:
                for subfolder in os.listdir(dir_path):
                    subfolder_path = os.path.join(dir_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        potential_paths.append(os.path.join(subfolder_path, "artifact", "dataset.pkl"))
                        potential_paths.append(os.path.join(subfolder_path, "dataset.pkl"))
            except Exception as e:
                print(f"Error listing subfolders in {dir_path}: {e}")
            
            # Try to load from the first valid path found
            dataset_loaded = False
            for dataset_path in potential_paths:
                if os.path.exists(dataset_path):
                    print(f"Found dataset at: {dataset_path}")
                    try:
                        with open(dataset_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        # Validate the data structure
                        if isinstance(data, dict) and "rows" in data:
                            if data["rows"] and len(data["rows"]) > 0:
                                datasets[dir_name] = data
                                print(f"Successfully loaded {len(data['rows'])} examples from {dir_name}")
                                dataset_loaded = True
                                break
                            else:
                                print(f"Dataset {dir_name} has empty rows")
                        else:
                            print(f"Dataset {dir_name} has invalid structure (missing 'rows' key)")
                    except Exception as e:
                        print(f"Failed to load {dataset_path}: {e}")
            
            if not dataset_loaded:
                print(f"No valid dataset found for directory: {dir_name}")
                
        except Exception as e:
            print(f"Error processing directory {dir_name}: {e}")
    
    print(f"Successfully loaded {len(datasets)} datasets: {list(datasets.keys())}")
    return datasets





# Streamlit interface (when available)
if HAS_STREAMLIT:
    
    st.set_page_config(
        page_title="Code Memory Synthesis Visualization",
        page_icon="🧠",
        layout="wide"
    )
    
    def display_training_example(example: TrainingExample, idx: int):
        """Display a single training example in an expandable format."""
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Output Tokens", example.num_output_tokens)
        with col2:
            st.metric("Messages", len(example.messages))
        with col3:
            st.metric("Type", example.type)
        with col4:
            if example.metadata and 'seed_prompt' in example.metadata:
                st.metric("Has Seed", "✅")
            else:
                st.metric("Has Seed", "❌")
        with col5:
            # Count total tool calls from metadata
            total_tool_calls = 0
            if example.metadata and 'tool_calls' in example.metadata:
                total_tool_calls = len(example.metadata['tool_calls'])
            st.metric("Tool Calls", total_tool_calls)
        
        # Display tool calls from metadata (if any)
        if example.metadata and 'tool_calls' in example.metadata and example.metadata['tool_calls']:
            st.subheader("🔧 Tool Calls Summary")
            
            # Group tool calls by name for summary
            tool_summary = {}
            for tool_call in example.metadata['tool_calls']:
                tool_name = tool_call.get('name', 'Unknown')
                if tool_name not in tool_summary:
                    tool_summary[tool_name] = {'count': 0, 'success': 0, 'failed': 0}
                tool_summary[tool_name]['count'] += 1
                if tool_call.get('success', False):
                    tool_summary[tool_name]['success'] += 1
                else:
                    tool_summary[tool_name]['failed'] += 1
            
            # Display tool summary in columns
            tool_cols = st.columns(min(len(tool_summary), 4))
            for idx, (tool_name, stats) in enumerate(tool_summary.items()):
                with tool_cols[idx % 4]:
                    st.metric(
                        f"🔧 {tool_name}",
                        f"{stats['count']} calls",
                        f"✅{stats['success']} ❌{stats['failed']}"
                    )
            
            # Detailed tool calls
            with st.expander("📋 Detailed Tool Calls"):
                for i, tool_call in enumerate(example.metadata['tool_calls']):
                    tool_name = tool_call.get('name', 'Unknown')
                    success_icon = "✅" if tool_call.get('success', False) else "❌"
                    
                    with st.expander(f"{success_icon} {tool_name} - Call {i+1}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Input:**")
                            if 'input' in tool_call:
                                st.code(json.dumps(tool_call['input'], indent=2), language="json")
                            
                            if 'raw_request' in tool_call:
                                st.markdown("**Raw Request:**")
                                st.code(tool_call['raw_request'], language="text")
                        
                        with col2:
                            st.markdown("**Output:**")
                            if 'output' in tool_call:
                                output_text = str(tool_call['output'])
                                # Truncate very long outputs
                                if len(output_text) > 1000:
                                    st.code(output_text[:1000] + "\n... [truncated]", language="text")
                                    with st.expander("Show full output"):
                                        st.code(output_text, language="text")
                                else:
                                    st.code(output_text, language="text")
                            
                            st.markdown(f"**Success:** {'✅ Yes' if tool_call.get('success', False) else '❌ No'}")
        
        # Display conversation
        st.subheader("💬 Conversation")
        for i, message in enumerate(example.messages):
            role_icon = {"system": "🔧", "user": "👤", "assistant": "🤖"}
            
            with st.expander(f"{role_icon.get(message.role, '💬')} {message.role.capitalize()} - Message {i+1}"):
                # Check for tool calls in content (legacy format)
                has_tool_calls = "<tool_call>" in message.content
                
                if has_tool_calls:
                    st.markdown("**Content with Tool Calls:**")
                    # Show content with tool calls highlighted
                    content_parts = message.content.split("<tool_call>")
                    for part_idx, part in enumerate(content_parts):
                        if part_idx == 0:
                            if part.strip():
                                st.markdown(part)
                        else:
                            st.markdown("---")
                            st.markdown("🔧 **Tool Call:**")
                            if "<tool_input>" in part:
                                tool_input = part.split("<tool_input>")[1].split("</tool_input>")[0]
                                st.code(tool_input, language="json")
                            if "<tool_output>" in part:
                                tool_output = part.split("<tool_output>")[1].split("</tool_output>")[0]
                                st.code(tool_output, language="text")
                            # Show any remaining content after tool call
                            remaining = part.split("</tool_call>")[-1]
                            if remaining.strip():
                                st.markdown(remaining)
                else:
                    st.markdown(message.content)
        
        # System prompt
        if example.system_prompt:
            with st.expander("🔧 System Prompt"):
                st.markdown(example.system_prompt)
        
        # Metadata
        if example.metadata:
            with st.expander("📋 Metadata"):
                st.json(example.metadata)
        
        # Token analysis if available
        if example.token_ids is not None and example.top_logprob_logprobs is not None:
            with st.expander("📊 Token Analysis"):
                st.write(f"Token IDs shape: {example.token_ids.shape}")
                st.write(f"Top logprobs shape: {example.top_logprob_logprobs.shape}")
                
                # Simple token statistics without plotly
                if len(example.top_logprob_logprobs.shape) >= 2:
                    avg_logprobs = np.mean(example.top_logprob_logprobs, axis=1)
                    st.write(f"Average logprob: {np.mean(avg_logprobs):.3f}")
                    st.write(f"Logprob std: {np.std(avg_logprobs):.3f}")

    

    def main():
        st.title("🧠 Code Memory Synthesis Visualization")
        
        # Allow user to specify output directory
        st.sidebar.title("Configuration")
        
        # Try different common output directories
        possible_dirs = ["output", ".", "../output", "../../output"]
        if "MEMORY_OUTPUT_DIR" in os.environ:
            possible_dirs.insert(0, os.environ["MEMORY_OUTPUT_DIR"])
        
        output_dir = st.sidebar.selectbox(
            "Output Directory",
            possible_dirs,
            help="Select the directory containing synthesis results"
        )
        
        # Load data
        @st.cache_data
        def load_all_data(output_dir: str):
            synthesis_data = load_synthesis_datasets(output_dir)
            return synthesis_data
        
        synthesis_datasets = load_all_data(output_dir)
        
        # Add refresh button to reload datasets
        if st.sidebar.button("🔄 Refresh Datasets"):
            st.cache_data.clear()
            st.rerun()
        
        st.sidebar.title("Navigation")
        
        if not synthesis_datasets:
            st.warning("No synthesis datasets found in the output directory.")
            st.info("Make sure you have run synthesis experiments and the output directory contains dataset.pkl files.")
            
            # Show helpful debugging info
            with st.expander("🔍 Debugging Information"):
                st.code(f"Looking for datasets in: {os.path.abspath('output')}")
                if os.path.exists("output"):
                    try:
                        dirs = [d for d in os.listdir("output") if os.path.isdir(os.path.join("output", d))]
                        st.write(f"Found directories: {dirs}")
                        for d in dirs[:5]:  # Show first 5 directories
                            dir_path = os.path.join("output", d)
                            try:
                                contents = os.listdir(dir_path)
                                st.write(f"Contents of {d}: {contents[:10]}...")  # Show first 10 items
                            except Exception as e:
                                st.write(f"Error reading {d}: {e}")
                    except Exception as e:
                        st.write(f"Error listing output directory: {e}")
                else:
                    st.write("Output directory does not exist")
            return

        # Dataset selector in the sidebar
        selected_dataset = st.sidebar.selectbox(
            "Select Dataset",
            sorted(list(synthesis_datasets.keys()), reverse=True)
        )
        
        st.header("🔍 Synthesis Dataset Details")
        
        if selected_dataset and 'rows' in synthesis_datasets[selected_dataset]:
            examples = synthesis_datasets[selected_dataset]['rows']
            
            # Example selector with tool call info
            def format_example(x):
                example = examples[x]
                tool_count = 0
                if example.metadata and 'tool_calls' in example.metadata:
                    tool_count = len(example.metadata['tool_calls'])
                return f"Example {x+1} ({example.type}, {example.num_output_tokens} tokens, {tool_count} tools)"
            
            example_idx = st.selectbox(
                "Select Example",
                range(len(examples)),
                format_func=format_example
            )
            
            st.subheader(f"Example {example_idx + 1}")
            display_training_example(examples[example_idx], example_idx)

    def test_data_loading():
        """Test function to verify data loading without Streamlit."""
        print("Testing data loading...")
        
        # Try different directories
        possible_dirs = ["output", ".", "../output", "../../output"]
        if "MEMORY_OUTPUT_DIR" in os.environ:
            possible_dirs.insert(0, os.environ["MEMORY_OUTPUT_DIR"])
        
        for output_dir in possible_dirs:
            print(f"\nTrying directory: {output_dir}")
            datasets = load_synthesis_datasets(output_dir)
            if datasets:
                print(f"Found {len(datasets)} datasets: {list(datasets.keys())}")
                for name, data in datasets.items():
                    if "rows" in data:
                        print(f"  {name}: {len(data['rows'])} examples")
                return datasets
        
        print("No datasets found in any directory")
        return {}

    if __name__ == "__main__":
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            test_data_loading()
        else:
            main()

