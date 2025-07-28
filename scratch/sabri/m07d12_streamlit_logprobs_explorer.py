#!/usr/bin/env python3
"""
Streamlit app for exploring logprobs in synthesize.py output.

Usage:
    streamlit run streamlit_logprobs_explorer.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from transformers import AutoTokenizer
import streamlit.components.v1 as components
import json


@st.cache_data
def load_dataset(dataset_path: str):
    """Load the dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data['rows']


@st.cache_resource
def load_tokenizer(tokenizer_name: str):
    """Load and cache the tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        return tokenizer, None
    except Exception as e:
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
            return tokenizer, f"Failed to load {tokenizer_name}, using Qwen/Qwen2.5-0.5B: {str(e)}"
        except Exception as e2:
            return None, f"Failed to load any tokenizer: {str(e2)}"


def decode_token(tokenizer, token_id: int) -> str:
    """Decode a single token ID to text."""
    if tokenizer is None:
        return f"[ID:{token_id}]"
    
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"[ID:{token_id},ERR]"


def filter_examples_with_logprobs(training_examples):
    """Filter examples that have logprobs."""
    examples_with_logprobs = []
    for idx, example in enumerate(training_examples):
        for msg_idx, message in enumerate(example.messages):
            if message.top_logprobs is not None:
                examples_with_logprobs.append((idx, msg_idx, example, message))
    return examples_with_logprobs


def analyze_topk_for_probability_mass(all_topk_logprobs: List[np.ndarray], target_mass: float = 0.97):
    """Analyze how many top-k tokens are needed to capture target probability mass."""
    topk_needed = []
    
    for token_logprobs in all_topk_logprobs:
        # Convert to probabilities and sort in descending order
        probs = np.exp(token_logprobs)
        probs_sorted = np.sort(probs)[::-1]
        
        # Calculate cumulative probability mass
        cumulative_probs = np.cumsum(probs_sorted)
        
        # Find how many tokens needed to reach target mass
        k_needed = np.argmax(cumulative_probs >= target_mass) + 1
        
        # Handle edge case where we never reach target mass
        if cumulative_probs[-1] < target_mass:
            k_needed = len(probs_sorted)
        
        topk_needed.append(k_needed)
    
    return topk_needed


def create_distribution_plots(all_logprobs, topk_needed):
    """Create distribution plots using Plotly."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Log Probabilities Distribution', 'Probabilities Distribution', 
                       'Top-K Tokens for 97% Mass', 'Probability vs Top-K'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Convert to arrays
    logprobs_array = np.array(all_logprobs)
    probs_array = np.exp(logprobs_array)
    
    # Logprobs histogram
    fig.add_trace(
        go.Histogram(x=logprobs_array, nbinsx=50, name="Log Probabilities"),
        row=1, col=1
    )
    
    # Probabilities histogram
    fig.add_trace(
        go.Histogram(x=probs_array, nbinsx=50, name="Probabilities"),
        row=1, col=2
    )
    
    # Top-K histogram
    if topk_needed:
        nbins = int(min(50, max(topk_needed) - min(topk_needed) + 1))
        fig.add_trace(
            go.Histogram(x=topk_needed, nbinsx=nbins, name="Top-K Needed"),
            row=2, col=1
        )
    
    # Box plot for probabilities
    fig.add_trace(
        go.Box(y=probs_array, name="Probabilities"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig


def create_token_hover_component(message, tokenizer, max_tokens=500):
    """Create an interactive token hover component."""
    if message.top_logprobs is None:
        return None
    
    # Prepare token data for visualization
    token_data = []
    num_tokens = min(max_tokens, message.top_logprobs.logprobs.shape[0])
    
    for i in range(num_tokens):
        token_logprobs = message.top_logprobs.logprobs[i, :]
        token_ids = message.top_logprobs.token_ids[i, :]
        
        probs = np.exp(token_logprobs)
        sorted_indices = np.argsort(probs)[::-1]
        
        chosen_text = decode_token(tokenizer, token_ids[0])
        
        # Get top 5 alternatives
        alternatives = []
        for rank in range(min(5, len(token_ids))):
            idx = sorted_indices[rank]
            alt_text = decode_token(tokenizer, token_ids[idx])
            alternatives.append({
                'text': alt_text,
                'prob': float(probs[idx]),
                'logprob': float(token_logprobs[idx]),
                'rank': rank + 1
            })
        
        # Calculate probability mass metrics
        cumulative_prob = np.cumsum(np.sort(probs)[::-1])
        tokens_for_97 = int(np.argmax(cumulative_prob >= 0.97) + 1)
        
        token_data.append({
            'text': chosen_text,
            'position': i,
            'prob': float(probs[0]),
            'logprob': float(token_logprobs[0]),
            'alternatives': alternatives,
            'tokens_for_97': tokens_for_97,
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10)))
        })
    
    # Create the HTML component
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .token-container {{
                font-family: 'Courier New', monospace;
                line-height: 1.8;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                margin: 10px 0;
            }}
            
            .token {{
                display: inline-block;
                padding: 2px 4px;
                margin: 1px;
                border-radius: 3px;
                cursor: pointer;
                transition: all 0.2s ease;
                position: relative;
                border: 1px solid transparent;
            }}
            
            .token:hover {{
                background-color: #007bff;
                color: white;
                transform: scale(1.05);
                z-index: 1000;
                border: 1px solid #0056b3;
            }}
            
            .tooltip {{
                position: absolute;
                background-color: #333;
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-size: 12px;
                min-width: 300px;
                max-width: 400px;
                z-index: 1001;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                display: none;
                pointer-events: none;
            }}
            
            .tooltip-header {{
                font-weight: bold;
                margin-bottom: 8px;
                border-bottom: 1px solid #555;
                padding-bottom: 4px;
            }}
            
            .alternative {{
                margin: 2px 0;
                padding: 2px 0;
            }}
            
            .chosen {{
                color: #4CAF50;
                font-weight: bold;
            }}
            
            .prob-bar {{
                display: inline-block;
                height: 10px;
                background-color: #007bff;
                margin-left: 5px;
                border-radius: 2px;
            }}
            
            .stats {{
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #555;
                font-size: 11px;
                color: #ccc;
            }}
        </style>
    </head>
    <body>
        <div class="token-container" id="tokenContainer">
            <!-- Tokens will be inserted here -->
        </div>
        
        <div class="tooltip" id="tooltip">
            <div class="tooltip-content"></div>
        </div>
        
        <script>
            const tokenData = {json.dumps(token_data)};
            const container = document.getElementById('tokenContainer');
            const tooltip = document.getElementById('tooltip');
            
            // Create tokens
            tokenData.forEach((token, index) => {{
                const tokenEl = document.createElement('span');
                tokenEl.className = 'token';
                tokenEl.textContent = token.text;
                tokenEl.style.backgroundColor = getColorForProbability(token.prob, token.prob_25, token.prob_75);
                
                tokenEl.addEventListener('mouseenter', (e) => {{
                    showTooltip(e, token);
                }});
                
                tokenEl.addEventListener('mouseleave', () => {{
                    hideTooltip();
                }});
                
                tokenEl.addEventListener('mousemove', (e) => {{
                    updateTooltipPosition(e);
                }});
                
                container.appendChild(tokenEl);
            }});
            
            function getColorForProbability(prob, prob_25, prob_75) {{
                // Fallback to simple color scale if quartiles are invalid
                if (prob_25 === undefined || prob_75 === undefined || prob_25 === prob_75) {{
                    if (prob < 0.1) {{
                        return `rgba(239, 68, 68, 0.6)`;
                    }} else if (prob < 0.5) {{
                        return `rgba(245, 158, 11, 0.6)`;
                    }} else {{
                        return `rgba(34, 197, 94, 0.6)`;
                    }}
                }}
                
                // Dynamic color scale based on probability distribution
                const range = prob_75 - prob_25;
                const intensity = range > 0 ? Math.max(0.2, Math.min(1, (prob - prob_25) / range * 0.8 + 0.2)) : 0.6;
                
                if (prob <= prob_25) {{
                    // Low probability - red tones
                    return `rgba(239, 68, 68, ${{0.4 + intensity * 0.4}})`;
                }} else if (prob <= prob_75) {{
                    // Medium probability - yellow/orange tones
                    return `rgba(245, 158, 11, ${{0.4 + intensity * 0.4}})`;
                }} else {{
                    // High probability - green tones
                    return `rgba(34, 197, 94, ${{0.4 + intensity * 0.4}})`;
                }}
            }}
            
            function showTooltip(event, token) {{
                const content = tooltip.querySelector('.tooltip-content');
                
                let html = `
                    <div class="tooltip-header">
                        Token ${{token.position + 1}}: "${{token.text}}"
                    </div>
                    <div><strong>Probability:</strong> ${{token.prob.toFixed(4)}} (${{(token.prob * 100).toFixed(2)}}%)</div>
                    <div><strong>Log Probability:</strong> ${{token.logprob.toFixed(4)}}</div>
                    <div><strong>Entropy:</strong> ${{token.entropy.toFixed(3)}}</div>
                    <div><strong>Tokens for 97% mass:</strong> ${{token.tokens_for_97}}</div>
                    <br>
                    <div><strong>Top Alternatives:</strong></div>
                `;
                
                token.alternatives.forEach(alt => {{
                    const barWidth = Math.max(5, alt.prob * 100);
                    const className = alt.rank === 1 ? 'alternative chosen' : 'alternative';
                    html += `
                        <div class="${{className}}">
                            ${{alt.rank}}. "${{alt.text}}" 
                            <span class="prob-bar" style="width: ${{barWidth}}px;"></span>
                            ${{(alt.prob * 100).toFixed(2)}}%
                        </div>
                    `;
                }});
                
                content.innerHTML = html;
                tooltip.style.display = 'block';
                updateTooltipPosition(event);
            }}
            
            function hideTooltip() {{
                tooltip.style.display = 'none';
            }}
            
            function updateTooltipPosition(event) {{
                const rect = container.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();
                
                let left = event.clientX - rect.left + 10;
                let top = event.clientY - rect.top - tooltipRect.height - 10;
                
                // Keep tooltip within container bounds
                if (left + tooltipRect.width > rect.width) {{
                    left = event.clientX - rect.left - tooltipRect.width - 10;
                }}
                if (top < 0) {{
                    top = event.clientY - rect.top + 20;
                }}
                
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content


def create_token_hover_component_with_highlight(message, tokenizer, highlight_idx, max_tokens=50):
    """Create an interactive token hover component with a specific token highlighted."""
    if message.top_logprobs is None:
        return None
    
    # Prepare token data for visualization
    token_data = []
    num_tokens = min(max_tokens, message.top_logprobs.logprobs.shape[0])
    
    # Calculate probability distribution for dynamic color scaling
    all_probs = []
    for i in range(num_tokens):
        token_logprobs = message.top_logprobs.logprobs[i, :]
        probs = np.exp(token_logprobs)
        all_probs.append(float(probs[0]))
    
    # Calculate dynamic thresholds based on distribution
    prob_25 = float(np.percentile(all_probs, 25))
    prob_75 = float(np.percentile(all_probs, 75))
    
    # Ensure we have valid thresholds
    if prob_25 == prob_75:
        prob_25 = float(np.min(all_probs))
        prob_75 = float(np.max(all_probs))
    
    for i in range(num_tokens):
        token_logprobs = message.top_logprobs.logprobs[i, :]
        token_ids = message.top_logprobs.token_ids[i, :]
        
        probs = np.exp(token_logprobs)
        sorted_indices = np.argsort(probs)[::-1]
        
        chosen_text = decode_token(tokenizer, token_ids[0])
        
        # Get top 5 alternatives
        alternatives = []
        for rank in range(min(5, len(token_ids))):
            idx = sorted_indices[rank]
            alt_text = decode_token(tokenizer, token_ids[idx])
            alternatives.append({
                'text': alt_text,
                'prob': float(probs[idx]),
                'logprob': float(token_logprobs[idx]),
                'rank': rank + 1
            })
        
        # Calculate probability mass metrics
        cumulative_prob = np.cumsum(np.sort(probs)[::-1])
        tokens_for_97 = int(np.argmax(cumulative_prob >= 0.97) + 1)
        
        token_data.append({
            'text': chosen_text,
            'position': i,
            'prob': float(probs[0]),
            'logprob': float(token_logprobs[0]),
            'alternatives': alternatives,
            'tokens_for_97': tokens_for_97,
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10))),
            'highlighted': i == highlight_idx,
            'prob_25': prob_25,
            'prob_75': prob_75
        })
    
    # Create the HTML component with highlighting
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .token-container {{
                font-family: 'Courier New', monospace;
                line-height: 1.8;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                margin: 10px 0;
            }}
            
            .token {{
                display: inline-block;
                padding: 2px 4px;
                margin: 1px;
                border-radius: 3px;
                cursor: pointer;
                transition: all 0.2s ease;
                position: relative;
                border: 1px solid transparent;
            }}
            
            .token.highlighted {{
                background-color: #ff6b35 !important;
                color: white;
                border: 2px solid #e55a2b;
                font-weight: bold;
                animation: pulse 2s infinite;
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            
            .token:hover {{
                background-color: #007bff;
                color: white;
                transform: scale(1.05);
                z-index: 1000;
                border: 1px solid #0056b3;
            }}
            
            .token.highlighted:hover {{
                background-color: #ff6b35;
                border: 2px solid #e55a2b;
            }}
            
            .tooltip {{
                position: absolute;
                background-color: #333;
                color: white;
                padding: 12px;
                border-radius: 6px;
                font-size: 12px;
                min-width: 300px;
                max-width: 400px;
                z-index: 1001;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                display: none;
                pointer-events: none;
            }}
            
            .tooltip-header {{
                font-weight: bold;
                margin-bottom: 8px;
                border-bottom: 1px solid #555;
                padding-bottom: 4px;
            }}
            
            .highlighted-header {{
                background-color: #ff6b35;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                margin-bottom: 8px;
            }}
            
            .alternative {{
                margin: 2px 0;
                padding: 2px 0;
            }}
            
            .chosen {{
                color: #4CAF50;
                font-weight: bold;
            }}
            
            .prob-bar {{
                display: inline-block;
                height: 10px;
                background-color: #007bff;
                margin-left: 5px;
                border-radius: 2px;
            }}
        </style>
    </head>
    <body>
        <div class="token-container" id="tokenContainer">
            <!-- Tokens will be inserted here -->
        </div>
        
        <div class="tooltip" id="tooltip">
            <div class="tooltip-content"></div>
        </div>
        
        <script>
            const tokenData = {json.dumps(token_data)};
            const container = document.getElementById('tokenContainer');
            const tooltip = document.getElementById('tooltip');
            
            // Create tokens
            tokenData.forEach((token, index) => {{
                const tokenEl = document.createElement('span');
                tokenEl.className = token.highlighted ? 'token highlighted' : 'token';
                tokenEl.textContent = token.text;
                
                if (!token.highlighted) {{
                    tokenEl.style.backgroundColor = getColorForProbability(token.prob, token.prob_25, token.prob_75);
                }}
                
                tokenEl.addEventListener('mouseenter', (e) => {{
                    showTooltip(e, token);
                }});
                
                tokenEl.addEventListener('mouseleave', () => {{
                    hideTooltip();
                }});
                
                tokenEl.addEventListener('mousemove', (e) => {{
                    updateTooltipPosition(e);
                }});
                
                container.appendChild(tokenEl);
            }});
            
            function getColorForProbability(prob) {{
                const intensity = Math.max(0.1, Math.min(1, prob));
                if (prob < 0.1) {{
                    return `rgba(255, 99, 99, ${{0.3 + intensity * 0.4}})`;
                }} else if (prob < 0.5) {{
                    return `rgba(255, 206, 84, ${{0.3 + intensity * 0.4}})`;
                }} else {{
                    return `rgba(75, 192, 192, ${{0.3 + intensity * 0.4}})`;
                }}
            }}
            
            function showTooltip(event, token) {{
                const content = tooltip.querySelector('.tooltip-content');
                
                let html = '';
                if (token.highlighted) {{
                    html += `<div class="highlighted-header">🎯 SELECTED TOKEN</div>`;
                }}
                
                html += `
                    <div class="tooltip-header">
                        Token ${{token.position + 1}}: "${{token.text}}"
                    </div>
                    <div><strong>Probability:</strong> ${{token.prob.toFixed(4)}} (${{(token.prob * 100).toFixed(2)}}%)</div>
                    <div><strong>Log Probability:</strong> ${{token.logprob.toFixed(4)}}</div>
                    <div><strong>Entropy:</strong> ${{token.entropy.toFixed(3)}}</div>
                    <div><strong>Tokens for 97% mass:</strong> ${{token.tokens_for_97}}</div>
                    <br>
                    <div><strong>Top Alternatives:</strong></div>
                `;
                
                token.alternatives.forEach(alt => {{
                    const barWidth = Math.max(5, alt.prob * 100);
                    const className = alt.rank === 1 ? 'alternative chosen' : 'alternative';
                    html += `
                        <div class="${{className}}">
                            ${{alt.rank}}. "${{alt.text}}" 
                            <span class="prob-bar" style="width: ${{barWidth}}px;"></span>
                            ${{(alt.prob * 100).toFixed(2)}}%
                        </div>
                    `;
                }});
                
                content.innerHTML = html;
                tooltip.style.display = 'block';
                updateTooltipPosition(event);
            }}
            
            function hideTooltip() {{
                tooltip.style.display = 'none';
            }}
            
            function updateTooltipPosition(event) {{
                const rect = container.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();
                
                let left = event.clientX - rect.left + 10;
                let top = event.clientY - rect.top - tooltipRect.height - 10;
                
                if (left + tooltipRect.width > rect.width) {{
                    left = event.clientX - rect.left - tooltipRect.width - 10;
                }}
                if (top < 0) {{
                    top = event.clientY - rect.top + 20;
                }}
                
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content


def main():
    st.set_page_config(page_title="Logprobs Explorer", layout="wide")
    
    st.title("🔍 Logprobs Explorer")
    st.markdown("Explore logprobs from synthesize.py output")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Dataset path input
        dataset_path = st.text_input(
            "Dataset Path", 
            value="/home/sabri/code-memory/outputs/2025-07-06-13-37-13-gmail_synthesis/gmail_synthesis-0/artifact/dataset.pkl",
            help="Path to the dataset.pkl file"
        )
        
        # Tokenizer selection
        tokenizer_name = st.selectbox(
            "Tokenizer",
            ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4b"],
            index=0
        )
        
        # Load data button
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                try:
                    training_examples = load_dataset(dataset_path)
                    st.session_state.training_examples = training_examples
                    st.session_state.examples_with_logprobs = filter_examples_with_logprobs(training_examples)
                    st.success(f"Loaded {len(training_examples)} examples, {len(st.session_state.examples_with_logprobs)} with logprobs")
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
                    return
        
        # Load tokenizer
        if 'tokenizer' not in st.session_state or st.session_state.get('tokenizer_name') != tokenizer_name:
            with st.spinner("Loading tokenizer..."):
                tokenizer, error = load_tokenizer(tokenizer_name)
                st.session_state.tokenizer = tokenizer
                st.session_state.tokenizer_name = tokenizer_name
                if error:
                    st.warning(error)
                else:
                    st.success(f"Loaded tokenizer: {tokenizer_name}")
    
    # Main content
    if 'training_examples' not in st.session_state:
        st.info("Please load a dataset using the sidebar.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Example Explorer", "🎯 Token Detail", "📈 Analysis"])
    
    with tab1:
        st.header("Dataset Overview")
        
        examples_with_logprobs = st.session_state.examples_with_logprobs
        training_examples = st.session_state.training_examples
        
        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Examples", len(training_examples))
        with col2:
            st.metric("Examples with Logprobs", len(examples_with_logprobs))
        with col3:
            total_tokens = sum(msg.top_logprobs.logprobs.shape[0] for _, _, _, msg in examples_with_logprobs)
            st.metric("Total Tokens", total_tokens)
        with col4:
            if examples_with_logprobs:
                avg_topk = np.mean([msg.top_logprobs.logprobs.shape[1] for _, _, _, msg in examples_with_logprobs])
                st.metric("Avg Top-K Size", f"{avg_topk:.1f}")
        
        # Extract all logprobs for analysis
        all_logprobs = []
        all_topk_logprobs = []
        
        for _, _, _, message in examples_with_logprobs:
            if message.top_logprobs is not None:
                chosen_logprobs = message.top_logprobs.logprobs[:, 0]
                all_logprobs.extend(chosen_logprobs.tolist())
                
                for token_idx in range(message.top_logprobs.logprobs.shape[0]):
                    token_topk_logprobs = message.top_logprobs.logprobs[token_idx, :]
                    all_topk_logprobs.append(token_topk_logprobs)
        
        if all_logprobs:
            # Analyze top-k requirements
            topk_needed = analyze_topk_for_probability_mass(all_topk_logprobs)
            
            # Statistics
            st.subheader("Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Logprobs Statistics:**")
                st.write(f"Mean: {np.mean(all_logprobs):.4f}")
                st.write(f"Median: {np.median(all_logprobs):.4f}")
                st.write(f"Std: {np.std(all_logprobs):.4f}")
                st.write(f"Min: {np.min(all_logprobs):.4f}")
                st.write(f"Max: {np.max(all_logprobs):.4f}")
            
            with col2:
                st.write("**Top-K Analysis (97% mass):**")
                st.write(f"Mean tokens needed: {np.mean(topk_needed):.2f}")
                st.write(f"Median tokens needed: {np.median(topk_needed):.0f}")
                st.write(f"95th percentile: {np.percentile(topk_needed, 95):.0f}")
                st.write(f"Max tokens needed: {np.max(topk_needed)}")
            
            # Distribution plots
            st.subheader("Distribution Plots")
            fig = create_distribution_plots(all_logprobs, topk_needed)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Example Explorer")
        
        examples_with_logprobs = st.session_state.examples_with_logprobs
        
        if not examples_with_logprobs:
            st.info("No examples with logprobs found.")
            return
        
        # Example selection
        example_idx = st.selectbox(
            "Select Example",
            range(len(examples_with_logprobs)),
            format_func=lambda x: f"Example {examples_with_logprobs[x][0]} - Message {examples_with_logprobs[x][1]} ({examples_with_logprobs[x][3].role})"
        )
        
        orig_example_idx, message_idx, example, message = examples_with_logprobs[example_idx]
        
        # Show example details
        st.subheader("Example Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Original Example Index:** {orig_example_idx}")
            st.write(f"**Message Index:** {message_idx}")
            st.write(f"**Example Type:** {example.type}")
            st.write(f"**Message Role:** {message.role}")
        
        with col2:
            st.write(f"**Message Length:** {len(message.content)} chars")
            if message.top_logprobs is not None:
                st.write(f"**Tokens:** {message.top_logprobs.logprobs.shape[0]}")
                st.write(f"**Top-K Size:** {message.top_logprobs.logprobs.shape[1]}")
        
        # Show message content
        st.subheader("Message Content")
        st.text_area("Content", value=message.content, height=200, disabled=True)
        
        # Interactive token visualization
        if message.top_logprobs is not None:
            st.subheader("🎯 Interactive Token Visualization")
            st.markdown("**Hover over tokens to see logprobs and alternatives!**")
            
            # Controls for the visualization
            col1, col2 = st.columns(2)
            with col1:
                max_tokens = st.slider("Max tokens to display", 50, 1000, 500, step=50)
            with col2:
                st.markdown("**Color coding:** 🟢 High probability (top 25%), 🟡 Medium probability (25-75%), 🔴 Low probability (bottom 25%)")
            
            # Create and display the interactive component
            html_content = create_token_hover_component(message, st.session_state.get('tokenizer'), max_tokens)
            if html_content:
                components.html(html_content, height=400, scrolling=True)
            else:
                st.info("No logprobs available for visualization.")
        
        # Token overview
        if message.top_logprobs is not None:
            st.subheader("Token Overview")
            
            # Pagination controls
            tokens_per_page = st.slider("Tokens per page", 5, 20, 10)
            total_tokens = message.top_logprobs.logprobs.shape[0]
            total_pages = (total_tokens + tokens_per_page - 1) // tokens_per_page
            
            page = st.selectbox(f"Page (1-{total_pages})", range(1, total_pages + 1)) - 1
            
            start_idx = page * tokens_per_page
            end_idx = min(start_idx + tokens_per_page, total_tokens)
            
            # Show tokens for current page
            token_data = []
            tokenizer = st.session_state.get('tokenizer')
            
            for i in range(start_idx, end_idx):
                token_logprobs = message.top_logprobs.logprobs[i, :]
                token_ids = message.top_logprobs.token_ids[i, :]
                
                probs = np.exp(token_logprobs)
                chosen_text = decode_token(tokenizer, token_ids[0])
                
                token_data.append({
                    'Position': i + 1,
                    'Token': chosen_text,
                    'Token ID': token_ids[0],
                    'Probability': f"{probs[0]:.4f}",
                    'Logprob': f"{token_logprobs[0]:.4f}",
                    'Rank 2': decode_token(tokenizer, token_ids[np.argsort(probs)[-2]]) if len(token_ids) > 1 else "N/A",
                    'Rank 3': decode_token(tokenizer, token_ids[np.argsort(probs)[-3]]) if len(token_ids) > 2 else "N/A"
                })
            
            st.dataframe(pd.DataFrame(token_data), use_container_width=True)
    
    with tab3:
        st.header("Token Detail View")
        
        examples_with_logprobs = st.session_state.examples_with_logprobs
        
        if not examples_with_logprobs:
            st.info("No examples with logprobs found.")
            return
        
        # Example and token selection
        col1, col2 = st.columns(2)
        
        with col1:
            example_idx = st.selectbox(
                "Select Example",
                range(len(examples_with_logprobs)),
                format_func=lambda x: f"Example {examples_with_logprobs[x][0]} - Message {examples_with_logprobs[x][1]}",
                key="token_detail_example"
            )
        
        orig_example_idx, message_idx, example, message = examples_with_logprobs[example_idx]
        
        if message.top_logprobs is not None:
            with col2:
                token_idx = st.selectbox(
                    "Select Token",
                    range(message.top_logprobs.logprobs.shape[0]),
                    format_func=lambda x: f"Token {x + 1}",
                    key="token_detail_token"
                )
            
            # Show detailed token analysis
            token_logprobs = message.top_logprobs.logprobs[token_idx, :]
            token_ids = message.top_logprobs.token_ids[token_idx, :]
            
            probs = np.exp(token_logprobs)
            sorted_indices = np.argsort(probs)[::-1]
            
            tokenizer = st.session_state.get('tokenizer')
            
            st.subheader(f"Token {token_idx + 1} Analysis")
            
            # Chosen token info
            chosen_text = decode_token(tokenizer, token_ids[0])
            st.write(f"**Chosen Token:** '{chosen_text}' (ID: {token_ids[0]})")
            st.write(f"**Probability:** {probs[0]:.4f}")
            st.write(f"**Logprob:** {token_logprobs[0]:.4f}")
            
            # Show interactive visualization for context around the selected token
            st.subheader("🎯 Token in Context")
            st.markdown("**Interactive view with selected token highlighted:**")
            
            # Create a version with the selected token highlighted
            context_start = max(0, token_idx - 20)
            context_end = min(message.top_logprobs.logprobs.shape[0], token_idx + 21)
            
            # Create a temporary message object for the context
            class ContextMessage:
                def __init__(self, original_message, start_idx, end_idx, highlight_idx):
                    self.top_logprobs = type('TopLogprobs', (), {
                        'logprobs': original_message.top_logprobs.logprobs[start_idx:end_idx],
                        'token_ids': original_message.top_logprobs.token_ids[start_idx:end_idx]
                    })()
                    self.highlight_idx = highlight_idx - start_idx
            
            context_message = ContextMessage(message, context_start, context_end, token_idx)
            
            # Create custom HTML that highlights the selected token
            html_content = create_token_hover_component_with_highlight(
                context_message, tokenizer, context_message.highlight_idx, max_tokens=50
            )
            if html_content:
                components.html(html_content, height=150, scrolling=True)
            
            # Top alternatives table
            st.subheader("Top Alternatives")
            alternatives_data = []
            
            for rank in range(min(10, len(token_ids))):
                idx = sorted_indices[rank]
                token_id = token_ids[idx]
                prob = probs[idx]
                logprob = token_logprobs[idx]
                token_text = decode_token(tokenizer, token_id)
                
                alternatives_data.append({
                    'Rank': rank + 1,
                    'Token': token_text,
                    'Token ID': token_id,
                    'Probability': f"{prob:.4f}",
                    'Logprob': f"{logprob:.4f}",
                    'Chosen': "✓" if rank == 0 else ""
                })
            
            st.dataframe(pd.DataFrame(alternatives_data), use_container_width=True)
            
            # Probability mass analysis
            st.subheader("Probability Mass Analysis")
            cumulative_prob = np.cumsum(np.sort(probs)[::-1])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                tokens_95 = np.argmax(cumulative_prob >= 0.95) + 1
                st.metric("Tokens for 95% mass", tokens_95)
            with col2:
                tokens_97 = np.argmax(cumulative_prob >= 0.97) + 1
                st.metric("Tokens for 97% mass", tokens_97)
            with col3:
                tokens_99 = np.argmax(cumulative_prob >= 0.99) + 1
                st.metric("Tokens for 99% mass", tokens_99)
            
            # Cumulative probability plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_prob) + 1)),
                y=cumulative_prob,
                mode='lines+markers',
                name='Cumulative Probability'
            ))
            fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="95%")
            fig.add_hline(y=0.97, line_dash="dash", line_color="orange", annotation_text="97%")
            fig.add_hline(y=0.99, line_dash="dash", line_color="green", annotation_text="99%")
            fig.update_layout(
                title="Cumulative Probability Mass",
                xaxis_title="Number of Top-K Tokens",
                yaxis_title="Cumulative Probability",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Statistical Analysis")
        
        examples_with_logprobs = st.session_state.examples_with_logprobs
        
        if not examples_with_logprobs:
            st.info("No examples with logprobs found.")
            return
        
        # Per-example analysis
        st.subheader("Per-Example Analysis")
        
        example_stats = []
        for orig_idx, msg_idx, example, message in examples_with_logprobs:
            if message.top_logprobs is not None:
                chosen_logprobs = message.top_logprobs.logprobs[:, 0]
                example_stats.append({
                    'Example': orig_idx,
                    'Message': msg_idx,
                    'Role': message.role,
                    'Tokens': len(chosen_logprobs),
                    'Mean Logprob': np.mean(chosen_logprobs),
                    'Std Logprob': np.std(chosen_logprobs),
                    'Min Logprob': np.min(chosen_logprobs),
                    'Max Logprob': np.max(chosen_logprobs),
                    'Mean Probability': np.mean(np.exp(chosen_logprobs))
                })
        
        if example_stats:
            df = pd.DataFrame(example_stats)
            st.dataframe(df, use_container_width=True)
            
            # Correlation analysis
            st.subheader("Correlation Analysis")
            
            # Scatter plot: Mean logprob vs number of tokens
            fig = px.scatter(
                df, 
                x='Tokens', 
                y='Mean Logprob',
                color='Role',
                title='Mean Logprob vs Number of Tokens',
                hover_data=['Example', 'Message']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution by role
            st.subheader("Distribution by Role")
            fig = px.box(
                df, 
                x='Role', 
                y='Mean Logprob',
                title='Logprob Distribution by Message Role'
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()