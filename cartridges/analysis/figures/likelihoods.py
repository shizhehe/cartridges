from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer

def visualize_text_likelihoods(
    row: Union[pd.Series, dict],
    log_probs: str = "token_logprobs",
    tokens: str="labels_tokens",
    tokenizer: Optional[PreTrainedTokenizer]=None,
    show_probs: bool=False,
):
    if isinstance(row, dict):
        row = pd.Series(row)
    
    floats = row[log_probs]#[1:]
    if show_probs:
        floats = np.exp(floats)
    if tokenizer is not None:
        strings = [tokenizer.decode(x) for x in row[tokens]] #[1:]]
    else:
        strings = [str(x) for x in row[tokens]] #[1:]]


    # Normalize floats for color mapping
    if show_probs:
        norm = plt.Normalize(0, 1)
    else:
        norm = plt.Normalize(-10, 0)
    colors = plt.cm.viridis(norm(floats))

    # Define the plot dimensions and layout
    num_columns = 20
    num_rows = (len(strings) + num_columns - 1) // num_columns  # Calculate the number of rows needed

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, num_rows * 0.75))
    ax.set_xlim(0, num_columns)
    ax.set_ylim(0, num_rows)
    ax.axis('off')

    # Plot each float value with a colored background and the corresponding string underneath
    for idx, (string, value) in enumerate(zip(strings, floats)):
        row = idx // num_columns
        col = idx % num_columns
        color = colors[idx]
        
        # Position for the float value
        x_pos = col + 0.5
        y_pos = num_rows - row - 1
        
        # Plot the float value with a colored background
        ax.text(x_pos, y_pos, f'{value:.3f}', fontsize=10, ha='center', va='center', color='black' if value > -4 else 'white',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
        
        # Plot the corresponding string underneath the float value
        ax.text(x_pos, y_pos - 0.4, string, fontsize=6, ha='center', va='center')

    return fig
