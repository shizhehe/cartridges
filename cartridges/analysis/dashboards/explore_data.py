import sys
import os
from typing import List, Dict, Any, Optional
import json

# Import the load_dataset function
from capsules.generate.run import ContextConvoDataset
import streamlit as st
import pandas as pd
import wandb

from capsules.generate.structs import Context, ContextConvo

@st.cache_data
def get_dataset_options():
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name="dataset", project="hazy-research/capsules").collections()
    ]
    return [artifact.name for collection in collections for artifact in collection.artifacts() if artifact.type == "dataset"]

st.set_page_config(
    page_title="Dataset Visualizer",
    page_icon="📊",
    layout="wide",
)

st.title("`ContextConvo` Visualizer")

# Input section
with st.sidebar:

    # Connect to W&B API and list available datasets
    try:
        dataset_options = get_dataset_options()

        # # Filter for dataset type artifacts
        # dataset_options = [f"{artifact.name}:{artifact.version}" for artifact in artifacts]
        
        if not dataset_options:
            dataset_options = ["No datasets found"]
            default_dataset = "No datasets found"
        else:
            default_dataset = dataset_options[0]
        
        dataset_path = st.sidebar.selectbox(
            "Select Dataset", 
            options=dataset_options,
            index=0
        )
        st.write("Press `c` to clear the cache and refresh the list of datasets.")
        st.write("---")
        
        # Construct full path
        if dataset_path != "No datasets found":
            dataset_path = f"hazy-research/capsules/{dataset_path}"
            st.write("*Selected Artifact Path*")
            st.code(dataset_path)

            
        
        is_wandb = True
    except Exception as e:
        st.sidebar.error(f"Error connecting to W&B: {e}")
        dataset_path = st.sidebar.text_input("Dataset Path", "hazy-research/capsules/dataset:v0")
        is_wandb = st.sidebar.checkbox("Is Weights & Biases Dataset", True)


@st.dialog(title="Document Content", width="large")
def view_context(context: Context):
    st.write(context.to_string())


# Cache the dataset loading function to improve performance
@st.cache_data
def load_cached_dataset(dataset_path, is_wandb) -> ContextConvoDataset:
    return ContextConvoDataset.load(
        dataset_path, is_wandb=is_wandb, validate_rows=False
    )

# Load the dataset when the path is provided
if dataset_path:
    try:
        dataset: ContextConvoDataset = load_cached_dataset(dataset_path, is_wandb)
        
        st.sidebar.write("## Dataset Configuration")
        st.sidebar.json(dataset.config.model_dump())

        # st.write(dataset)
        # Create a button to open the document in a modal
        if st.button("View Document"):
            view_context(dataset.context)

        # We didn't validate the rows, so we to actually conver them
        data = [
            {
                "question": row["messages"][0]["content"], 
                "answer": row["messages"][1]["content"],
            }
            for row in dataset.rows
        ]
        
        # Create a dataframe with selection callback
        event = st.dataframe(
            data,
            use_container_width=True,
            selection_mode="single-row",
            key='data',
            on_select="rerun"
        )
    
        # Display selected row in chat interface
        if len(event.selection["rows"]) > 0:
            selected_idx = event.selection["rows"][0]
            selected_row = ContextConvo.model_validate(dataset.rows[selected_idx])
            
            st.markdown("### Selected Conversation")
            
            # Create a chat message for each message in the conversation
            for message in selected_row.messages:
                with st.chat_message(message.role):
                    st.markdown(message.content)
                    
                    # If there's sample data available, show it in an expander
                    if message.sample:
                        with st.expander("Show sample data"):
                            st.json(message.sample.model_dump())

        else:
            st.write("*Select a row above to view the conversation.*")
            

            
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.info("Please enter a dataset path to visualize the data.")

