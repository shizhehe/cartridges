from typing import Optional
import torch
import streamlit as st

import os
import wandb
from transformers import AutoTokenizer
from capsules.clients.together import TogetherClient
from capsules.train import TrainConfig, CacheAndModel
from capsules.kv_initialization.base import TrainableCache
from capsules.generation import generate as generate_text
from capsules.datasets import TEMPLATE

device = "cuda" if torch.cuda.is_available() else "cpu"



def load_model_and_cache_from_wandb(
    wandb_run_id: str,
    filename: str,
    device: str = "cuda",
) -> tuple[CacheAndModel, AutoTokenizer, TrainConfig]:
    st.write("⚙️ (1/7) Loading config...")
    train_config = TrainConfig.from_wandb(wandb_run_id, strict=False)

    st.write(f"🤖 (2/7) Loading base model...")
    if hasattr(train_config.model, "pretrained_model_name_or_path"):
        st.info(train_config.model.pretrained_model_name_or_path)
    model = train_config.model.instantiate().to(device)

    st.write(f"🔤 (3/7) Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer)

    # breakpoint()
    # st.write(f"💾 (4/7) Downloading cache...")
    # # Check cache path
    # cache_path = os.path.join(train_config.run_dir, filename)
    # if not os.path.exists(cache_path):
    #     cache_path = os.path.join(os.path.expanduser("~/wandb_cache"), filename)
    #     if not os.path.exists(cache_path):
    #         raise FileNotFoundError(f"Cache file {cache_path} not found.")
    #     else:
    #         cache_path = os.path.expanduser("~/wandb_cache")
    # else:
    #     cache_path = train_config.run_dir

    # out = wandb.restore(
    #     filename, run_path=wandb_run_id, root=cache_path,
    # )

    out = wandb.restore(
        filename, run_path=wandb_run_id, root=train_config.run_dir,
    )

    st.write(f"⏳ (5/7) Restoring cache...")
    st.info(out.name)
    cache = TrainableCache.from_pretrained(
        os.path.join(train_config.run_dir, filename), 
        device=device
    )
    st.write(f"🔤 (6/7) Loading context and ICL model...")
    context_str, icl_client = _load_icl_model(train_config, tokenizer)

    st.write(f"✅ (7/7) Cache loaded successfully!")
    return CacheAndModel(cache=cache, model=model), tokenizer, train_config, context_str, icl_client


def _load_icl_model(train_config: TrainConfig, tokenizer: AutoTokenizer):
    # FIXME: add more models
    mapping = {
        "meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct-Turbo"
    }
    model_name = train_config.model.pretrained_model_name_or_path
    if model_name in mapping:
        model_name = mapping[model_name]

        icl_client = TogetherClient.Config(
            model_name=model_name,
        ).instantiate()
    else:
        raise ValueError(f"Model {model_name} not found in mapping.")

    from capsules.datasets import ContextConvoDataset
    # dataset = ContextConvoDataset.load(
    #     train_config.dataset.data_sources[0][0],
    #     is_wandb=True,
    # )
    # context = dataset.context

    # FIXME: this is hardcoded to the length for llama 3b
    MAX_LENGTH = 110_000
    
    # context_str = tokenizer.decode(
    #     tokenizer.encode(context.to_string())[:MAX_LENGTH]
    # )
    return "todo", icl_client

def get_saved_cache_files(run_id: str) -> list[str]:
    import wandb
    import re


    api = wandb.Api()

    # Get all files from the run
    files = [file.name for file in api.run(run_id).files()]

    # Filter for cache-*.pt files using regex
    cache_files = [file for file in files if re.match(r"^cache-.*\.pt$", file)]

    # Extract the epoch or step number from each cache file and create a mapping
    file_to_step = {}
    for file in cache_files:
        # Try to match both epoch and step patterns
        match = re.search(r"cache-(epoch|step)(\d+)\.pt", file)
        if match:
            step_num = int(match.group(2))
            file_to_step[file] = step_num

    # Sort the files by their step/epoch number
    sorted_cache_files = sorted(cache_files, key=lambda x: file_to_step.get(x, 0), reverse=True)
    return sorted_cache_files
    

class ModelManager:
    def __init__(self):
        self.current_run_id: Optional[str] = None
        self.current_filename: Optional[str] = None
        
        self.model: Optional[CacheAndModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.current_run_id: Optional[str] = None
        self.train_config: Optional[TrainConfig] = None
        
        # for ICL generations
        self.context_str: Optional[str] = None
        self.icl_client: Optional[TogetherClient] = None

    def load_model(self, run_id: str, filename: str):
        if (run_id != self.current_run_id) or (filename != self.current_filename):
            self.model, self.tokenizer, self.train_config, self.context_str, self.icl_client = load_model_and_cache_from_wandb(
                run_id, filename, device=device
            )
            self.current_run_id = run_id
            self.current_filename = filename    
            
    def is_model_loaded(self):
        return self.model is not None and self.tokenizer is not None
    
tokenizer: Optional[AutoTokenizer] = None

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager: ModelManager = ModelManager() # type: ignore
model_manager: ModelManager = st.session_state.model_manager
if 'messages' not in st.session_state:
    st.session_state.messages: list[dict] = []

# Function to clear chat history
def clear_chat():
    st.session_state.messages = []


@st.dialog("Context", width="large")
def show_context():
    st.write(model_manager.context_str)


# Sidebar for model loading
with st.sidebar:
    run_id = st.text_input(
        "Wandb Run ID",
        value="hazy-research/capsules/ghm1tny6",
        help="You can find the run id in the overview tab of the wandb run."
    )
    error_message = None
    try:
        filename_options = get_saved_cache_files(run_id)
        if len(filename_options) == 0:
            filename_options = None
            error_message = "No cache files found for this run."
            
    except Exception as e:
        error_message = f"Error loading cache files: {str(e)}"
    
    if error_message is not None:
        st.error(error_message)
    else:
        checkpoint_option = st.selectbox(
            "Checkpoint",
            [f"last"] + filename_options,
            index=0
        )
        if checkpoint_option == "last":
            checkpoint_option = filename_options[0]

    if st.button("Load", type="primary", key="load_button", use_container_width=True, 
                #   icon="✨"
        ):
        with st.status("Loading model..."):
            try:
                model_manager.load_model(run_id, filename=checkpoint_option)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                raise e

    if model_manager.is_model_loaded():
        current_run_id = model_manager.current_run_id
        current_filename = model_manager.current_filename  
        st.success(f"Model from run `{current_run_id.split('/')[-1]}` and step `{current_filename}` loaded.", icon="✅")
    else:
        st.error(" No model is currently loaded.", icon="🚫")
    
    if model_manager.is_model_loaded():
        model_manager: ModelManager = model_manager
        tokenizer: AutoTokenizer = model_manager.tokenizer

    # Checkbox for ICL generation
    if model_manager.is_model_loaded():
        use_icl = st.checkbox(
            "Produce in-context generations for comparison",
            value=False,
            # help=""
        )
        # Button to show context
        if model_manager.is_model_loaded() and model_manager.context_str is not None:
            if st.button("Show Context", type="secondary", key="show_context_button"):
                show_context()

    st.write("---")
    max_new_tokens = st.slider(
        "Maximum number of tokens to generate per response",
        value=512,
        min_value=1,
        max_value=4096,
        step=1
    )

    if model_manager.is_model_loaded():
        st.write("---")
        st.write(f"**Configuration for run `{current_run_id.split('/')[-1]}`**")
        st.json(
            model_manager.train_config,
            expanded=False
        )

if not model_manager.is_model_loaded():
    st.error("⬅️ Please load a model in the sidebar to chat.")
else:
    st.title(f"Chat with `{model_manager.current_run_id.split('/')[-1]}`")


def display_assistant_message(
    message: dict,
):
    st.write(message["content"])
    if message["icl_response"] is not None:
        st.write("---")
        st.info("**In-context generation** \n\n" + message["icl_response"])
        

# Chat interface
if model_manager.is_model_loaded():
    if st.button("Clear Chat", type="primary", key="clear_chat_button", use_container_width=False):
        clear_chat()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_assistant_message(message)
            else:
                st.markdown(message["content"])

    
    if prompt := st.chat_input("Enter your message here..."):# Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
        


                    
                input_ids = model_manager.tokenizer.apply_chat_template(
                    conversation=st.session_state.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    template=TEMPLATE,
                    return_tensors="pt",
                ).to(device)
                
                cache_response = generate_text(
                    input_ids,
                    model_manager.model,
                    model_manager.tokenizer,
                    max_new_tokens=max_new_tokens
                )

                if use_icl:
                    assert model_manager.context_str is not None
                    icl_response = model_manager.icl_client.chat(
                        [
                            [{
                                "role": "system",
                                "content": f"You are an expert on the following context. \n\n{model_manager.context_str}"
                            }] + st.session_state.messages
                        ],
                        temperature=0.0,
                    ).samples[0].text
                else:
                    icl_response = None

                new_message = {
                    "role": "assistant",
                    "content": cache_response,
                    "icl_response": icl_response
                }

                display_assistant_message(new_message)
        
  
              
                
        # Add assistant response to chat history
        st.session_state.messages.append(new_message)