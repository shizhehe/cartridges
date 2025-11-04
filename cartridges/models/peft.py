from pathlib import Path
from typing import Optional, Literal

from transformers import PreTrainedModel
from peft import PeftModel

import wandb

def load_peft_into_model(
    base_model: PreTrainedModel,
    peft_dir: Path,
) -> PeftModel:
    """
    Given a base HF model and a local directory with PEFT adapter files,
    return a PeftModel with the adapter loaded.
    """
    if not (peft_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {peft_dir}")
    if not (peft_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found in {peft_dir}")

    model = PeftModel.from_pretrained(base_model, str(peft_dir))
    return model
