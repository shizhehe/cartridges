# cartridges/wandb_peft.py
from __future__ import annotations
import os, re, shutil
from pathlib import Path
from typing import Optional, Literal

import torch
from transformers import PreTrainedModel, AutoModelForCausalLM
from peft import PeftModel

import wandb

__all__ = [
    "list_peft_steps_in_run",
    "download_peft_from_run",
    "download_peft_from_artifact",
    "load_peft_into_model",
]

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_peft_steps_in_run(full_run_path: str) -> list[int]:
    """
    Inspect a W&B run's files and return available peft-step numbers,
    assuming you logged PEFT folders like 'peft-step{K}/adapter_model.safetensors'.
    """
    api = wandb.Api()
    run = api.run(full_run_path)
    steps = set()
    for f in run.files():
        # Expect names like: 'peft-step123/adapter_model.safetensors'
        m = re.match(r"^peft-step(\d+)/(adapter_model\.safetensors|adapter_config\.json)$", f.name)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)

def download_peft_from_run(
    full_run_path: str,
    dest_dir: Path,
    step: Optional[int] = None,
) -> Path:
    """
    Download a PEFT adapter folder from a W&B run (logged via wandb.save).
    Returns the local directory containing adapter files.
    If step is None, grabs the latest available step.
    """
    steps = list_peft_steps_in_run(full_run_path)
    if not steps:
        raise ValueError(f"No PEFT adapter files found in run: {full_run_path}")

    if step is None:
        step = steps[-1]

    # pull both adapter files
    dest = _ensure_dir(dest_dir / f"peft-step{step}")
    for fname in ("adapter_model.safetensors", "adapter_config.json"):
        wandb.restore(
            root=str(dest_dir),
            run_path=full_run_path,
            filename=f"peft-step{step}/{fname}",
        )
        # wandb.restore puts the exact relative path under dest_dir; move into dest
        src = dest_dir / f"peft-step{step}" / fname
        if not src.exists():
            # Some older runs might upload without the folder prefix; try to fetch plain files.
            alt = dest_dir / fname
            if not alt.exists():
                raise FileNotFoundError(f"Missing {fname} after restore from {full_run_path}")
            src = alt
        shutil.move(str(src), str(dest / fname))

    return dest

def download_peft_from_artifact(
    artifact_ref: str,  # e.g. "entity/project/llama3-lora-adapter:latest"
    dest_dir: Path,
) -> Path:
    """
    Download a PEFT adapter stored as a W&B Artifact (recommended).
    Expects the artifact directory to contain adapter_model.safetensors + adapter_config.json.
    Returns the local directory with those files.
    """
    run = wandb.run
    if run is None:
        raise ValueError("Wandb run is not initialized")
    art = run.use_artifact(artifact_ref, type="model")
    src_dir = Path(art.download())
    dest = _ensure_dir(dest_dir / Path(artifact_ref.split(":")[0]).name)

    for fname in ("adapter_model.safetensors", "adapter_config.json"):
        src = src_dir / fname
        if not src.exists():
            # allow nested (some users store under 'peft/' etc.)
            cands = list(src_dir.rglob(fname))
            if not cands:
                raise FileNotFoundError(f"{fname} not found in artifact {artifact_ref}")
            src = cands[0]
        shutil.copy2(src, dest / fname)

    return dest

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
