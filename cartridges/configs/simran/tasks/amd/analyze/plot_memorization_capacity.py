import wandb
from typing import Optional, List
from capsules.utils.wandb import fetch_wandb_runs
import matplotlib.pyplot as plt

# Define the different NTP experiment runs and their corresponding token sizes
ntp_experiments = {
    "NTP (4096)": ("hazy-research/capsules/rnw4tecy", 4096),
    "NTP (2048)": ("hazy-research/capsules/bg6jxy3z", 2048),
    # "NTP (1024)": ("hazy-research/capsules/68z7e3k0", 1024),
    "NTP (512)": ("hazy-research/capsules/wpffz7ot", 512),
    "NTP (64)": ("hazy-research/capsules/u5e56ajk", 64),
    "NTP (16)": ("hazy-research/capsules/4qaa7hyq", 16),
}

def fetch_runs(wandb_run_ids: Optional[List[str]]):
    if wandb_run_ids is not None:
        wandb_run_ids = [x.split("/")[-1] for x in wandb_run_ids]
    else:
        wandb_run_ids = []
    
    df, steps = fetch_wandb_runs(
        filters={
            "$or": [
                {"name": {"$in": wandb_run_ids}},
            ]
        },
        return_steps=True,
    )
    return df

colors = ["#7CB9BC", "#8D69B8", "#DE836C", "#68AC5A", "#E59852"]

# Collect memorization scores for each NTP size
ntp_sizes = []
memorization_scores = []

for label, (wandb_path, ntp_size) in ntp_experiments.items():
    scores = fetch_runs(wandb_run_ids=[wandb_path])
    matching_cols = [col for col in scores.columns if 'perplexity' in col.lower() and "eval_finance-memorization" in col.lower()]
    
    if matching_cols:
        score = scores[matching_cols[0]].iloc[-1]
        ntp_sizes.append(ntp_size)
        memorization_scores.append(score)
    else:
        print(f"Warning: No memorization score found for {label} ({wandb_path})")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(ntp_sizes, memorization_scores, marker='o', linewidth=2, color=colors[0])
plt.xscale('log')  # optional: log scale if token sizes span orders of magnitude
plt.xlabel("Cache Size (Tokens)")
plt.ylabel("Perplexity (Memorization Slice)")
plt.title("Effect of Cache Size on Memorization")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("capsules/configs/simran/tasks/amd/analyze/memorization_vs_ntp_size.png")
plt.show()
