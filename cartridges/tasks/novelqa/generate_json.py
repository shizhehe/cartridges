from datasets import load_dataset
import json
import os 
import wandb
import pandas as pd
from collections import defaultdict


# load from wandb

def load_wandb_table(artifact_path: str, artifact_name: str) -> pd.DataFrame:
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")
    print(f"Artifact name: {artifact_name}")

    table_file = os.path.join(artifact_dir, "generate_novelqa_generate_B01", "table.table.json")
    print(f"Loading table from: {table_file}")
    if not os.path.exists(table_file):
        raise FileNotFoundError(f"Expected table file not found at: {table_file}")

    with open(table_file, "r") as f:
        table_json = json.load(f)
        print(table_json.keys())

    # Extract columns and data
    columns = table_json["columns"]
    data = table_json["data"]
    df = pd.DataFrame(data, columns=columns)
    return df


# save for leaderboard

pathname = "hazy-research/capsules/run-rfmcs03s-generate_novelqa_generate_B01table:v0" # icl
pathname = "hazy-research/capsules/run-68u4k1a9-generate_novelqa_generate_B01table:v0" # rag topk=1
pathname = 'hazy-research/capsules/run-ve6sx0uq-generate_novelqa_generate_B01table:v0' # rag topk=2
pathname = "hazy-research/capsules/run-zwb38eyx-generate_novelqa_generate_B01table:v0" # rag topk=4
df = load_wandb_table(pathname, "table")

if not os.path.exists("outputs"):
    os.makedirs("outputs")

if not os.path.exists("outputs/"):
    os.makedirs("outputs/res_mc")

out_path = "outputs/res_mc/res_mc.json"
output = defaultdict(list)
for idx, row in df.iterrows():
    closest_letter = row['closest_letter']
    if closest_letter == "null" or not closest_letter:
        closest_letter = "A"
    title = row['book_title'].lower().replace(".", "").replace(" ", "").replace(",", "").replace("-", "").replace("_", "")
    output[title].append(closest_letter)

# for the remaining books just fill in "A"
fmeta = "/home/simarora/code/capsules/scratch/simran/NovelQA/bookmeta.json"
fquestions = "/home/simarora/code/capsules/scratch/simran/NovelQA/Data/PublicDomain/"
fquestions2 = "/home/simarora/code/capsules/scratch/simran/NovelQA/Data/CopyrightProtected/"
metadata = json.load(open(fmeta))
for book in metadata:
    title = metadata[book]["title"].lower().replace(".", "").replace(" ", "").replace(",", "").replace("-", "").replace("_", "")
    if title not in output:
        try:
            qpath = f"{fquestions}{book}.json"
            qs = json.load(open(qpath))
        except:
            qpath = f"{fquestions2}{book}.json"
            qs = json.load(open(qpath))
        output[title] = ["A"] * len(qs)

with open(out_path, "w") as f:
    json.dump(output, f, indent=4)



