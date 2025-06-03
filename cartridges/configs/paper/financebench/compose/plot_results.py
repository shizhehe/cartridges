import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

from capsules.analysis.utils import  HUE_TO_COLOR

# increase the font size
plt.rcParams.update({'font.size': 14})


def size_to_bytes(cache_size):
    num_layers = 28
    hidden_size = 3072
    bytes = cache_size*2*num_layers*hidden_size*2
    return bytes

import statistics

def add_composition_data(data, path = "cache_size_to_deps_to_scores.json"):
    with open(path, "r") as f:
        cache_size_to_deps_to_scores = json.load(f)
    for cache_size, deps_scores in cache_size_to_deps_to_scores.items():
        if int(cache_size) > 4096:
            continue
        for deps, score in deps_scores.items():
            if type(score) == list:
                if not score:
                    continue
                try:
                    score = statistics.median(score)
                except:
                    breakpoint()
            if plot_type == "compose_baseline": 
                tag = f"Cartridge Composition"
            else:
                tag = f"Cartridge Composition ({deps.capitalize()} Dep.)"
            data.append({
                "Cache Size": size_to_bytes(int(cache_size) * 2),
                "Dependencies": tag,
                "Score": score
            })
    return data


def add_baseline_cartridge(data):
    if plot_type == "compose_baseline":
        with open("/home/simarora/code/capsules/baseline_cache_size_to_deps_to_scores.json", "r") as f:
            baseline_cache_size_to_deps_to_scores = json.load(f)

        for cache_size, deps_scores in baseline_cache_size_to_deps_to_scores.items():
            if int(cache_size) > 4096:
                continue
            for deps, score in deps_scores.items():
                if type(score) == list:
                    score = sum(score) / len(score)
                if plot_type == "compose_baseline" and deps == "yes": 
                    continue

                data.append({
                    "Cache Size": size_to_bytes(int(cache_size)),
                    "Dependencies": f"Cartridge Baseline",
                    "Score": score
                })
    return data


def get_icl_baseline():

    with open("/home/simarora/code/capsules/icl_baseline_results.json", "r") as f:
        icl_baseline_results = json.load(f)

    cache_sizes = []
    scores = []
    for pair, results in icl_baseline_results.items():
        cache_size = results["cache_size"]
        score = results["score"]
        cache_sizes.append(size_to_bytes(cache_size))
        scores.append(score)

    avg_cache_size = sum(cache_sizes) / len(cache_sizes)
    avg_score = sum(scores) / len(scores)

    return avg_cache_size, avg_score


def plot(data, avg_cache_size, avg_score, idx = 0):

    df = pd.DataFrame(data)

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=df["Dependencies"].nunique())

    for i, (dep, group) in enumerate(df.groupby("Dependencies")):
        print(f"Plotting {dep}")

        group_sorted = group.sort_values("Cache Size")
        group_sorted = group_sorted.groupby("Cache Size").agg({"Score": "mean"}).reset_index()
        

        plt.plot(group_sorted["Cache Size"], group_sorted["Score"],
                label=dep, marker="o", markersize=6, color=palette[i], linewidth=2)
        
    # Add ICL baseline
    if plot_type == "compose_baseline":
        plt.axhline(y=avg_score, color='r', linestyle='--', label='ICL Baseline')
        plt.scatter(avg_cache_size, avg_score, color='r', s=50, label='')
        plt.xscale("log")

    plt.xlabel("Cache Size (Bytes)")
    plt.ylabel("log (Perplexity)")
    plt.grid(True)

    if plot_type == "compose_baseline":
        plt.title("Composing Cartridges")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')

    else:
        plt.title("Composition with Global Dependencies")
        plt.legend(title="Dependencies", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if plot_type == "compose_baseline":
        plt.savefig("capsules/configs/paper/financebench/compose/plots/composition_baseline.png")
    else:
        plt.savefig(f"capsules/configs/paper/financebench/compose/plots/{idx}_global_dependencies.png")

    # Also save a bar plot that shows averaged across all cache sizes
    if plot_type == "compose_baseline": 
        
        plt.figure(figsize=(5, 2.5))

        # add ICL to the df
        icl_df = pd.DataFrame({
            "Cache Size": [avg_cache_size],
            "Dependencies": ["ICL Baseline"],
            "Score": [avg_score]
        })
        df = pd.concat([df, icl_df], ignore_index=True)

        subdf = df.groupby("Dependencies").agg({"Score": "mean"}).reset_index()
        subdf = subdf.sort_values("Score")
        plt.ylabel("Method")
        plt.xlabel("⬅︎ log(perplexity)")  
        plt.title("Cartridge Composition")

        DEPENDENCIES_TO_HUE = {
            "Cartridge Composition": "cartridge_composition",
            "Cartridge Baseline": "cartridge",
            "ICL Baseline": "icl",
        }
        for row in subdf.to_dict(orient="records"):
            plt.barh(
                row["Dependencies"], 
                row["Score"], 
                color=HUE_TO_COLOR[DEPENDENCIES_TO_HUE[row["Dependencies"]]],
                edgecolor='black',
                linewidth=1.2,
            )

        # Add the average cache size text to the right of each bar
        avg_cache_sizes = []
        for dep in subdf["Dependencies"]:
            avg_cache_size = df[df["Dependencies"] == dep]["Cache Size"].mean()
            size = avg_cache_size / (1024**3)  # Convert to GB
            plt.text(subdf["Score"].max() + 0.1, dep, f"%.1f GB" % (size), va='center')
        plt.xlim(1.5, subdf["Score"].max() + 0.5)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)


        sns.despine()
        plt.tight_layout()
        path = f"capsules/configs/paper/financebench/compose/plots/{idx}_bar_plot.png"
        plt.savefig(path)
        plt.savefig(path.replace(".png", ".pdf"))
        print(f"Saved bar plot to {path}")
        print(f"Saved bar plot to {path.replace('.png', '.pdf')}")



for plot_type in ["compose_baseline", "global_dependencies"]:

    import os
    paths = os.listdir("/home/simarora/code/capsules")
    paths = [p for p in paths if "_cache_size_to_deps_to_scores_11" in p and 'baseline11']
    avg_cache_size, avg_score = get_icl_baseline()

    # clear all files in plot_path
    plot_path = "/home/simarora/code/capsules/capsules/configs/paper/financebench/compose/"
    for f in os.listdir(plot_path):
        if f.endswith(".png"):
            os.remove(os.path.join(plot_path, f))

    all_data = []

    for i, p in enumerate(paths):
        data = []

        company1, company2 = p.split("_cache_size_to_deps_to_scores")[0].split("_")
        idx = f"{company1}_{company2}"

        p = "/home/simarora/code/capsules/" + p
        
        all_data = add_composition_data(all_data, path = p)
        if plot_type == "compose_baseline":
            all_data = add_baseline_cartridge(all_data)
        else:
            data = add_composition_data(data, path = p)
            # plot(data, avg_cache_size, avg_score, idx = f"{company1}_{company2}")

    plot(all_data, avg_cache_size, avg_score, idx = "all")


