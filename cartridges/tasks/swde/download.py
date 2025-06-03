import os

from tqdm import tqdm


DIR_PATHS = [
    "data/swde_movie_allmovie",
    "data/swde_movie_amctv",
    "data/swde_movie_hollywood",
    "data/swde_movie_iheartmovies",
    "data/swde_movie_imdb",
    "data/swde_movie_metacritic",
    "data/swde_movie_rottentomatoes",
    "data/swde_movie_yahoo",
    "data/swde_university_collegeprowler",
]

docs_template ="https://huggingface.co/datasets/hazyresearch/evaporate/resolve/main/{path}/docs.tar.gz"
table_template = "https://huggingface.co/datasets/hazyresearch/evaporate/resolve/main/{path}/table.json"



if __name__ == "__main__":
    destination = "/data/sabri/data/evaporate/"

    os.makedirs(destination, exist_ok=True)

    for path in tqdm(DIR_PATHS):
        docs_url = docs_template.format(path=path)
        table_url = table_template.format(path=path)

        docs_path = f"{destination}/{path}/docs.tar.gz"
        table_path = f"{destination}/{path}/table.json"

        os.makedirs(f"{destination}/{path}", exist_ok=True)
        if os.path.exists(docs_path) and os.path.exists(table_path):
            print(f"Skipping {path} because it already exists")
            continue

        os.system(f"wget {docs_url} -O {docs_path}")
        os.system(f"wget {table_url} -O {table_path}")


    # list paths in data_path
    data_path = os.path.join(destination, "data")
    data_path_files = os.listdir(data_path)
    data_path_files = [f for f in data_path_files if "swde" in f]
    for path in data_path_files:
        sub_path = os.path.join(data_path, path)
        # tar unzip 'docs.tar.gz' in the sub_paths
        if os.path.exists(f"{sub_path}/docs.tar.gz"):
            print(f"unzipping {sub_path}/docs.tar.gz")
            os.system(f"tar -xvf {sub_path}/docs.tar.gz -C {sub_path}")