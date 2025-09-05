DATA_SOURCES = [
    # "/data/sabri/cartridges/2025-07-26-12-21-32-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0/artifact/dataset.pkl",
    # "/data/sabri/cartridges/2025-07-26-13-40-44-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0/artifact/dataset.pkl",
    # "/data/sabri/cartridges/2025-08-05-09-43-20-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0/artifact/dataset.pkl",

    # # "/data/sabri/cartridges/2025-07-26-12-02-19-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl"

    # "/data/sabri/cartridges/2025-07-27-14-11-52-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl",
    # "/data/sabri/cartridges/2025-07-27-15-00-07-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl",

    "/data/sabri/cartridges/2025-07-30-19-03-42-m07d28_mtob_synthesize/m07d28_mtob_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",
    "/data/sabri/cartridges/2025-07-30-19-18-45-m07d28_mtob_synthesize/m07d28_mtob_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",

    "/data/sabri/cartridges/2025-07-28-18-33-28-m07d28_mtob_synthesize/m07d28_mtob_synthesize_qwen3-4b_n65536-0/artifact/dataset.pkl",
    "/data/sabri/cartridges/2025-07-28-20-04-14-m07d28_mtob_synthesize/m07d28_mtob_synthesize_qwen3-4b_n65536-0/artifact/dataset.pkl",

    "/data/sabri/cartridges/2025-07-28-12-18-54-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k1-0/artifact/dataset.pkl",
    "/data/sabri/cartridges/2025-07-28-12-28-46-m07d28_niah_synthesize/m07d28_niah_synthesize_llama-3.2-3b_n65536_k1-0/artifact/dataset.pkl",
]

from cartridges.utils.hf import upload_run_dir_to_hf

seen = set()

for idx, data_source in enumerate(DATA_SOURCES):

    run_dir = data_source.replace("/artifact/dataset.pkl", "")
    repo_id = run_dir.split("/")[-1]
    repo_id = f"hazyresearch/{repo_id}"

    for idx in range(10):
        if repo_id in seen:
            repo_id = repo_id[:-2] + f"-{idx}"
        else:
            break
    else:
        raise ValueError(f"Failed to find a unique repo_id for {repo_id}")

    seen.add(repo_id)

    dataset_url = upload_run_dir_to_hf(
        run_dir=run_dir,
        repo_id=repo_id,
        private=False,
        commit_message=f"Upload {data_source}",
        collection_slug="hazyresearch/cartridges-689f93fa4fecdee6cf77c11e"
    )
    print(f"({idx+1}/{len(DATA_SOURCES)}) ✅ Dataset uploaded successfully: {dataset_url}")