import argparse
import threading
import time
import random
import concurrent.futures
import pandas as pd
from openai import OpenAI, APIError
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os
from typing import List, Dict, Any, Optional, Set
import tqdm
import random

BUFFER_TOKENS = 20_000

REPEAT_SIZE = 50_000

HARDCODED_ENGLISH_WORDS: List[str] = [
    "apple",
    "banana",
    "carrot",
    "dog",
    "elephant",
    "flower",
    "guitar",
    "house",
    "igloo",
    "jungle",
    "kangaroo",
    "lemon",
    "mountain",
    "notebook",
    "ocean",
    "penguin",
    "quilt",
    "rainbow",
    "strawberry",
    "tiger",
    "umbrella",
    "violin",
    "waterfall",
    "xylophone",
    "yacht",
    "zebra",
    "acorn",
    "butterfly",
    "cactus",
    "dolphin",
    "eagle",
    "forest",
    "giraffe",
    "hamburger",
    "island",
    "jacket",
    "koala",
    "lighthouse",
    "mushroom",
    "noodle",
    "octopus",
    "panda",
    "queen",
    "rocket",
    "sunflower",
    "telescope",
    "unicorn",
    "volcano",
    "watermelon",
    "yarn",
    "zipper",
    "alligator",
    "balloon",
    "candle",
    "diamond",
    "engine",
    "feather",
    "galaxy",
    "harmonica",
    "iceberg",
    "jellyfish",
    "key",
    "lantern",
    "magnet",
    "nest",
    "orange",
    "parrot",
    "puzzle",
    "robot",
    "saxophone",
    "train",
    "urchin",
    "vase",
    "whistle",
    "zeppelin",
    "anchor",
    "biscuit",
    "castle",
    "dragon",
    "envelope",
    "fountain",
    "garden",
    "helmet",
    "iguana",
    "jewel",
    "kite",
    "ladder",
    "mirror",
    "necklace",
    "owl",
    "piano",
    "quokka",
    "river",
    "sandwich",
    "toaster",
    "ukulele",
    "vacuum",
    "windmill",
    "yo-yo",
    "zucchini",
    "ant",
    "bear",
    "cat",
    "deer",
    "eel",
    "fox",
    "goat",
    "hare",
    "ibex",
    "jaguar",
    "kiwi",
    "lion",
    "mole",
    "newt",
    "otter",
    "pig",
    "quail",
    "rat",
    "seal",
    "toad",
    "urchin",
    "vole",
    "wolf",
    "yak",
    "zebu",
    "book",
    "chair",
    "desk",
    "lamp",
    "pen",
    "ruler",
    "table",
    "watch",
    "door",
    "window",
    "floor",
    "roof",
    "wall",
    "brick",
    "cloud",
    "grass",
    "leaf",
    "moon",
    "rain",
    "sand",
    "snow",
    "star",
    "stone",
    "sun",
    "tree",
]


def prepare_candidate_tokens(
    tokenizer: PreTrainedTokenizerFast, words: List[str]
) -> List[int]:
    """
    Tokenizes a list of words and returns a flat list of their token IDs.
    """
    candidate_ids: List[int] = []
    if not words:
        print(
            "Warning: HARDCODED_ENGLISH_WORDS list is empty. No candidate tokens for prefix generation."
        )
        return candidate_ids

    for word in words:
        # We use add_special_tokens=False to get only the tokens for the word itself.
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        candidate_ids.extend(token_ids)

    if not candidate_ids:
        print(
            "Warning: Tokenization of HARDCODED_ENGLISH_WORDS resulted in an empty candidate token list."
        )
    else:
        print(
            f"Generated {len(candidate_ids)} candidate tokens from {len(words)} hardcoded words."
        )
    return candidate_ids


def generate_prefix(
    tokenizer: PreTrainedTokenizerFast, num_tokens: int, candidate_token_ids: List[int]
) -> str:
    """
    Generates a prefix string by sampling from candidate_token_ids.
    """
    assert num_tokens > 0

    if not candidate_token_ids:
        raise ValueError("Cannot generate prefix: candidate_token_ids list is empty.")

    generate_suffix = ". The following are all the countries in the world, in Alphabetical Order: Afghanistan, Albania, Algeria,"
    generate_suffix_tokens = tokenizer.encode(generate_suffix)
    num_prefix_tokens = num_tokens - 1 - (len(generate_suffix_tokens))  # -1 for BOS

    selected_token_ids = random.choices(candidate_token_ids, k=num_prefix_tokens)
    prefix_string = tokenizer.decode(selected_token_ids + generate_suffix_tokens)

    return prefix_string


def api_request_worker(
    thread_id: int,
    num_iters: int,
    num_decode_tokens: int,
    model_name: str,
    base_url: str,
    prefix_text: str,
):
    # print(f"Thread-{thread_id}: Starting with {num_iters} iterations.")
    client = OpenAI(base_url=base_url)
    worker_results = []
    for i in range(num_iters):
        request_data: Dict[str, Any] = {
            "thread_id": thread_id,
            "iteration": i,
            "decode_tokens_target": num_decode_tokens,
        }

        start_time = time.time()

        response = client.completions.create(
            model=model_name,
            prompt=prefix_text,  # Pass the generated prefix text as the prompt
            max_tokens=num_decode_tokens,
            # Other parameters like temperature could be set if needed.
            temperature=0.0,
        )
        if response.usage.completion_tokens != num_decode_tokens:
            print(response.choices[0])
        #     breakpoint()

        # try:
        assert response.usage.completion_tokens == num_decode_tokens
        assert response.choices[0].finish_reason == "length"
        # except:
        #     breakpoint()

        end_time = time.time()
        latency = end_time - start_time

        # The response structure for usage and finish_reason is generally similar
        request_data.update(
            {
                "request_start_time": start_time,
                "request_end_time": end_time,
                "latency": latency,
            }
        )

        worker_results.append(request_data)

    return worker_results


def main(prefix_size: int):
    parser = argparse.ArgumentParser(
        description="LLM Inference Server Throughput Tester (using completions.create)"
    )
    parser.add_argument(
        "--kv_cache_size",
        type=int,
        required=True,
        help="Total token capacity of the KV cache.",
    )
    # parser.add_argument(
    #     "--prefix_size",
    #     type=int,
    #     required=True,
    #     help="Number of tokens for the initial prefix.",
    # )
    parser.add_argument(
        "--num_decode_tokens",
        type=int,
        default=128,
        help="Number of tokens to generate after the prefix.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=10,
        help="Number of requests each thread makes.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Name of the model to use for API calls.",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        help="Base URL for the OpenAI-compatible server.",
        default="http://localhost:30000/v1",
    )

    args = parser.parse_args()

    # Validate inputs
    if prefix_size < 0:
        print("Error: prefix_size must be non-negative.")
        return
    if args.num_decode_tokens <= 0:
        print("Error: num_decode_tokens must be positive.")
        return
    if args.kv_cache_size <= 0:
        print("Error: kv_cache_size must be positive.")
        return

    denominator = prefix_size + args.num_decode_tokens
    assert denominator > 0
    assert args.kv_cache_size > 0

    num_concurrent_requests = (args.kv_cache_size - BUFFER_TOKENS) // denominator
    # num_concurrent_requests = 1

    assert num_concurrent_requests > 0

    print(f"--- Configuration ---")
    print(f"KV Cache Size: {args.kv_cache_size}")
    print(f"Prefix Size: {prefix_size}")
    print(f"Num Decode Tokens: {args.num_decode_tokens}")
    print(f"Num Iterations per Thread: {args.num_iters}")
    print(f"Model Name: {args.model_name}")
    print(f"API Base URL: {args.api_base_url}")
    print(f"Calculated Num Concurrent Requests: {num_concurrent_requests}")
    print(f"--- Starting Test ---")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Prepare candidate tokens from hardcoded words
    candidate_token_ids = prepare_candidate_tokens(tokenizer, HARDCODED_ENGLISH_WORDS)
    assert (
        len(candidate_token_ids) > 0
    ), "Candidate token IDs list is empty. Cannot proceed with prefix generation."

    prefixes = set()
    for i in range(num_concurrent_requests):
        prefix_text = generate_prefix(tokenizer, prefix_size, candidate_token_ids)
        assert prefix_text not in prefixes
        prefixes.add(prefix_text)
    prefixes = list(prefixes)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_concurrent_requests
    ) as executor:
        futures = []

        for i, prefix_text in zip(range(num_concurrent_requests), prefixes):
            future = executor.submit(
                api_request_worker,
                i,
                args.num_iters,
                args.num_decode_tokens,
                args.model_name,
                args.api_base_url,
                prefix_text,
            )
            futures.append(future)

        results = []
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            "waiting for threads to finish",
            total=len(futures),
        ):
            results += future.result()

    print("--- Test Finished ---")

    results_df = pd.DataFrame(results)
    desired_columns = [
        "thread_id",
        "iteration",
        "latency",
        "request_start_time",
        "request_end_time",
    ]
    # Ensure all desired columns are present, add missing ones with None if they weren't added during request_data population
    for col in desired_columns:
        if col not in results_df.columns:
            results_df[col] = None

    results_df = results_df[desired_columns]

    print("\n--- Results (CSV format) ---")
    # print(results_df.to_csv(index=False))

    # Ensure completion tokens are numeric before summing, handle potential None values

    middle_results = results_df[
        (results_df["iteration"] > 3) & (results_df["iteration"] < args.num_iters - 2)
    ]
    print(f"\n--- Summary ---")

    assert len(set(middle_results["thread_id"])) == num_concurrent_requests
    tps = (
        args.num_decode_tokens / middle_results["latency"].mean()
    ) * num_concurrent_requests
    print("TPS:", f"{tps:.2f}")
    return tps


if __name__ == "__main__":
    random.seed(50)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prefix_size_to_tps = {}

    for prefix_size in [
        # 128,
        # 256,
        # 512,
        # 2**10,
        # 2**11,
        # 2**12,
        # 2**13,
        2**14,
        2**15,
        2**16,
        120_000,
    ]:
        print(f"--- Running test with prefix size: {prefix_size} ---")
        prefix_size_to_tps[prefix_size] = float(main(prefix_size))
    print("Prefix size to TPS mapping")
    print(prefix_size_to_tps)
