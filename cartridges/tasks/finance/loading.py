from functools import lru_cache
import os
import random
from pathlib import Path
import requests

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from pydrantic import ObjectConfig


def extract_text_from_pdf_url(pdf_url):
    import pymupdf
    # Fetch the PDF content from the URL
    response = requests.get(pdf_url)
    response.raise_for_status()  # Ensure the request was successful

    # Open the PDF from the fetched content
    pdf_data = response.content
    doc = pymupdf.open(stream=pdf_data, filetype="pdf")

    all_text = ""
    # Iterate over each page
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load page by index
        text = page.get_text()  # Extract text from page
        all_text += text
        #print(f"Page {page_num + 1}:\n{text}\n")

    # Close the document
    doc.close()
    return all_text

def process_dataset_pdfs(df: pd.DataFrame):
    """
    Process PDFs for each row in the dataset
    Adds a 'pdf_text' column with the extracted text
    """
    
    # Process each URL in the dataset
    results = []
    for _, row in tqdm(df.iterrows(), desc="Processing PDFs"):  # Replace 'pdf_url' with your column name
        url = row['doc_link']
        try:
            text = extract_text_from_pdf_url(url)
            results.append({
                **row,
                "pdf_text": text,
            })
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    return pd.DataFrame(results)

@lru_cache(maxsize=1)
def load_finance():
    from datasets import load_dataset

    dataset = load_dataset(
        "PatronusAI/financebench", split="train", trust_remote_code=True
    )

    dataset_dir = Path(dataset.cache_files[0]['filename']).parent

    # download the pdfs and process them
    # since this takes a while, we cache the result to the huggingface directory
    path = os.path.join(dataset_dir, "bench_with_pdfs.feather")
    if os.path.exists(path):
        df = pd.read_feather(path)
    else:
        df = dataset.to_pandas()
        df = process_dataset_pdfs(df)
        df.to_feather(path)
    return df
