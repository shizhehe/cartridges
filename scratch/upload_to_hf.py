#!/usr/bin/env python3
"""
Script to upload a synthesized dataset to HuggingFace post-hoc.

Usage:
    python upload_to_hf.py <run_directory> <hf_repo_id> [--private] [--token <token>]

Examples:
    python upload_to_hf.py ./runs/my_dataset username/my-dataset
    python upload_to_hf.py ./runs/my_dataset username/my-dataset --private
    python upload_to_hf.py ./runs/my_dataset username/my-dataset --token hf_abc123
"""

import argparse
import sys
from pathlib import Path

from cartridges.utils.hf import upload_run_dir_to_hf
from cartridges.utils import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Upload a synthesized dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("run_dir", help="Path to the run directory containing the synthesized dataset")
    parser.add_argument("repo_id", help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", help="HuggingFace token (defaults to HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--commit-message", help="Custom commit message")
    
    args = parser.parse_args()
    
    # Validate run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    if not run_dir.is_dir():
        logger.error(f"Path is not a directory: {args.run_dir}")
        sys.exit(1)
    
    # Check for required files
    artifact_dir = run_dir / "artifact"
    config_file = run_dir / "config.yaml"
    
    if not artifact_dir.exists():
        logger.error(f"Artifact directory not found: {artifact_dir}")
        sys.exit(1)
    
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
    
    # Check for dataset file
    dataset_files = list(artifact_dir.glob("*.parquet")) + list(artifact_dir.glob("*.pkl"))
    if not dataset_files:
        logger.error(f"No dataset file (.parquet or .pkl) found in {artifact_dir}")
        sys.exit(1)
    
    try:
        logger.info(f"Uploading {args.run_dir} to {args.repo_id}")
        dataset_url = upload_run_dir_to_hf(
            run_dir=args.run_dir,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
        print(f"✅ Dataset uploaded successfully: {dataset_url}")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()