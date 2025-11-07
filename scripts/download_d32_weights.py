#!/usr/bin/env python3
"""
download_d32_weights.py - Download d32 base checkpoint from Hugging Face

This script downloads the nanochat d32 model weights from:
https://huggingface.co/karpathy/nanochat-d32

The checkpoint will be organized into:
  base_checkpoints/d32/
    ├── model.pt          (model weights)
    ├── config.json       (model configuration)
    └── tokenizer.pkl     (tokenizer)

Usage:
  python scripts/download_d32_weights.py
  
Requirements:
  pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

def download_d32_weights():
    """Download d32 checkpoint from Hugging Face."""
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    # Configuration
    repo_id = "karpathy/nanochat-d32"
    output_dir = Path("base_checkpoints/d32")
    
    print("="*70)
    print("Downloading nanochat d32 base checkpoint")
    print("="*70)
    print(f"Repository: {repo_id}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {output_dir}")
    
    # List all files in the repo
    print(f"\nListing files in {repo_id}...")
    try:
        all_files = list_repo_files(repo_id)
        print(f"Found {len(all_files)} files in repository")
    except Exception as e:
        print(f"ERROR: Could not list repository files: {e}")
        sys.exit(1)
    
    # Filter for checkpoint files
    checkpoint_files = [
        f for f in all_files 
        if f.endswith(('.pt', '.json', '.pkl', '.bin', '.safetensors'))
    ]
    
    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in {repo_id}")
        print(f"Available files: {all_files}")
        sys.exit(1)
    
    print(f"\nCheckpoint files to download:")
    for f in checkpoint_files:
        print(f"  - {f}")
    print()
    
    # Download each file
    downloaded_files = []
    total_size = 0
    
    for filename in checkpoint_files:
        print(f"Downloading: {filename}")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=output_dir,
                local_dir_use_symlinks=False  # Actually copy files, don't symlink
            )
            
            # Get file size
            file_size = os.path.getsize(downloaded_path)
            total_size += file_size
            size_mb = file_size / (1024 * 1024)
            
            print(f"  ✓ Downloaded: {downloaded_path} ({size_mb:.2f} MB)")
            downloaded_files.append(downloaded_path)
            
        except Exception as e:
            print(f"  ✗ ERROR downloading {filename}: {e}")
            continue
    
    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Total files downloaded: {len(downloaded_files)}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Location: {output_dir.absolute()}")
    print()
    
    # List final directory structure
    print("Final directory structure:")
    for file in sorted(output_dir.rglob("*")):
        if file.is_file():
            rel_path = file.relative_to(output_dir)
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.2f} MB)")
    
    print()
    print("Next steps:")
    print("  1. Verify files: ls -lh base_checkpoints/d32/")
    print("  2. Check total size is ~4-5 GB")
    print("  3. Proceed with training validation")
    print()

if __name__ == "__main__":
    download_d32_weights()