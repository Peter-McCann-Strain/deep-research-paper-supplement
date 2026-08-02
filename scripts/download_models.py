#!/usr/bin/env python3
"""Download local models for P9/P10 patterns.

Downloads:
  - Qwen/Qwen2.5-7B-Instruct (P9 baseline)
  - GAIR/DeepResearcher-7b (P10 RL-trained)

Models are cached in HuggingFace's default cache (~/.cache/huggingface/).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_model(model_id: str):
    """Download model weights and tokenizer."""
    from huggingface_hub import snapshot_download
    print(f"\nDownloading {model_id}...")
    path = snapshot_download(model_id)
    print(f"  Saved to: {path}")
    return path


def verify_model(model_id: str):
    """Quick verification that model loads."""
    from transformers import AutoTokenizer
    print(f"Verifying tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  OK")


if __name__ == "__main__":
    models = [
        "Qwen/Qwen2.5-7B-Instruct",    # P9 baseline
        "GAIR/DeepResearcher-7b",         # P10 RL-trained
    ]

    for model_id in models:
        try:
            download_model(model_id)
            verify_model(model_id)
        except Exception as e:
            print(f"  ERROR downloading {model_id}: {e}")
            print("  You can retry later or download manually.")

    print("\nDone! Models cached in ~/.cache/huggingface/")
