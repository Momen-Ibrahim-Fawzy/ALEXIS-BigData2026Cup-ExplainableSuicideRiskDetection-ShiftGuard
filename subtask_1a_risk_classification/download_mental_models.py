"""
Download mental-health RoBERTa models from HuggingFace.

Run this on a server WITH internet access, then copy the output folder
to the offline server at:
  ~/.cache/huggingface/hub/

Usage:
    python download_mental_models.py
    python download_mental_models.py --check-only     # just test reachability
    python download_mental_models.py --output /path/to/custom/cache
"""

import os
import sys
import argparse
import requests

MODELS = [
    "mental/mental-roberta-base",
    "AIMH/mental-roberta-large",
    "dstefa/mental-roberta_stress_classification",
    "karangupta224/mental_roberta_stress",
    "welsachy/mental-roberta-base-finetuned-depression",
    "lishaangral/roberta-mental-health-v2",
]


def check_reachable(model_id: str) -> tuple[bool, str]:
    """Return (reachable, reason) by hitting the HuggingFace API."""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return True, "OK"
        elif r.status_code == 404:
            return False, "model not found (404)"
        elif r.status_code == 401:
            return False, "private repo — needs HF_TOKEN"
        else:
            return False, f"HTTP {r.status_code}"
    except requests.ConnectionError:
        return False, "no internet connection"
    except requests.Timeout:
        return False, "timeout"


def download_model(model_id: str, cache_dir: str | None) -> bool:
    from transformers import AutoTokenizer, AutoModel

    print(f"  Downloading tokenizer ...")
    AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    print(f"  Downloading model weights ...")
    AutoModel.from_pretrained(model_id, cache_dir=cache_dir)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Only check reachability, do not download")
    parser.add_argument("--output", default=None,
                        help="Custom cache directory (default: ~/.cache/huggingface/hub)")
    args = parser.parse_args()

    cache_dir = args.output  # None = HuggingFace default

    print("=" * 60)
    print("  Mental-health model downloader")
    print("=" * 60)
    if cache_dir:
        print(f"  Cache dir: {cache_dir}")
    else:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        print(f"  Cache dir: {hf_home}/hub  (default)")
    print()

    # Step 1: check all models first
    print("Checking reachability...")
    print(f"  {'Model':<52} Status")
    print(f"  {'-'*52} ------")
    reachable = {}
    for model_id in MODELS:
        ok, reason = check_reachable(model_id)
        reachable[model_id] = ok
        tag = "✓ reachable" if ok else f"✗ {reason}"
        print(f"  {model_id:<52} {tag}")

    available = [m for m, ok in reachable.items() if ok]
    blocked   = [m for m, ok in reachable.items() if not ok]

    print()
    print(f"  {len(available)} reachable, {len(blocked)} blocked")

    if blocked:
        print()
        print("BLOCKED (cannot download here):")
        for m in blocked:
            print(f"  {m}")

    if args.check_only or not available:
        return

    # Step 2: download reachable models
    print()
    print("=" * 60)
    print("Downloading reachable models...")
    print("=" * 60)

    failed = []
    for model_id in available:
        print(f"\n[{model_id}]")
        try:
            download_model(model_id, cache_dir)
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(model_id)

    print()
    print("=" * 60)
    succeeded = [m for m in available if m not in failed]
    print(f"  Downloaded {len(succeeded)}/{len(available)} models successfully.")
    if failed:
        print(f"  Failed: {failed}")
    print()
    print("Next step — copy the cache to the offline server:")
    hf_hub = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub"
    ) if not cache_dir else cache_dir
    for model_id in succeeded:
        folder = "models--" + model_id.replace("/", "--")
        print(f"  rsync -av {hf_hub}/{folder}/ offline-server:~/.cache/huggingface/hub/{folder}/")
    print("=" * 60)


if __name__ == "__main__":
    main()