"""Download and verify a pinned local DINOv2 ViT-B/14 checkpoint."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import re

from hybrid_rag.full_image_method.corpus_image_embedding import (
    MODEL, MODEL_DIR, MODEL_FILES, check_hashes, file_hash, run_step, save,
)


def download(args, folder, meta, progress):
    from huggingface_hub import HfApi, hf_hub_download

    if meta.get("output_sha256"):
        check_hashes(meta["output_sha256"])
    progress.set("resolve model revision")
    info = HfApi().model_info(MODEL, revision=meta.get("revision") or "main", files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Không resolve được commit SHA của model.")
    meta.update(status="RUNNING", model=MODEL, revision=info.sha)
    save(MODEL_DIR / "manifest.json", meta)
    entries = {item.rfilename: item for item in info.siblings}
    progress.set("download model files", len(MODEL_FILES))
    for name in MODEL_FILES:
        entry = entries[name]
        path = Path(hf_hub_download(MODEL, name, revision=info.sha, local_dir=MODEL_DIR))
        digest = file_hash(path)
        if entry.size is not None and path.stat().st_size != entry.size:
            raise ValueError(f"Model tải thiếu byte: {name}")
        if entry.lfs and digest != entry.lfs.sha256:
            raise ValueError(f"SHA256 upstream không khớp: {name}")
        meta.setdefault("output_sha256", {})[str(path.relative_to(ROOT))] = digest
        save(MODEL_DIR / "manifest.json", meta)
        progress.done += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    args.command = "download"
    run_step(args, MODEL_DIR, download, Path(__file__))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
