"""Download the selected VLM checkpoint only; do not load the model or run inference."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread
import time

import requests
from huggingface_hub import HfApi, hf_hub_url, snapshot_download
from describe_images import CACHE, MODEL, REVISION, progress


if __name__ == "__main__":
    print(f"Downloading {MODEL} into {CACHE}", flush=True)
    with progress("download-metadata", 0, 1):
        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
        snapshot = Path(snapshot_download(
            MODEL, revision=info.sha, cache_dir=CACHE, max_workers=4,
            allow_patterns=["*.json", "*.txt", "*.jinja"],
        ))
    weights = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
    if not weights or any(Path(f.rfilename).name != f.rfilename or not f.lfs for f in weights):
        raise RuntimeError("Expected root-level safetensors files with LFS size/SHA256 metadata.")
    total = sum(f.size for f in weights)
    received = {f.rfilename: 0 for f in weights}
    finished = set()
    lock, stopped = Lock(), Event()
    started = time.perf_counter()

    def report():
        while not stopped.wait(30):
            with lock:
                count, size = len(finished), sum(received.values())
            elapsed = time.perf_counter() - started
            eta = f"{(total - size) * elapsed / size:.0f}s" if size else "unknown"
            print(f"[download-weights] {count}/{len(weights)} files; {size}/{total} bytes "
                  f"({100 * size / total:.1f}%); elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def download(file):
        target = snapshot / file.rfilename
        partial = target.with_suffix(target.suffix + ".incomplete")
        digest = sha256()
        source = target if target.exists() else partial
        offset = source.stat().st_size if source.exists() else 0
        if offset > file.size:
            raise RuntimeError(f"Oversized file: {source}; inspect it before retrying.")
        if source.exists():
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        with lock:
            received[file.rfilename] = offset
        if not target.exists() and offset < file.size:
            # Direct resolve GET avoids the stalled HEAD redirect download path on this machine.
            with requests.get(hf_hub_url(MODEL, file.rfilename, revision=info.sha),
                              headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                if offset and response.status_code == 200:
                    offset, digest = 0, sha256()
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
                        offset += len(chunk)
                        with lock:
                            received[file.rfilename] = offset
            if offset != file.size:
                raise RuntimeError(f"Incomplete {file.rfilename}: {offset}/{file.size}; rerun downloader.")
        if digest.hexdigest() != file.lfs.sha256:
            raise RuntimeError(f"SHA256 mismatch: {source}; inspect/remove only this file before retrying.")
        if not target.exists():
            partial.replace(target)
        with lock:
            finished.add(file.rfilename)
        print(f"[download-weights] verified {file.rfilename}", flush=True)

    reporter = Thread(target=report, daemon=True)
    reporter.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(download, weights))
    finally:
        stopped.set()
        reporter.join()
    print(f"[model-download] 1/1 (100%) completed: {snapshot}", flush=True)
