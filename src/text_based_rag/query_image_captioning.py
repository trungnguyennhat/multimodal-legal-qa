"""Caption only images referenced by the train split; no query embedding or retrieval."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from threading import Event, Thread
import time

from multimodal_legal_rag.data import read_json_list
from .image_captioning import ROOT, MODEL, REVISION, CACHE, MIN_PIXELS, MAX_PIXELS, save

INPUT = ROOT / "data/processed/splits/train.json"
IMAGES = ROOT / "data/processed/images/train"
OUTPUT = ROOT / "artifacts/results/text_based_rag/query_image_captioning"
MAX_NEW_TOKENS = 384
TRAIN_PROMPT = """Mô tả biển báo trong ảnh. Không cần mô tả quá chi tiết. Chỉ mô tả hình dạng của các biển, và chỉ mô tả hình có trong biển. Không cần giải thích tên gọi hay ý nghĩa của biển đó là gì."""


def main():
    rows = read_json_list(INPUT)
    query_ids, unique, mapping = set(), {}, []
    by_filename = {}
    for row in rows:
        query_id, source_image_id = row.get("id"), row.get("image_id")
        if not isinstance(query_id, str) or query_id in query_ids:
            raise ValueError("Query thiếu ID hoặc trùng ID trong train split.")
        if (not isinstance(source_image_id, str) or not source_image_id
                or Path(source_image_id).name != source_image_id
                or any(char in source_image_id for char in ("/", "\\", ":"))):
            raise ValueError(f"image_id không hợp lệ: {query_id}")
        query_ids.add(query_id)
        filename = source_image_id + ".jpg"
        path = (IMAGES / filename).resolve()
        if path.parent != IMAGES.resolve() or not path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh train: {filename}")
        if filename not in by_filename:
            image_id = "sha256:" + sha256(path.read_bytes()).hexdigest()
            by_filename[filename] = image_id
            unique.setdefault(image_id, {"image_id": image_id, "filename": filename,
                                         "path": str(path), "query_ids": []})
        image_id = by_filename[filename]
        unique[image_id]["query_ids"].append(query_id)
        caption_file = Path(unique[image_id]["filename"]).stem + "_description.json"
        mapping.append({"query_id": query_id, "source_image_id": source_image_id,
                        "filename": filename, "image_id": image_id,
                        "caption_file": "captions/" + caption_file})
    if not unique:
        raise ValueError("Train split không có ảnh để caption.")
    signature = sha256(json.dumps(mapping, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    previous_path = OUTPUT / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("train_images_signature") != signature:
            raise ValueError("Danh sách query hoặc ảnh train đã thay đổi. Tự lưu output cũ sang nơi khác trước khi chạy lại.")
    (OUTPUT / "captions").mkdir(parents=True, exist_ok=True)
    settings = {"model": MODEL, "revision": REVISION, "prompt": TRAIN_PROMPT,
                "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
                "max_new_tokens": MAX_NEW_TOKENS, "crop": False, "do_sample": False,
                "dtype": "bfloat16", "attention": "sdpa",
                "libraries": {name: version(name) for name in ("torch", "transformers", "bitsandbytes", "Pillow")}}
    paths = {image_id: OUTPUT / "captions" / (Path(item["filename"]).stem + "_description.json")
             for image_id, item in unique.items()}
    assert len({str(path).casefold() for path in paths.values()}) == len(unique), "Trùng tên checkpoint"
    results, incomplete = {}, []
    for image_id, path in paths.items():
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("image_id") != image_id:
                raise ValueError(f"Ảnh đã thay đổi: {path.name}")
            if saved.get("status") == "COMPLETED" and saved.get("description", "").strip():
                results[image_id] = saved
    resumed = len(results)
    manifest = {"status": "IN_PROGRESS", "split": "train", "input": str(INPUT),
                "input_sha256": sha256(INPUT.read_bytes()).hexdigest(),
                "train_images_signature": signature, "settings": settings,
                "queries": len(rows), "image_files": len(by_filename), "unique_image_hashes": len(unique),
                "started_at": datetime.now(timezone.utc).isoformat(), "resumed_images": resumed}
    started = time.perf_counter()
    stopped = Event()
    phase = "load-qwen"

    def report():
        elapsed = time.perf_counter() - started
        done = len(results) + len(incomplete)
        processed = done - resumed
        eta = f"{elapsed * (len(unique) - done) / processed:.0f}s" if processed else "unknown"
        print(f"[{phase}] {done}/{len(unique)} ({100 * done / len(unique):.1f}%) "
              f"chưa_xong={len(incomplete)} elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        save(OUTPUT / "manifest.json", manifest)
        temporary = OUTPUT / "query_image_mapping.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for item in mapping:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary.replace(OUTPUT / "query_image_mapping.jsonl")
        print(f"[resume] Giữ {resumed} ảnh đã xong; chỉ caption ảnh train còn thiếu.", flush=True)
        report()
        if len(results) < len(unique):
            import torch
            from PIL import Image, ImageOps
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Cần GPU NVIDIA CUDA hỗ trợ BF16 trong .venv.")
            free, total = torch.cuda.mem_get_info()
            budget = min(12 * 1024**3, free - 2 * 1024**3)
            if budget < 9 * 1024**3:
                raise RuntimeError("Cần ít nhất 11 GiB VRAM trống. Đóng process BGE/Qwen khác rồi chạy lại.")
            # shortcut: allocator cap excludes other CUDA allocations; keep competing GPU applications closed.
            torch.cuda.set_per_process_memory_fraction(budget / total)
            manifest["resources"] = {"gpu": torch.cuda.get_device_name(0), "total_vram_bytes": total,
                                     "free_vram_before_load": free, "pytorch_budget_bytes": budget}
            save(OUTPUT / "manifest.json", manifest)
            processor = AutoProcessor.from_pretrained(
                MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True, use_fast=False,
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True,
                dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
            if not model.is_loaded_in_4bit:
                raise RuntimeError("Dừng: Qwen không được nạp ở 4-bit.")
            phase = "train-image-caption-inference"
            for image_id, item in unique.items():
                if image_id in results:
                    continue
                image_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                with Image.open(item["path"]) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                try:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": item["path"]}, {"type": "text", "text": TRAIN_PROMPT}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                    if inputs.input_ids.shape[1] > 2048:
                        raise ValueError(f'Input vượt 2048 token: {item["filename"]}')
                    with torch.inference_mode():
                        output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
                    generated = output[0, inputs.input_ids.shape[1]:]
                    description = processor.decode(generated, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).strip()
                    hit_limit = generated.numel() >= MAX_NEW_TOKENS
                    result = {"image_id": image_id, "filename": item["filename"], "description": description,
                              "settings": settings, "status": "COMPLETED" if description and not hit_limit else "INCOMPLETE",
                              "token_limit_reached": hit_limit, "generated_tokens": generated.numel(),
                              "size": list(image.size), "elapsed_seconds": time.perf_counter() - image_started,
                              "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                              "gpu_free_bytes": torch.cuda.mem_get_info()[0]}
                    save(paths[image_id], result)
                    if result["status"] == "INCOMPLETE":
                        incomplete.append(item["filename"])
                        print(f'[warning] {item["filename"]}: caption rỗng/chạm token; tiếp tục ảnh sau.', flush=True)
                    else:
                        results[image_id] = result
                    print(f'[image] {item["filename"]}\n{description}', flush=True)
                    del inputs, output, generated
                    torch.cuda.empty_cache()
                    report()
                finally:
                    image.close()
        phase = "export-captions"
        temporary = OUTPUT / "image_captions.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for image_id in unique:
                value = json.loads(paths[image_id].read_text(encoding="utf-8"))
                assert value["image_id"] == image_id and value["status"] in {"COMPLETED", "INCOMPLETE"}
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        temporary.replace(OUTPUT / "image_captions.jsonl")
        manifest.update(status="INCOMPLETE" if incomplete else "COMPLETED", captioned_images=len(results),
                        incomplete_images=incomplete, elapsed_seconds=time.perf_counter() - started)
        save(OUTPUT / "manifest.json", manifest)
        report()
        print(f"[output] {OUTPUT}; status={manifest['status']}", flush=True)
    except BaseException as exc:
        manifest.update(status="FAILED", error=str(exc), captioned_images=len(results), incomplete_images=incomplete)
        save(OUTPUT / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
