"""Caption corpus images locally, checkpoint each image, and enrich figure blocks."""

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
import json
import time

from .corpus_normalization import OUTPUT as INPUT, ROOT, preview

MODEL = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
REVISION = "10d842936325905934ea24e64325d774693f166e"
CACHE = ROOT / "models/huggingface"
MIN_PIXELS = 128 * 32 * 32
MAX_PIXELS = 768 * 32 * 32
MAX_NEW_TOKENS = 512
LAW_PROMPT = """Mô tả các hình minh họa trong ảnh. Không cần mô tả quá chi tiết. Chỉ mô tả hình dạng của các hình và chữ xuất hiện trong hình và ảnh. Không cần giải thích ý nghĩa của các hình đó là gì. Nếu ảnh quá mờ không thấy text thì hãy bỏ qua không mô tả text"""


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def blocks(sequence):
    for block in sequence:
        yield block
        for cell in block.get("cells", []):
            yield from blocks(cell["blocks"])
        yield from blocks(block.get("other_blocks", []))


def main():
    preprocessing = json.loads((INPUT / "manifest.json").read_text(encoding="utf-8"))
    if preprocessing["status"] != "COMPLETED" or preprocessing["schema_version"] != 2:
        raise ValueError("Cần corpus chuẩn hóa schema 2 với manifest COMPLETED.")
    records = read_lines(INPUT / "structured_corpus.jsonl")
    images = read_lines(INPUT / "image_manifest.jsonl")
    unique = {}
    for image in images:
        digest = sha256(Path(image["path"]).read_bytes()).hexdigest()
        if digest != image["sha256"] or image["image_id"] != "sha256:" + digest:
            raise ValueError(f'Ảnh đã thay đổi sau chuẩn hóa: {image["path"]}')
        unique.setdefault(image["image_id"], image)
    figures = [b for r in records for b in blocks(r["blocks"]) if b["type"] == "figure"]
    missing = [b["block_id"] for b in figures if b.get("image_id") not in unique]
    if missing:
        raise ValueError(f"Có {len(missing)} block hình thiếu ảnh: {missing[:10]}. Xem issues của preprocessing.")
    config = {
        "model": MODEL, "revision": REVISION, "prompt": LAW_PROMPT,
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS, "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False, "crop": False, "dtype": "bfloat16", "attention": "sdpa",
        "corpus_sha256": sha256((INPUT / "structured_corpus.jsonl").read_bytes()).hexdigest(),
        "images_sha256": sha256((INPUT / "image_manifest.jsonl").read_bytes()).hexdigest(),
    }
    # Library versions also identify the cache so upgrades cannot silently reuse old captions.
    import importlib.metadata
    config["libraries"] = {name: importlib.metadata.version(name)
                           for name in ("torch", "transformers", "bitsandbytes", "Pillow", "accelerate")}
    fingerprint = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run = ROOT / "artifacts/results/text_based_rag/image_captioning"
    previous = {}
    if (run / "manifest.json").exists():
        previous = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if any(previous.get("config", {}).get(key) != config[key]
               for key in ("corpus_sha256", "images_sha256")):
            raise ValueError("Corpus hoặc danh sách ảnh đã thay đổi. Hãy lưu thư mục image_captioning "
                             "cũ sang nơi khác trước khi xử lý bộ dữ liệu mới.")
    (run / "captions").mkdir(parents=True, exist_ok=True)
    stems = Counter(Path(image["filename"]).stem.casefold() for image in unique.values())
    caption_paths = {}
    for image_id, image in unique.items():
        filename = Path(image["filename"]).name
        stem = Path(filename).stem
        name = stem if stems[stem.casefold()] == 1 else filename
        caption_paths[image_id] = run / "captions" / (name + "_description.json")
    assert len({str(path).casefold() for path in caption_paths.values()}) == len(unique), "Trùng tên file caption"
    manifest = {"status": "IN_PROGRESS", "config": config, "input": str(INPUT),
                "started_at": datetime.now(timezone.utc).isoformat(), "unique_images": len(unique)}
    history = dict(previous.get("caption_settings", {}))
    if previous.get("config"):
        old = previous["config"]
        history[sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()] = old
    history[fingerprint] = config
    manifest["caption_settings"] = history
    captions = {}
    for image_id, image in unique.items():
        path = caption_paths[image_id]
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("status") == "COMPLETED" and saved.get("image_id") == image_id
                    and saved.get("description", "").strip()):
                captions[image_id] = saved
    resumed = len(captions)
    incomplete = []
    print(f"[resume] Giữ {resumed} caption đã xong; còn {len(unique) - resumed} ảnh dùng prompt hiện tại.", flush=True)
    started = time.perf_counter()
    stopped = Event()
    phase = "load-model"

    def report():
        elapsed = time.perf_counter() - started
        done = len(captions) + len(incomplete)
        processed = done - resumed
        remaining = len(unique) - done
        eta = f"{elapsed * remaining / processed:.0f}s" if processed else "unknown"
        print(f"[{phase}] {done}/{len(unique)} "
              f"({100 * done / max(1, len(unique)):.1f}%) chưa_xong={len(incomplete)} elapsed={elapsed:.0f}s ETA={eta}", flush=True)

    def heartbeat():
        while not stopped.wait(30):
            report()

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    save(run / "manifest.json", manifest)
    report()
    try:
        if len(captions) < len(unique):
            import torch
            from PIL import Image, ImageOps
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Cần GPU NVIDIA CUDA hỗ trợ BF16 trong .venv.")
            free, total = torch.cuda.mem_get_info()
            budget = min(12 * 1024**3, free - 2 * 1024**3)
            if budget < 9 * 1024**3:
                raise RuntimeError("Cần ít nhất 11 GiB VRAM trống; đóng ứng dụng dùng GPU rồi chạy lại.")
            # shortcut: allocator cap excludes external CUDA allocations; monitor GPU and close competing apps.
            torch.cuda.set_per_process_memory_fraction(budget / total)
            manifest["resources"] = {"gpu": torch.cuda.get_device_name(0), "total_vram_bytes": total,
                                     "free_vram_before_load": free, "pytorch_budget_bytes": budget}
            save(run / "manifest.json", manifest)
            processor = AutoProcessor.from_pretrained(
                MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True, use_fast=False,
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True,
                dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa").eval()
            if not model.is_loaded_in_4bit:
                raise RuntimeError("Dừng: checkpoint không được nạp ở 4-bit.")
            phase = "caption-inference"
            for image_id, item in unique.items():
                if image_id in captions:
                    continue
                path = caption_paths[image_id]
                image_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                with Image.open(item["path"]) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                try:
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": item["path"]}, {"type": "text", "text": LAW_PROMPT}]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                    if inputs.input_ids.shape[1] > 2048:
                        raise ValueError("Input vượt 2048 token; cần kiểm tra kích thước ảnh/prompt.")
                    with torch.inference_mode():
                        output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
                    generated = output[0, inputs.input_ids.shape[1]:]
                    description = processor.decode(generated, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False).strip()
                    result = {"image_id": image_id, "filename": item["filename"],
                              "config_fingerprint": fingerprint, "description": description,
                              "generated_tokens": generated.numel(), "size": list(image.size),
                              "elapsed_seconds": time.perf_counter() - image_started,
                              "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                              "gpu_free_bytes": torch.cuda.mem_get_info()[0],
                              "status": "COMPLETED" if description and generated.numel() < MAX_NEW_TOKENS else "INCOMPLETE"}
                    save(path, result)
                    if result["status"] != "COMPLETED":
                        incomplete.append(item["filename"])
                        print(f'[warning] {item["filename"]}: caption rỗng hoặc chạm giới hạn token; '
                              f'đã lưu {path.name}, tiếp tục ảnh kế tiếp.', flush=True)
                    else:
                        captions[image_id] = result
                    print(f'[image] {item["filename"]}: {description}', flush=True)
                    report()
                    del inputs, output, generated
                    torch.cuda.empty_cache()
                finally:
                    image.close()
        if incomplete:
            manifest.update(status="INCOMPLETE", captioned_images=len(captions),
                            incomplete_images=incomplete, elapsed_seconds=time.perf_counter() - started)
            save(run / "manifest.json", manifest)
            report()
            print(f'[output] {run}\nChưa xuất corpus hoàn chỉnh. Cần chạy lại {len(incomplete)} ảnh: '
                  + ", ".join(incomplete), flush=True)
            return
        phase = "export-corpus"
        for figure in figures:
            figure["description_vlm"] = captions[figure["image_id"]]["description"]
        assert len(captions) == len(unique) and all(b["description_vlm"] for b in figures)
        for name, values in (("image_captions.jsonl", captions.values()), ("structured_corpus.jsonl", records)):
            temporary = run / (name + ".tmp")
            with temporary.open("w", encoding="utf-8") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            temporary.replace(run / name)
        temporary = run / "corpus_preview.md.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(f'\n# {record["document_id"]} / {record["id"]} — {record["title"]}\n')
                preview(record["blocks"], stream)
        temporary.replace(run / "corpus_preview.md")
        manifest.update(status="COMPLETED", captioned_images=len(captions), figure_occurrences=len(figures),
                        elapsed_seconds=time.perf_counter() - started)
        save(run / "manifest.json", manifest)
        report()
        print(f"[output] {run}", flush=True)
    except BaseException as exc:
        manifest.update(status="FAILED", error=str(exc), captioned_images=len(captions))
        save(run / "manifest.json", manifest)
        raise
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
