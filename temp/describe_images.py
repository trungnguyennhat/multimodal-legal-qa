"""Describe two local images with Qwen; run manually with the project's .venv."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
import argparse
import json
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
REVISION = "10d842936325905934ea24e64325d774693f166e"
CACHE = ROOT / "models" / "huggingface"
IMAGES = (
    ROOT / "data/processed/images/train/train_1_1.jpg",
    ROOT / "data/processed/images/law/image001.png",
)
MIN_PIXELS = 128 * 32 * 32
MAX_PIXELS = 768 * 32 * 32
MAX_NEW_TOKENS = 384
TRAIN_PROMPT = """Mô tả biển báo trong ảnh. Không cần mô tả quá chi tiết. Chỉ mô tả hình dạng của các biển, và chỉ mô tả hình có trong biển. Không cần giải thích tên gọi hay ý nghĩa của biển đó là gì."""

LAW_PROMPT = """Mô tả các hình minh họa trong ảnh. Không cần mô tả quá chi tiết. Chỉ mô tả hình dạng của các hình. Không cần giải thích ý nghĩa của các hình đó là gì."""


@contextmanager
def progress(step, done, total):
    started = time.perf_counter()
    stopped = Event()

    def report():
        while not stopped.wait(30):
            elapsed = time.perf_counter() - started
            print(
                f"[{step}] {done}/{total} ({100 * done / total:.0f}%) "
                f"elapsed={elapsed:.0f}s ETA=chưa đủ dữ liệu", flush=True,
            )

    print(f"[{step}] {done}/{total} ({100 * done / total:.0f}%) bắt đầu", flush=True)
    worker = Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-image", choices=("train_1_1.jpg", "train_1_2.jpg", "train_1_3.jpg"), default="train_1_1.jpg")
    parser.add_argument("--crop-train", action="store_true", help="Compare full train image with manual crops")
    args = parser.parse_args()
    train_image = IMAGES[0].with_name(args.train_image)
    for image_path in (train_image, IMAGES[1]):
        if not image_path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh: {image_path}")

    with progress("import-dependencies", 0, 1):
        import torch
        import transformers
        from PIL import Image, ImageOps
        import bitsandbytes
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Cần PyTorch CUDA trong .venv và GPU NVIDIA để chạy script này.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU hiện tại không hỗ trợ BF16; script dùng cấu hình BF16 đã có.")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory_budget = min(12 * 1024**3, free_bytes - 2 * 1024**3)
    if memory_budget < 9 * 1024**3:
        raise RuntimeError("Cần ít nhất 11 GiB VRAM trống để giữ 2 GiB headroom. Đóng ứng dụng dùng GPU rồi chạy lại.")
    # shortcut: allocator cap excludes other processes/non-PyTorch allocations; inspect nvidia-smi during inference.
    torch.cuda.set_per_process_memory_fraction(memory_budget / total_bytes)
    print(f"Model: {MODEL}; cache local: {CACHE}", flush=True)
    print(f"Quantization: checkpoint NF4 double quant; vision/lm_head BF16; bitsandbytes={bitsandbytes.__version__}", flush=True)
    print(f"VRAM free={free_bytes / 1024**3:.2f} GiB; PyTorch budget={memory_budget / 1024**3:.2f} GiB", flush=True)
    print(f"torch={torch.__version__}; transformers={transformers.__version__}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}; BF16; SDPA; greedy", flush=True)
    print(f"min_pixels={MIN_PIXELS}; max_pixels={MAX_PIXELS}; max_new_tokens={MAX_NEW_TOKENS}")
    with progress("model-loading", 0, 1):
        processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True, use_fast=False,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, cache_dir=CACHE, local_files_only=True,
            dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        ).eval()
    quantization = model.config.quantization_config
    if not model.is_loaded_in_4bit:
        raise RuntimeError("Checkpoint phải được nạp ở 4-bit; dừng để tránh dùng nhầm BF16 toàn model.")
    print(f"Model revision: {getattr(model.config, '_commit_hash', None)}", flush=True)

    created_at = datetime.now(timezone.utc)
    result_path = ROOT / "temp" / f"qwen_descriptions_{created_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    cases = [(train_image, None, "train-full")]
    if args.crop_train:
        # shortcut: manual coordinates for these three images, use a detector for other images.
        boxes = {
            "train_1_1.jpg": [(428, 134, 503, 208), (466, 0, 623, 134)],
            "train_1_2.jpg": [(214, 153, 317, 249)],
            "train_1_3.jpg": [(655, 0, 1168, 682)],
        }[args.train_image]
        cases.extend((train_image, box, f"train-crop-{i}") for i, box in enumerate(boxes, 1))
    cases.append((IMAGES[1], None, "law-full"))
    results = {
        "created_at": created_at.isoformat(),
        "model": MODEL, "model_revision": getattr(model.config, "_commit_hash", None),
        "bitsandbytes": bitsandbytes.__version__,
        "quantization": quantization.to_dict() if hasattr(quantization, "to_dict") else quantization,
        "gpu_total_bytes": total_bytes, "gpu_free_bytes_before_load": free_bytes,
        "pytorch_memory_budget_bytes": memory_budget,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "attention": "sdpa",
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
        "completed": False, "images": [],
    }

    for index, (image_path, box, case) in enumerate(cases):
        prompt = TRAIN_PROMPT if image_path.parent.name == "train" else LAW_PROMPT
        print(f"\nPrompt cho {image_path.name}:\n{prompt}", flush=True)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        crop_path = None
        with progress(f"inference:{case}", index, len(cases)):
            with Image.open(image_path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            original_size = image.size
            if box is not None:
                expected_size = {
                    "train_1_1.jpg": (680, 454),
                    "train_1_2.jpg": (800, 522),
                    "train_1_3.jpg": (1200, 900),
                }[args.train_image]
                if original_size != expected_size:
                    raise ValueError(f"Manual crops require {args.train_image} at {expected_size}")
                left, top, right, bottom = box
                if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                    raise ValueError(f"Invalid crop: {box}")
                cropped = image.crop(box)
                image.close()
                image = cropped
                crop_path = result_path.with_name(f"{result_path.stem}_{case}.png")
                image.save(crop_path)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            if inputs.input_ids.shape[1] > 2048:
                raise ValueError("Input vượt 2048 token; rút ngắn prompt/giảm MAX_PIXELS để giữ headroom VRAM.")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            generated = output[0, inputs.input_ids.shape[1]:]
            caption = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"\n===== {case} | {image_path.relative_to(ROOT)} | đầu vào {image.width}x{image.height} =====")
        print(caption, flush=True)
        print(
            f"[inference] {index + 1}/{len(cases)} ({100 * (index + 1) / len(cases):.0f}%) "
            f"image_elapsed={time.perf_counter() - started:.1f}s; tokens={generated.numel()}; "
            f"peak_VRAM={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB; "
            f"peak_reserved={torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB; "
            f"GPU_free_now={torch.cuda.mem_get_info()[0] / 1024**3:.2f} GiB", flush=True,
        )
        if generated.numel() >= MAX_NEW_TOKENS:
            print("Lưu ý: output chạm giới hạn token, mô tả có thể chưa kết thúc.", flush=True)
        results["images"].append({
            "case": case, "crop_box": box, "original_size": original_size,
            "crop_image": crop_path.relative_to(ROOT).as_posix() if crop_path else None,
            "image": image_path.relative_to(ROOT).as_posix(),
            "prompt": prompt, "description": caption,
            "width": image.width, "height": image.height,
            "elapsed_seconds": time.perf_counter() - started,
            "generated_tokens": generated.numel(),
            "token_limit_reached": generated.numel() >= MAX_NEW_TOKENS,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_free_bytes_after_inference": torch.cuda.mem_get_info()[0],
        })
        results["completed"] = len(results["images"]) == len(cases)
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[output] Đã lưu: {result_path}", flush=True)
        del inputs, output, generated
        image.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
