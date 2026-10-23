"""Download YOLOE and preview baseline images or a selected set of real query photos."""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from hybrid_rag.yolo_train.detector_training import Progress, VERSION, digest, download_file, save


MODELS = ROOT / "models/yolo_e_train"
PREVIEWS = ROOT / "temp/sign_preview"
OUTPUT = PREVIEWS / "yolo_e_train"
ASSETS = {
    "yoloe-v8l-seg.pt": 107405586,
    "mobileclip_blt.ts": 599764649,
}
PROMPTS = [
    "traffic sign",
    "red circular traffic sign",
    "blue circular traffic sign",
    "red triangle traffic sign",
    "blue rectangle traffic sign",
    "green rectangle traffic sign",
    "white rectangle text traffic sign",
    "supplementary traffic sign",
    "traffic sign with time restrictions",
    "highway direction sign",
]


def download():
    manifest = OUTPUT / "download_manifest.json"
    files = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    with Progress("download YOLOE") as progress:
        for name, size in ASSETS.items():
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}"
            sha256 = download_file(url, MODELS / name, progress, size, files.get(name, {}).get("sha256"))
            files[name] = {"url": url, "path": str(MODELS / name), "bytes": size, "sha256": sha256,
                           "integrity": "local SHA256; upstream release không công bố checksum"}
            save(manifest, files)


def preview(args):
    images = []
    train_root = (ROOT / "data/processed/images/train").resolve()
    output = OUTPUT / "real_photos" if args.real_photos else OUTPUT
    if args.real_photos:
        selection_file = output / "selection.json"
        selected = json.loads(selection_file.read_text(encoding="utf-8"))["images"]
        if not 10 <= len(selected) <= 15:
            raise ValueError("Danh sách real_photos cần 10–15 ảnh.")
        for index, filename in enumerate(selected, 1):
            path = (train_root / filename).resolve()
            if not path.is_file() or not path.is_relative_to(train_root):
                raise ValueError(f"Ảnh query train không hợp lệ: {path}")
            images.append({"folder": f"image_{index:03d}", "path": str(path), "sha256": digest(path),
                           "selection_sha256": digest(selection_file)})
    else:
        for index in range(1, 12):
            previous = PREVIEWS / f"image_{index:03d}/predictions.json"
            metadata = json.loads(previous.read_text(encoding="utf-8"))
            path = Path(metadata["image"]).resolve()
            if not path.is_file() or not path.is_relative_to(train_root):
                raise ValueError(f"Ảnh preview không phải ảnh train hợp lệ: {path}")
            images.append({"folder": f"image_{index:03d}", "path": str(path),
                           "sha256": digest(path), "baseline_predictions_sha256": digest(previous)})
    if len({item["path"] for item in images}) != len(images):
        raise ValueError("Danh sách preview có ảnh trùng.")
    assets = json.loads((OUTPUT / "download_manifest.json").read_text(encoding="utf-8"))
    for name, size in ASSETS.items():
        path = MODELS / name
        if not path.is_file() or path.stat().st_size != size or digest(path) != assets[name]["sha256"]:
            raise ValueError(f"File model thiếu hoặc thay đổi: {path}; chạy download trước.")
    if importlib.metadata.version("ultralytics") != VERSION or importlib.util.find_spec("clip") is None:
        raise RuntimeError("Cần cài requirements.txt của yolo_e_train trong .venv trước.")
    config = {"images": images, "weights_sha256": assets["yoloe-v8l-seg.pt"]["sha256"],
              "text_encoder_sha256": assets["mobileclip_blt.ts"]["sha256"],
              "prompts": PROMPTS, "conf": args.conf, "imgsz": args.imgsz, "iou": 0.7,
              "agnostic_nms": True, "device": args.device, "crop_padding": 0.05,
              "script_sha256": digest(__file__)}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["config"] != config:
        raise ValueError("Output đã có cấu hình khác; không tự ghi đè kết quả. Giữ bản cũ trước khi thử cấu hình mới.")
    os.environ["YOLO_CONFIG_DIR"] = str(output / "settings")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    started = time.monotonic()
    info = {"status": "RUNNING", "config": config, "results": []}
    save(manifest, info)
    with Progress("load YOLOE and text encoder") as progress:
        try:
            import torch
            from PIL import Image, ImageDraw, ImageOps
            from ultralytics import YOLOE, settings
            from ultralytics.nn.text_model import MobileCLIPTS

            integrations = ("sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "wandb")
            settings.update({**{key: False for key in integrations if key in settings},
                             "weights_dir": str(MODELS), "runs_dir": str(output)})
            if args.device == "0" and not torch.cuda.is_available():
                raise RuntimeError("CUDA không khả dụng; không tự chuyển sang CPU.")
            if args.device == "0":
                torch.cuda.reset_peak_memory_stats()
            model = YOLOE(str(MODELS / "yoloe-v8l-seg.pt"))
            # Explicit local encoder avoids set_classes() downloading into the current directory.
            encoder = MobileCLIPTS(device=torch.device("cpu"), weight=str(MODELS / "mobileclip_blt.ts"))
            model.model.clip_model = encoder
            embeddings = model.model.get_text_pe(PROMPTS, cache_clip_model=True)
            model.set_classes(PROMPTS, embeddings=embeddings)
            del model.model.clip_model
            del encoder
            info["packages"] = {name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "torchvision", "clip")}
            progress.set("preview images", 0, len(images))
            for item in images:
                folder = output / item["folder"]
                prediction_file = folder / "predictions.json"
                if prediction_file.exists():
                    saved = json.loads(prediction_file.read_text(encoding="utf-8"))
                    expected = [folder / "original.jpg", folder / "annotated.jpg"] + [folder / d["crop"] for d in saved["detections"]]
                    if any(not path.is_file() for path in expected):
                        raise ValueError(f"Kết quả cũ thiếu file trong {folder}; cần kiểm tra, không tự ghi đè.")
                    info["results"].append({"folder": item["folder"], "image": item["path"], "detections": len(saved["detections"])})
                    progress.done += 1
                    print(f"Tái sử dụng kết quả: {folder}", flush=True)
                    continue
                image_started = time.monotonic()
                folder.mkdir(parents=True, exist_ok=True)
                with Image.open(item["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                result = model.predict(source=image, conf=args.conf, imgsz=args.imgsz, iou=0.7,
                                       device=args.device, agnostic_nms=True, max_det=300,
                                       save=False, verbose=True)[0]
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                detections = []
                for index, box in enumerate(result.boxes, 1):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    confidence, class_id = float(box.conf[0].cpu()), int(box.cls[0].cpu())
                    pad_x, pad_y = (x2-x1)*0.05, (y2-y1)*0.05
                    bounds = (max(0, math.floor(x1-pad_x)), max(0, math.floor(y1-pad_y)),
                              min(image.width, math.ceil(x2+pad_x)), min(image.height, math.ceil(y2+pad_y)))
                    crop_name = f"sign_{index:02d}.jpg"
                    image.crop(bounds).save(folder / crop_name, quality=95)
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                    draw.text((x1, max(0, y1-14)), f"{index}: {confidence:.3f}", fill="red")
                    detections.append({"index": index, "class_id": class_id, "prompt": result.names[class_id],
                                       "confidence": confidence, "xyxy": [x1, y1, x2, y2],
                                       "crop_xyxy_with_5_percent_padding": bounds, "crop": crop_name})
                image.save(folder / "original.jpg", quality=95)
                annotated.save(folder / "annotated.jpg", quality=95)
                save(prediction_file, {"image": item["path"], "image_sha256": item["sha256"],
                                       "image_size": image.size, "conf": args.conf, "imgsz": args.imgsz,
                                       "prompts": PROMPTS, "detections": detections,
                                       "elapsed_seconds": time.monotonic()-image_started})
                info["results"].append({"folder": item["folder"], "image": item["path"], "detections": len(detections)})
                progress.done += 1
                save(manifest, info)
                print(f"[{progress.done}/{len(images)}] {Path(item['path']).name}: {len(detections)} biển -> {folder}", flush=True)
            info["status"] = "COMPLETED"
        except BaseException as error:
            info.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            info["elapsed_seconds"] = time.monotonic()-started
            if "torch" in locals() and args.device == "0" and torch.cuda.is_available():
                info["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            save(manifest, info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Tải checkpoint YOLOE và MobileCLIP text encoder về models/yolo_e_train")
    run = sub.add_parser("preview", help="Chạy đúng 11 ảnh từ temp/sign_preview/image_001..011")
    run.add_argument("--real-photos", action="store_true",
                     help="Chạy bộ 12 ảnh chụp query train mới đã lấy mẫu/kiểm tra; lưu riêng trong real_photos")
    run.add_argument("--conf", type=float, default=0.25)
    run.add_argument("--imgsz", type=int, default=640)
    run.add_argument("--device", choices=("0", "cpu"), default="0")
    args = parser.parse_args()
    if args.command == "download":
        download()
    else:
        if not 0 < args.conf < 1 or args.imgsz <= 0:
            parser.error("Cần 0 < conf < 1 và imgsz > 0.")
        preview(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
