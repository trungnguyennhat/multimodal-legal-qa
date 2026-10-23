"""Download, prepare, fine-tune and evaluate a one-class traffic-sign detector."""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import threading
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/yolo_train"
MODELS = ROOT / "models/yolo_train"
OUTPUT = ROOT / "artifacts/experiments/hybrid_rag/yolo_train"
PREPARED = DATA / "prepared"
REPO = "star092304/Traffic-sign-detection-VietNam"
REVISION = "bd948a572cefbea92a1c523a2784c1ea9cf1dbe1"
ARCHIVE = "traffic-sign-detection-vietnam.zip"
ARCHIVE_SHA256 = "d8c8a777c691c15c15d72b1b2c9b0b2abd55090d1505dd3974f5fbab9ca2bbca"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
VERSION = "8.4.182"
SPLITS = ("train", "val", "test")
EXCLUDED = {54: "Green Light", 55: "Red Light"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class Progress:
    """Heartbeat remains visible while downloads or a model batch are blocking."""

    def __init__(self, stage):
        self.started = self.stage_started = time.monotonic()
        self.stage, self.done, self.total = stage, 0, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def set(self, stage, done=0, total=0):
        if stage != self.stage:
            self.stage_started = time.monotonic()
        self.stage, self.done, self.total = stage, done, total

    def log(self):
        now = time.monotonic()
        done, total = self.done, self.total
        percent = f"{100 * done / total:.1f}%" if total else "N/A"
        eta = (now - self.stage_started) * (total - done) / done if total and done else None
        remaining = f"{max(0, eta):.0f}s" if eta is not None else "chưa đủ dữ liệu"
        print(f"[{self.stage}] {done}/{total or '?'} ({percent}); "
              f"elapsed={now - self.started:.0f}s; ETA bước hiện tại={remaining}", flush=True)

    def heartbeat(self):
        while not self.stop.wait(30):
            self.log()

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.log()


def download_file(url, path, progress, size=None, sha256=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"File đã có nhưng sai kích thước: {path}. Không tự ghi đè.")
        actual = digest(path)
        if sha256 is not None and actual != sha256:
            raise ValueError(f"File đã có nhưng sai SHA256: {path}. Không tự ghi đè.")
        print(f"Tái sử dụng: {path}", flush=True)
        return actual
    part = path.with_name(path.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if size is not None and offset > size:
        raise ValueError(f"File tải dở lớn hơn nguồn: {part}; cần kiểm tra/xóa thủ công.")
    if size is None or offset != size:
        headers = {"User-Agent": "multimodal-legal-qa/yolo-train"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        progress.set(f"download {path.name} (bytes)", offset, size or 0)
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            if offset and response.status == 206:
                if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server trả Content-Range không khớp phần tải dở.")
            elif response.status == 200:
                offset = 0
            else:
                raise ValueError(f"HTTP status không hợp lệ: {response.status}")
            content_length = response.headers.get("Content-Length")
            total = size or (offset + int(content_length) if content_length else 0)
            progress.set(f"download {path.name} (bytes)", offset, total)
            with part.open("ab" if offset else "wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    progress.done += len(block)
            if total and part.stat().st_size != total:
                raise ValueError(f"Tải chưa đủ: {part}. Chạy lại download để tiếp tục.")
    actual = digest(part)
    if sha256 is not None and actual != sha256:
        raise ValueError(f"SHA256 không khớp: {part}; cần kiểm tra/xóa phần tải dở trước khi thử lại.")
    part.replace(path)
    return actual


def download(args):
    manifest = OUTPUT / "download/manifest.json"
    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    results = previous.get("files", {})
    with Progress("download") as progress:
        if args.target in ("dataset", "all"):
            base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
            for name, size, checksum in ((ARCHIVE, 904774951, ARCHIVE_SHA256), ("classid.csv", 4261, None)):
                path = DATA / name
                actual = download_file(base + name, path, progress, size, checksum or results.get(name, {}).get("sha256"))
                results[name] = {"path": str(path), "url": base + name, "sha256": actual}
                save(manifest, {"dataset_revision": REVISION, "files": results})
        if args.target in ("model", "all"):
            path = MODELS / "yolo11n.pt"
            actual = download_file(MODEL_URL, path, progress, 5613764, results.get("yolo11n.pt", {}).get("sha256"))
            if not zipfile.is_zipfile(path):
                raise ValueError(f"Checkpoint không phải archive PyTorch dự kiến: {path}")
            results["yolo11n.pt"] = {"path": str(path), "url": MODEL_URL, "sha256": actual,
                                    "integrity": "local SHA256; upstream release không công bố checksum"}
            save(manifest, {"dataset_revision": REVISION, "files": results})


def prepare(args):
    from PIL import Image

    if PREPARED.exists():
        raise FileExistsError(f"{PREPARED} đã tồn tại. Không tự ghi đè; nếu lần trước lỗi, kiểm tra và dọn thư mục này thủ công.")
    archive = DATA / ARCHIVE
    with Progress("verify dataset SHA256") as progress:
        if digest(archive) != ARCHIVE_SHA256:
            raise ValueError("ZIP sai SHA256; chạy download trước.")
        with (DATA / "classid.csv").open(encoding="utf-8-sig", newline="") as stream:
            classes = {int(row["ID"]): row["classes-en"].strip() for row in csv.DictReader(stream)}
        if set(classes) != set(range(82)) or any(classes[k] != v for k, v in EXCLUDED.items()):
            raise ValueError("Bảng lớp không khớp revision đã xác minh; không tự chuyển class ID.")
        records, seen, duplicates, invalid_images = [], {}, [], []
        stats = {s: {"images": 0, "boxes": 0, "removed_lights": 0, "background_images": 0} for s in SPLITS}
        source_counts = {str(c): 0 for c in classes}
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError("ZIP có tên entry trùng nhau.")
            files = set(names)
            images = sorted(n for n in names if len(PurePosixPath(n).parts) == 4
                            and PurePosixPath(n).parts[0] == "dataset"
                            and PurePosixPath(n).parts[1] in SPLITS
                            and PurePosixPath(n).parts[2] == "images"
                            and PurePosixPath(n).suffix.lower() in (".jpg", ".jpeg", ".png"))
            # Keep evaluation images; remove their copies from lower-priority splits only on request.
            priority = {"test": 0, "val": 1, "train": 2}
            images.sort(key=lambda n: (priority[PurePosixPath(n).parts[1]], n))
            if not images or any(not any(n.startswith(f"dataset/{s}/images/") for n in images) for s in SPLITS):
                raise ValueError("Không tìm thấy đủ train/val/test trong ZIP.")
            progress.set("validate images and boxes", 0, len(images))
            output_labels = set()
            for name in images:
                parts = PurePosixPath(name)
                split, filename = parts.parts[1], parts.name
                if ":" in filename or "\\" in filename:
                    raise ValueError(f"Tên file không an toàn: {name}")
                label_name = f"dataset/{split}/labels/{parts.stem}.txt"
                if label_name not in files:
                    raise ValueError(f"Thiếu nhãn {label_name}; không tự coi là background.")
                if label_name in output_labels:
                    raise ValueError(f"Nhiều ảnh dùng cùng label: {label_name}")
                output_labels.add(label_name)
                raw = source.read(name)
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    image.verify()
                image_hash = hashlib.sha256(raw).hexdigest()
                normalized, image_classes, errors = [], [], []
                for number, line in enumerate(source.read(label_name).decode("utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        errors.append({"line": number, "text": line, "reason": "cần class cx cy w h"})
                        continue
                    try:
                        class_id = int(fields[0])
                        x, y, w, h = map(float, fields[1:])
                    except ValueError:
                        errors.append({"line": number, "text": line, "reason": "class/tọa độ không phải số hợp lệ"})
                        continue
                    if (class_id not in classes or not all(math.isfinite(v) for v in (x, y, w, h))
                            or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                            or min(x-w/2, y-h/2) < -1e-4 or max(x+w/2, y+h/2) > 1.0001):
                        errors.append({"line": number, "text": line, "reason": "class/box không hợp lệ"})
                        continue
                    image_classes.append(class_id)
                    if class_id not in EXCLUDED:
                        normalized.append("0 " + " ".join(fields[1:]))
                if errors:
                    invalid_images.append({"image": name, "label": label_name, "errors": errors})
                    progress.done += 1
                    continue
                if image_hash in seen and seen[image_hash][0] != split:
                    duplicates.append({"first": seen[image_hash][1], "second": name})
                    if args.deduplicate_splits:
                        progress.done += 1
                        continue
                else:
                    seen[image_hash] = (split, name)
                for class_id in image_classes:
                    source_counts[str(class_id)] += 1
                    stats[split]["removed_lights"] += class_id in EXCLUDED
                label = "\n".join(normalized) + ("\n" if normalized else "")
                relative_image = f"{split}/images/{filename}"
                relative_label = f"{split}/labels/{parts.stem}.txt"
                records.append({"source": name, "image": relative_image, "label": relative_label,
                                "image_sha256": image_hash, "label_sha256": hashlib.sha256(label.encode()).hexdigest(),
                                "width": width, "height": height, "boxes": len(normalized), "text": label})
                stats[split]["images"] += 1
                stats[split]["boxes"] += len(normalized)
                stats[split]["background_images"] += not normalized
                progress.done += 1
            actual_labels = {n for n in files if n.startswith(tuple(f"dataset/{s}/labels/" for s in SPLITS)) and n.endswith(".txt")}
            if actual_labels != output_labels:
                raise ValueError("Có nhãn không ghép được ảnh; cần kiểm tra dataset trước khi chuẩn bị.")
            report = {"dataset": REPO, "revision": REVISION, "archive_sha256": ARCHIVE_SHA256,
                      "classes": classes, "excluded": EXCLUDED, "source_box_counts": source_counts,
                      "splits": stats, "exact_cross_split_duplicates": duplicates,
                      "invalid_images": invalid_images, "skip_invalid_images": args.skip_invalid_images,
                      "deduplicate_splits": args.deduplicate_splits,
                      "duplicate_policy": "keep test > val > train; drop lower-priority copies" if args.deduplicate_splits else "error",
                      "near_duplicate_check": "not performed"}
            save(OUTPUT / "prepare/report.json", report)
            if invalid_images and not args.skip_invalid_images:
                raise ValueError(f"Có {len(invalid_images)} ảnh có nhãn lỗi; xem prepare/report.json. "
                                 "Nếu chấp nhận loại toàn bộ các ảnh đó, chạy prepare --skip-invalid-images.")
            if invalid_images:
                print(f"Loại {len(invalid_images)} ảnh có nhãn lỗi cùng toàn bộ box của ảnh; xem prepare/report.json.", flush=True)
            if duplicates and not args.deduplicate_splits:
                raise ValueError("Có ảnh trùng byte giữa các split; xem prepare/report.json. "
                                 "Nếu chấp nhận loại bản trùng theo ưu tiên test > val > train, thêm --deduplicate-splits.")
            if duplicates:
                print(f"Loại {len(duplicates)} bản ảnh trùng giữa các split; giữ ưu tiên test > val > train.", flush=True)
            if any(stats[s]["boxes"] == 0 for s in SPLITS):
                raise ValueError("Có split không còn box biển báo.")
            PREPARED.mkdir(parents=True)
            progress.set("write prepared dataset", 0, len(records))
            for record in records:
                image_path, label_path = PREPARED / record["image"], PREPARED / record["label"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(record["source"]) as stream, image_path.open("wb") as target:
                    shutil.copyfileobj(stream, target)
                label_path.write_text(record.pop("text"), encoding="utf-8", newline="\n")
                progress.done += 1
        dataset_yaml = ('path: ' + json.dumps(PREPARED.as_posix(), ensure_ascii=False) + '\n'
                        'train: train/images\nval: val/images\ntest: test/images\nnames:\n  0: traffic_sign\n')
        (PREPARED / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
        save(PREPARED / "manifest.json", {**report, "status": "COMPLETED", "files": records,
                                        "yaml_sha256": digest(PREPARED / "dataset.yaml")})
        print(json.dumps(stats, ensure_ascii=False, indent=2))


def verify_prepared(progress):
    manifest = PREPARED / "manifest.json"
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if info.get("status") != "COMPLETED" or info.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Dataset chưa chuẩn bị xong hoặc sai nguồn.")
    if digest(PREPARED / "dataset.yaml") != info["yaml_sha256"]:
        raise ValueError("dataset.yaml đã thay đổi; không chạy với manifest cũ.")
    expected = {record[key] for record in info["files"] for key in ("image", "label")}
    actual = {p.relative_to(PREPARED).as_posix() for split in SPLITS for kind in ("images", "labels")
              for p in (PREPARED / split / kind).rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Danh sách ảnh/nhãn khác manifest (có file thêm hoặc thiếu); không tự tiếp tục.")
    progress.set("verify prepared dataset", 0, len(info["files"]))
    for record in info["files"]:
        for key in ("image", "label"):
            path = (PREPARED / record[key]).resolve()
            if not path.is_relative_to(PREPARED.resolve()) or digest(path) != record[key + "_sha256"]:
                raise ValueError(f"Dataset thay đổi hoặc path không hợp lệ: {path}")
        progress.done += 1
    # shortcut: exact hashes only, add perceptual/group checks if source leakage is suspected.
    return digest(manifest)


def load_runtime():
    if importlib.metadata.version("ultralytics") != VERSION:
        raise RuntimeError(f"Cần ultralytics=={VERSION}; cài requirements.txt của bước này.")
    os.environ["YOLO_CONFIG_DIR"] = str(OUTPUT / "settings")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO, settings
    import torch

    integrations = ("sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "wandb")
    settings.update({**{key: False for key in integrations if key in settings},
                     "weights_dir": str(MODELS), "runs_dir": str(OUTPUT)})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return YOLO, torch


def callbacks(model, progress):
    def epoch_start(trainer):
        progress.set(f"train epoch {trainer.epoch + 1}/{trainer.epochs} batches", 0, len(trainer.train_loader))

    def batch_end(_trainer):
        progress.done += 1

    def val_start(validator):
        progress.set("validation batches", 0, len(validator.dataloader))

    model.add_callback("on_train_epoch_start", epoch_start)
    model.add_callback("on_train_batch_end", batch_end)
    model.add_callback("on_val_start", val_start)
    model.add_callback("on_val_batch_end", batch_end)


def run_model(args):
    with Progress(args.command) as progress:
        dataset_hash = verify_prepared(progress)
        resuming = args.command == "train" and args.resume
        folder = OUTPUT / args.command / args.name
        previous = None
        if resuming:
            previous = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if previous.get("command") != "train" or previous.get("status") == "COMPLETED":
                raise ValueError("Chỉ resume lần training bị ngắt, không resume lần đã hoàn thành.")
            if previous.get("dataset_manifest_sha256") != dataset_hash:
                raise ValueError("Dataset khác lần training cũ; không thể resume.")
        weights = (folder / "weights/last.pt" if resuming else args.weights or MODELS / "yolo11n.pt").resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Chưa có checkpoint {weights}; chạy download/train hoặc chỉ định --weights.")
        if folder.exists() and not resuming:
            raise FileExistsError(f"{folder} đã tồn tại; dùng --name mới để giữ nguyên thí nghiệm cũ.")
        exported = MODELS / args.name
        if args.command == "train" and exported.exists():
            raise FileExistsError(f"{exported} đã tồn tại; dùng --name mới.")
        YOLO, torch = load_runtime()
        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng. Sửa môi trường GPU hoặc chủ động dùng --device cpu.")
        folder.mkdir(parents=True, exist_ok=resuming)
        started = time.monotonic()
        info = {"status": "RUNNING", "command": args.command, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "script_sha256": digest(__file__), "dataset_manifest_sha256": dataset_hash,
                "weights": str(weights), "weights_sha256": digest(weights), "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "pillow")},
                "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "0" else None,
                "external_detection_data": True}
        if previous is not None:
            info["resume_history"] = previous.get("resume_history", []) + [
                {key: value for key, value in previous.items() if key != "resume_history"}]
        save(folder / "manifest.json", info)
        try:
            progress.set("load model")
            model = YOLO(str(weights), task="detect")
            if resuming and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
                raise ValueError("last.pt không còn epoch/optimizer để resume; không tự chuyển thành fine-tune mới.")
            callbacks(model, progress)
            common = dict(data=str(PREPARED / "dataset.yaml"), imgsz=args.imgsz, batch=args.batch,
                          device=args.device, workers=0, project=str(folder.parent), name=folder.name,
                          exist_ok=True, plots=False, seed=2025, deterministic=True)
            if args.command == "train":
                def record_training_start(trainer):
                    info["effective_training"] = {key: getattr(trainer.args, key) for key in
                                                  ("epochs", "patience", "imgsz", "batch", "device", "workers",
                                                   "optimizer", "lr0", "amp", "seed", "save_dir")}
                    info["start_epoch"] = trainer.start_epoch + 1
                    info["early_stopping_best_epoch"] = trainer.stopper.best_epoch
                    save(folder / "manifest.json", info)

                model.add_callback("on_train_start", record_training_start)
                if resuming:
                    model.train(resume=str(weights), data=str(PREPARED / "dataset.yaml"),
                                patience=args.patience, device=args.device, workers=0, plots=False,
                                save_dir=str(folder))
                else:
                    model.train(**common, epochs=args.epochs, patience=args.patience, optimizer="AdamW", lr0=0.001,
                                fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
                                amp=False, cache=False, save=True, val=True)
                exported.mkdir(parents=True)
                for name in ("best.pt", "last.pt"):
                    checkpoint = folder / "weights" / name
                    shutil.copy2(checkpoint, exported / name)
                info["exported_weights"] = {name: {"path": str(exported / name), "sha256": digest(exported / name)}
                                            for name in ("best.pt", "last.pt")}
                info["checkpoint_selection"] = "Ultralytics validation fitness; see results.csv"
            else:
                result = model.val(**common, split=args.split, conf=args.conf, iou=args.iou,
                                   max_det=300, save_json=True, save_txt=True, save_conf=True)
                metrics = {"split": args.split, "metrics": {k: float(v) for k, v in result.results_dict.items()},
                           "speed_ms_per_image": {k: float(v) for k, v in result.speed.items()},
                           "confidence_threshold": args.conf, "nms_iou": args.iou,
                           "note": "AP evaluation uses a low confidence threshold; P/R follow Ultralytics best-F1 operating point, not necessarily conf."}
                save(folder / "metrics.json", metrics)
            info["status"] = "COMPLETED"
        except BaseException as error:
            info.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            info["elapsed_seconds"] = time.monotonic() - started
            info["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if args.device == "0" else 0
            save(folder / "manifest.json", info)
        print(f"Kết quả: {folder}", flush=True)
        if args.command == "train":
            print(f"Checkpoint: {exported / 'best.pt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    downloading = commands.add_parser("download", help="Tải dataset/model, tiếp tục file .part khi chạy lại")
    downloading.add_argument("--target", choices=("dataset", "model", "all"), default="all")
    preparing = commands.add_parser("prepare", help="Kiểm tra ZIP, chuyển nhãn một lớp và chuẩn bị dataset")
    preparing.add_argument("--skip-invalid-images", action="store_true",
                           help="Chấp nhận loại toàn bộ ảnh có nhãn sai định dạng/class/box; ghi vào report.json")
    preparing.add_argument("--deduplicate-splits", action="store_true",
                           help="Chấp nhận loại bản ảnh trùng byte ở split ưu tiên thấp hơn: test > val > train")
    for command in ("train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--name", default="yolo11n_signs" if command == "train" else None)
        sub.add_argument("--weights", type=Path, help="Checkpoint local; evaluate nên chỉ định best.pt của lần train")
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--batch", type=int, default=8)
        sub.add_argument("--device", choices=("0", "cpu"), default="0")
        if command == "train":
            sub.add_argument("--resume", action="store_true",
                             help="Chạy tiếp last.pt của --name sau khi dừng tiến trình cũ; chỉ đổi patience/device")
            sub.add_argument("--epochs", type=int, default=100)
            sub.add_argument("--patience", type=int, default=20)
        else:
            sub.add_argument("--split", choices=("val", "test"), default="val")
            sub.add_argument("--conf", type=float, default=0.001)
            sub.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    if args.command == "evaluate" and args.name is None:
        args.name = args.split
    if args.command == "train" and args.resume and args.weights is not None:
        parser.error("--resume dùng weights/last.pt của --name; không truyền --weights.")
    if hasattr(args, "name") and (not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name)):
        parser.error("--name chỉ được chứa chữ ASCII, số, dấu - và _.")
    if any(getattr(args, key, 1) <= 0 for key in ("imgsz", "batch", "epochs", "patience")):
        parser.error("imgsz, batch, epochs, patience phải dương.")
    if args.command == "evaluate" and not (0 < args.conf < 1 and 0 < args.iou < 1):
        parser.error("conf và iou phải nằm trong (0,1).")
    if args.command == "evaluate" and args.weights is None:
        args.weights = MODELS / "yolo11n_signs/best.pt"
    {"download": download, "prepare": prepare, "train": run_model, "evaluate": run_model}[args.command](args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
