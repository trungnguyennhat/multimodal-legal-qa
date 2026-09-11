"""Stage 1 data tools for the official VLSP 2025 MLQA-TSR release."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "VLSP2025-MLQA-TSR"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"

DATA_FILES = {
    "train": Path("dataset/train data/vlsp_2025_train.json"),
    "public_test": Path("dataset/public_test data/vlsp_2025_public_test.json"),
    "law": Path("dataset/law_db/vlsp2025_law.json"),
}

IMAGE_ARCHIVES = {
    "train": Path("dataset/train data/train_images.zip"),
    "public_test": Path("dataset/public_test data/public_test_images.zip"),
    "private_test": Path("dataset/private_test data (post submission)/private_test_images.zip"),
    "law": Path("dataset/law_db/images.zip"),
}

PRIVATE_SUBMISSION_ARCHIVE = Path("dataset/private_test data (post submission)/sample_submission/submission.zip")

IMAGE_PATTERN = re.compile(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>")


def read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSON array and reject incompatible top-level shapes."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được JSON {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} phải là một JSON array chứa các object")
    return value


def load_dataset(source: Path = DEFAULT_SOURCE) -> dict[str, list[dict[str, Any]]]:
    """Load the labeled train/public-test sets and the legal corpus."""
    missing = [str(source / relative) for relative in DATA_FILES.values() if not (source / relative).is_file()]
    if missing:
        raise ValueError("Thiếu file dữ liệu:\n- " + "\n- ".join(missing))
    return {name: read_json_list(source / relative) for name, relative in DATA_FILES.items()}


def load_private_submission(source: Path = DEFAULT_SOURCE) -> dict[str, list[dict[str, Any]]]:
    """Load example private-test predictions; these files are not gold labels."""
    path = source / PRIVATE_SUBMISSION_ARCHIVE
    try:
        with zipfile.ZipFile(path) as archive:
            return {
                "retrieval": json.loads(archive.read("submission_task1.json")),
                "qa": json.loads(archive.read("submission_task2.json")),
            }
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Không đọc được private sample submission {path}: {exc}") from exc


def archive_filenames(path: Path, warnings: list[str] | None = None) -> set[str]:
    """Return real image filenames in a ZIP and reject damaged image payloads."""
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP hỏng tại member: {bad}")
            names: set[str] = set()
            mismatches: Counter[tuple[str, str]] = Counter()
            for info in archive.infolist():
                filename = PurePosixPath(info.filename).name
                if info.is_dir() or info.filename.startswith("__MACOSX/") or filename.startswith("."):
                    continue
                suffix = Path(filename).suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png"}:
                    continue
                header = archive.read(info)[:12]
                detected = (
                    "jpeg" if header.startswith(b"\xff\xd8\xff")
                    else "png" if header.startswith(b"\x89PNG\r\n\x1a\n")
                    else "webp" if header.startswith(b"RIFF") and header[8:12] == b"WEBP"
                    else "gif" if header.startswith((b"GIF87a", b"GIF89a"))
                    else None
                )
                if detected is None:
                    raise ValueError(f"Ảnh hỏng hoặc không nhận diện được trong {path}: {info.filename}")
                expected = "jpeg" if suffix in {".jpg", ".jpeg"} else "png"
                if detected != expected:
                    mismatches[(expected, detected)] += 1
                names.add(filename)
            if mismatches and warnings is not None:
                warnings.append(
                    f"{path.name} có {sum(mismatches.values())} ảnh mang đuôi sai nội dung: {dict(mismatches)}"
                )
            return names
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Không đọc được ZIP {path}: {exc}") from exc


def _normalized_answer(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value).strip())


def validate(source: Path = DEFAULT_SOURCE) -> tuple[list[str], list[str], dict[str, Any]]:
    """Validate structure, references and image archives without changing source data."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = load_dataset(source)
    except ValueError as exc:
        return [str(exc)], warnings, {}

    law_keys: set[tuple[str, str]] = set()
    law_image_refs: set[str] = set()
    law_stats: list[dict[str, Any]] = []
    for law_pos, law in enumerate(data["law"]):
        if not all(key in law for key in ("id", "title", "articles")) or not isinstance(law.get("articles"), list):
            errors.append(f"law[{law_pos}] thiếu id/title/articles hợp lệ")
            continue
        article_ids: Counter[str] = Counter()
        for article_pos, article in enumerate(law["articles"]):
            if not all(key in article for key in ("id", "title", "text")):
                errors.append(f"law[{law_pos}].articles[{article_pos}] thiếu id/title/text")
                continue
            key = (str(law["id"]), str(article["id"]))
            law_keys.add(key)
            article_ids[key[1]] += 1
            law_image_refs.update(IMAGE_PATTERN.findall(str(article["text"])))
        duplicates = {key: count for key, count in article_ids.items() if count > 1}
        if duplicates:
            warnings.append(f"{law['id']} có article_id trùng: {duplicates}")
        law_stats.append({"law_id": law["id"], "articles": len(law["articles"]), "unique_article_ids": len(article_ids)})

    dataset_stats: dict[str, Any] = {}
    for split_name in ("train", "public_test"):
        samples = data[split_name]
        ids: Counter[str] = Counter()
        images: Counter[str] = Counter()
        question_types: Counter[str] = Counter()
        answers: Counter[str] = Counter()
        unknown_refs: Counter[tuple[str, str]] = Counter()
        for position, sample in enumerate(samples):
            required = {"id", "image_id", "question", "relevant_articles", "question_type", "answer"}
            missing = sorted(required - sample.keys())
            if missing:
                errors.append(f"{split_name}[{position}] thiếu field: {missing}")
                continue
            sample_id = str(sample["id"])
            ids[sample_id] += 1
            images[str(sample["image_id"])] += 1
            question_type = str(sample["question_type"])
            answer = _normalized_answer(sample["answer"])
            question_types[question_type] += 1
            answers[answer] += 1
            if question_type == "Multiple choice":
                if not isinstance(sample.get("choices"), dict) or set(sample["choices"]) != {"A", "B", "C", "D"}:
                    errors.append(f"{split_name}:{sample_id} phải có choices A/B/C/D")
                if answer not in {"A", "B", "C", "D"}:
                    errors.append(f"{split_name}:{sample_id} có đáp án multiple-choice không hợp lệ: {answer}")
            elif question_type == "Yes/No":
                if answer not in {"Đúng", "Sai"}:
                    errors.append(f"{split_name}:{sample_id} có đáp án Yes/No không hợp lệ: {answer}")
            else:
                errors.append(f"{split_name}:{sample_id} có question_type không hợp lệ: {question_type}")
            if not isinstance(sample["relevant_articles"], list) or not sample["relevant_articles"]:
                errors.append(f"{split_name}:{sample_id} không có relevant_articles")
                continue
            for ref in sample["relevant_articles"]:
                if not isinstance(ref, dict) or "law_id" not in ref or "article_id" not in ref:
                    errors.append(f"{split_name}:{sample_id} có citation sai schema")
                    continue
                key = (str(ref["law_id"]), str(ref["article_id"]))
                if key not in law_keys:
                    unknown_refs[key] += 1
        duplicated_ids = sorted(key for key, count in ids.items() if count > 1)
        if duplicated_ids:
            errors.append(f"{split_name} có sample id trùng: {duplicated_ids}")

        archive_path = source / IMAGE_ARCHIVES[split_name]
        if not archive_path.is_file():
            errors.append(f"Thiếu archive ảnh: {archive_path}")
            image_names: set[str] = set()
        else:
            try:
                image_names = archive_filenames(archive_path, warnings)
            except ValueError as exc:
                errors.append(str(exc))
                image_names = set()
        missing_images = sorted(f"{image_id}.jpg" for image_id in images if f"{image_id}.jpg" not in image_names)
        if missing_images:
            errors.append(f"{split_name} thiếu {len(missing_images)} ảnh: {missing_images[:10]}")
        if unknown_refs:
            warnings.append(
                f"{split_name} có {sum(unknown_refs.values())} citation chưa khớp corpus "
                f"({len(unknown_refs)} cặp): {dict(unknown_refs.most_common(10))}"
            )
        dataset_stats[split_name] = {
            "samples": len(samples),
            "unique_images": len(images),
            "question_types": dict(question_types),
            "answers": dict(answers),
            "unknown_citations": sum(unknown_refs.values()),
        }

    try:
        private = load_private_submission(source)
        retrieval_ids = [str(row.get("id")) for row in private["retrieval"]]
        qa_ids = [str(row.get("id")) for row in private["qa"]]
        if len(retrieval_ids) != len(set(retrieval_ids)) or len(qa_ids) != len(set(qa_ids)):
            errors.append("Private test có sample id trùng")
        if set(retrieval_ids) != set(qa_ids):
            errors.append("Task 1 và Task 2 private test không có cùng tập sample id")
        private_images = {str(row.get("image_id")) for row in private["qa"]}
        private_archive = source / IMAGE_ARCHIVES["private_test"]
        image_names = archive_filenames(private_archive, warnings)
        missing_images = sorted(f"{image_id}.jpg" for image_id in private_images if f"{image_id}.jpg" not in image_names)
        if missing_images:
            errors.append(f"private_test thiếu {len(missing_images)} ảnh: {missing_images[:10]}")
        unknown_private_refs: Counter[tuple[str, str]] = Counter()
        noncanonical_answers: list[str] = []
        for task_name, rows in private.items():
            for row in rows:
                for ref in row.get("relevant_articles", []):
                    key = (str(ref.get("law_id")), str(ref.get("article_id")))
                    if key not in law_keys:
                        unknown_private_refs[key] += 1
                if task_name == "qa" and _normalized_answer(row.get("answer")) not in {"A", "B", "C", "D", "Đúng", "Sai"}:
                    noncanonical_answers.append(str(row.get("answer")))
        if unknown_private_refs:
            warnings.append(
                f"private_test có {sum(unknown_private_refs.values())} citation chưa khớp corpus "
                f"({len(unknown_private_refs)} cặp): {dict(unknown_private_refs.most_common(10))}"
            )
        if noncanonical_answers:
            warnings.append(
                f"sample submission có {len(noncanonical_answers)} output QA không chuẩn, xác nhận đây là prediction mẫu, không phải gold"
            )
        dataset_stats["private_test"] = {
            "samples": len(private["qa"]),
            "unique_images": len(private_images),
            "retrieval_sample_predictions": len(private["retrieval"]),
            "qa_sample_predictions": len(private["qa"]),
            "qa_noncanonical_outputs": len(noncanonical_answers),
            "question_types": dict(Counter(str(row.get("question_type")) for row in private["qa"])),
            "unknown_citations": sum(unknown_private_refs.values()),
        }
    except ValueError as exc:
        errors.append(str(exc))

    law_archive = source / IMAGE_ARCHIVES["law"]
    if law_archive.is_file():
        try:
            law_images = archive_filenames(law_archive, warnings)
            missing_law_images = sorted(law_image_refs - law_images)
            if missing_law_images:
                errors.append(f"Corpus thiếu {len(missing_law_images)} ảnh được tham chiếu: {missing_law_images[:10]}")
        except ValueError as exc:
            errors.append(str(exc))
    else:
        errors.append(f"Thiếu archive ảnh luật: {law_archive}")

    stats = {
        "source": str(source),
        "datasets": dataset_stats,
        "laws": law_stats,
        "law_image_references": len(law_image_refs),
    }
    return errors, warnings, stats


def split_train(
    source: Path = DEFAULT_SOURCE,
    output: Path = DEFAULT_OUTPUT / "splits",
    seed: int = 2025,
    dev_ratio: float = 0.2,
) -> dict[str, Any]:
    """Create a deterministic image-group split so identical images cannot leak."""
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio phải nằm giữa 0 và 1")
    train = load_dataset(source)["train"]
    image_ids = sorted({str(sample["image_id"]) for sample in train})
    random.Random(seed).shuffle(image_ids)
    dev_image_count = max(1, round(len(image_ids) * dev_ratio))
    dev_images = set(image_ids[:dev_image_count])
    train_rows = [sample for sample in train if str(sample["image_id"]) not in dev_images]
    dev_rows = [sample for sample in train if str(sample["image_id"]) in dev_images]
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.json").write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "dev.json").write_text(json.dumps(dev_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "seed": seed,
        "dev_ratio_requested": dev_ratio,
        "group_key": "image_id",
        "train_samples": len(train_rows),
        "dev_samples": len(dev_rows),
        "train_images": len(image_ids) - len(dev_images),
        "dev_images": len(dev_images),
        "sample_overlap": len({row["id"] for row in train_rows} & {row["id"] for row in dev_rows}),
        "image_overlap": len({row["image_id"] for row in train_rows} & {row["image_id"] for row in dev_rows}),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _safe_extract_flat(archive_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/"):
                continue
            filename = PurePosixPath(info.filename).name
            if not filename or filename.startswith("."):
                continue
            target = destination / filename
            with archive.open(info) as source_file, target.open("wb") as target_file:
                shutil.copyfileobj(source_file, target_file)
            count += 1
    return count


def prepare_images(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT / "images") -> dict[str, int]:
    """Extract official image archives to stable, ignored directories."""
    counts: dict[str, int] = {}
    for name, relative in IMAGE_ARCHIVES.items():
        archive_path = source / relative
        if not archive_path.is_file():
            raise ValueError(f"Thiếu archive ảnh: {archive_path}")
        counts[name] = _safe_extract_flat(archive_path, output / name)
    return counts


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Thư mục official repository")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Kiểm tra schema, citation và archive ảnh")
    subparsers.add_parser("stats", help="In thống kê dữ liệu")
    prepare_parser = subparsers.add_parser("prepare", help="Giải nén ảnh vào data/processed/images")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "images")
    split_parser = subparsers.add_parser("split", help="Chia train/dev theo image_id")
    split_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "splits")
    split_parser.add_argument("--seed", type=int, default=2025)
    split_parser.add_argument("--dev-ratio", type=float, default=0.2)
    args = parser.parse_args(argv)

    try:
        if args.command in {"validate", "stats"}:
            errors, warnings, stats = validate(args.source)
            if args.command == "stats":
                _print_json(stats)
            else:
                print(f"ERRORS: {len(errors)}")
                for item in errors:
                    print(f"- {item}")
                print(f"WARNINGS: {len(warnings)}")
                for item in warnings:
                    print(f"- {item}")
                _print_json(stats)
            return 1 if errors else 0
        if args.command == "prepare":
            _print_json(prepare_images(args.source, args.output))
            return 0
        if args.command == "split":
            _print_json(split_train(args.source, args.output, args.seed, args.dev_ratio))
            return 0
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
