"""Caption corpus images locally, checkpoint each image, and enrich figure blocks."""

from collections import Counter
import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
import json
import mimetypes
import socket
import subprocess
import time

from .corpus_normalization import OUTPUT as INPUT, ROOT, preview

MODEL = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
REVISION = "10d842936325905934ea24e64325d774693f166e"
CACHE = ROOT / "models/huggingface"
MIN_PIXELS = 128 * 32 * 32
MAX_PIXELS = 768 * 32 * 32
MAX_NEW_TOKENS = 512
CORPUS_MODEL = ROOT / "models/huggingface/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
PROJECTOR = CORPUS_MODEL.with_name("mmproj-F16.gguf")
SERVER = ROOT / ".venv/tools/llama.cpp-b11624/llama-server.exe"
CORPUS_MAX_TOKENS = 4096
LAW_PROMPT = """Bạn là chuyên gia số hóa quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ Việt Nam.
Nhiệm vụ: Kết hợp HÌNH ẢNH đính kèm và VĂN BẢN ĐIỀU LUẬT gốc để trích xuất mô tả chuẩn hóa phục vụ tìm kiếm văn bản.

Nội dung điều luật gốc:
"{law_article_text}"

BƯỚC 1 - PHÂN LOẠI HÌNH ẢNH (`image_category`):
Hãy xác định hình ảnh thuộc đúng 1 trong 3 loại sau:
- "SIGN_ILLUSTRATION": Hình vẽ mẫu biển báo giao thông cụ thể (có màu sắc, biểu tượng giao thông bên trong).
- "TECHNICAL_DIAGRAM": Bản vẽ kỹ thuật đen trắng quy định kích thước hình học, thông số ký hiệu, hoặc sơ đồ khoảng cách/chiều cao lắp đặt.
- "MARKING_AND_SIGNAL": Hình vẽ vạch kẻ đường, đèn tín hiệu, cọc tiêu, rào chắn hoặc hiệu lệnh người điều khiển giao thông.

BƯỚC 2 - QUY TẮC TRÍCH XUẤT THEO TỪNG LOẠI:
1. Nếu là "SIGN_ILLUSTRATION":
   - Sử dụng từ vựng chuẩn về Nhóm biển, Hình dạng  và Màu sắc.
   - Nêu rõ mã biển (trích từ điều luật), hình vẽ biểu tượng bên trong và ý nghĩa.
2. Nếu là "TECHNICAL_DIAGRAM":
   - TUYỆT ĐỐI KHÔNG mô tả màu đen/trắng của nét vẽ kỹ thuật thành màu sắc của biển báo.
   - Không coi các chữ cái ký hiệu kích thước là chữ in trên biển báo thực tế.
   - Liệt kê các hình khối xuất hiện trong bản vẽ và giải thích ý nghĩa của từng ký hiệu kích thước/góc độ dựa trên văn bản điều luật đi kèm.
3. Nếu là "MARKING_AND_SIGNAL":
   - Mô tả hình thái vạch kẻ, tín hiệu đèn hoặc tư thế điều khiển giao thông và tác dụng pháp lý.

Hãy trả về kết quả duy nhất dưới định dạng JSON theo cấu trúc sau:
{
  "image_category": "SIGN_ILLUSTRATION | TECHNICAL_DIAGRAM | MARKING_AND_SIGNAL",
  "summary": "Tóm tắt 1 câu về nội dung chính của hình ảnh trong ngữ cảnh điều luật",
  "items": [
    {
      "identifier": "Mã biển (nếu là SIGN_ILLUSTRATION) hoặc Tên hình khối/đối tượng kỹ thuật",
      "visual_or_geometric_features": "Nếu là biển báo cụ thể: ghi Hình dạng + Màu sắc + Biểu tượng. Nếu là bản vẽ kỹ thuật: ghi Hình dạng hình học + Các thông số/ký hiệu kích thước đi kèm trên hình",
      "meaning_or_specification": "Ý nghĩa báo hiệu giao thông HOẶC quy định kỹ thuật của các thông số đó trong điều luật"
    }
  ]
}"""
IMAGE_CATEGORIES = ("SIGN_ILLUSTRATION", "TECHNICAL_DIAGRAM", "MARKING_AND_SIGNAL")
ITEM_FIELDS = ("identifier", "visual_or_geometric_features", "meaning_or_specification")
DESCRIPTION_SCHEMA = {"type": "object", "required": ["image_category", "summary", "items"],
                      "additionalProperties": False, "properties": {
                          "image_category": {"type": "string", "enum": list(IMAGE_CATEGORIES)},
                          "summary": {"type": "string"}, "items": {"type": "array", "items": {
                              "type": "object", "required": list(ITEM_FIELDS), "additionalProperties": False,
                              "properties": {key: {"type": "string"} for key in ITEM_FIELDS}}}}}


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Caption only this corpus image filename, saving a separate run.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate every selected caption and replace existing caption output.")
    args = parser.parse_args()
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
    contexts = {}
    for record in records:
        article_text = "\n".join(b["text"] for b in blocks(record["blocks"]) if b.get("text"))
        context = f'{record["document_id"]} / Điều {record["id"]} – {record["title"]}\n{article_text}'
        for image_id in {b["image_id"] for b in blocks(record["blocks"]) if b["type"] == "figure"}:
            contexts.setdefault(image_id, []).append(context)
    if args.image:
        unique = {image_id: image for image_id, image in unique.items() if image["filename"] == args.image}
        if not unique:
            raise ValueError(f"Không tìm thấy ảnh trong corpus: {args.image}")
    for path in (CORPUS_MODEL, PROJECTOR, SERVER):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu model/projector/runtime: {path}")
    from .corpus_embedding import file_hash
    config = {
        "model": str(CORPUS_MODEL), "projector": str(PROJECTOR), "prompt": LAW_PROMPT,
        "max_new_tokens": CORPUS_MAX_TOKENS, "temperature": 0, "seed": 2025,
        "enable_thinking": False, "crop": False, "context_size": 16384,
        "input_fields": ["image", "law_article_text"], "response_schema": DESCRIPTION_SCHEMA,
        "corpus_sha256": sha256((INPUT / "structured_corpus.jsonl").read_bytes()).hexdigest(),
        "images_sha256": sha256((INPUT / "image_manifest.jsonl").read_bytes()).hexdigest(),
    }
    # Library versions also identify the cache so upgrades cannot silently reuse old captions.
    import importlib.metadata
    config["libraries"] = {"requests": importlib.metadata.version("requests")}
    config["runtime"] = "llama.cpp-b11624-win-cuda-13.4"
    fingerprint = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run = ROOT / "artifacts/experiments/text_based_rag/corpus/corpus_image_captioning"
    if args.image:
        run = ROOT / "artifacts/experiments/text_based_rag/corpus/corpus_caption_review" / (
            Path(args.image).stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    previous = {}
    if (run / "manifest.json").exists():
        previous = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("config") != config and not args.overwrite:
            raise ValueError("Prompt/model/config đã thay đổi; dùng --overwrite để chạy lại toàn bộ "
                             "caption trong corpus_image_captioning hoặc --image để xem riêng một ảnh.")
        if not args.overwrite and any(previous.get("config", {}).get(key) != config[key]
               for key in ("corpus_sha256", "images_sha256")):
            raise ValueError("Corpus hoặc danh sách ảnh đã thay đổi. Hãy lưu thư mục corpus_image_captioning "
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
    captions = {}
    for image_id, image in unique.items():
        path = caption_paths[image_id]
        if path.exists() and not args.overwrite:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved.get("status") == "COMPLETED" and saved.get("image_id") == image_id
                    and saved.get("config_fingerprint") == fingerprint
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
    process = None
    server_log = None
    save(run / "manifest.json", manifest)
    report()
    try:
        if len(captions) < len(unique):
            import requests
            manifest["resources"] = {"backend": "llama.cpp CUDA", "gpu_layers": "all"}
            if args.image:
                phase = "verify-model-and-runtime"
                manifest["model_sha256"] = {p.name: file_hash(p) for p in (CORPUS_MODEL, PROJECTOR, SERVER)}
            save(run / "manifest.json", manifest)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            if args.image:
                server_log = (run / "llama-server.log").open("w", encoding="utf-8")
            command = [str(SERVER), "-m", str(CORPUS_MODEL), "--mmproj", str(PROJECTOR),
                       "--host", "127.0.0.1", "--port", str(port), "-c", "16384", "-ngl", "all",
                       "--parallel", "1", "--jinja", "--reasoning", "off"]
            process = subprocess.Popen(command, stdout=server_log or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            phase = "load-qwen35-gguf"
            deadline = time.perf_counter() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server dừng với exit code {process.returncode}; "
                                       "kiểm tra runtime/model và VRAM; dùng --image image002.jpg để xem log riêng.")
                try:
                    if requests.get(url + "/health", timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.perf_counter() > deadline:
                    raise TimeoutError("Model chưa sẵn sàng sau 180s; kiểm tra VRAM/runtime, "
                                       "dùng --image image002.jpg để xem log riêng.")
                time.sleep(1)
            phase = "caption-inference"
            for image_id, item in unique.items():
                if image_id in captions:
                    continue
                path = caption_paths[image_id]
                image_started = time.perf_counter()
                law_article_text = "\n\n".join(contexts[image_id])
                prompt = LAW_PROMPT.replace("{law_article_text}", law_article_text)
                if args.image:
                    (run / (Path(path.name).stem + "_prompt.txt")).write_text(prompt, encoding="utf-8")
                encoded = base64.b64encode(Path(item["path"]).read_bytes()).decode("ascii")
                media_type = mimetypes.guess_type(item["filename"])[0] or "application/octet-stream"
                messages = [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64," + encoded}},
                    {"type": "text", "text": prompt}]}]
                response = requests.post(url + "/v1/chat/completions", json={
                    "messages": messages, "temperature": 0, "seed": 2025,
                    "max_tokens": CORPUS_MAX_TOKENS, "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_object", "schema": DESCRIPTION_SCHEMA}}, timeout=(10, 900))
                response.raise_for_status()
                generated = response.json()
                if args.image:
                    save(run / (Path(path.name).stem + "_response.json"), generated)
                choice = generated["choices"][0]
                description = choice["message"]["content"]
                result = {"image_id": image_id, "filename": item["filename"],
                          "config_fingerprint": fingerprint, "description": description,
                          "generated_tokens": generated["usage"]["completion_tokens"],
                          "elapsed_seconds": time.perf_counter() - image_started,
                          "finish_reason": choice["finish_reason"], "status": "INCOMPLETE"}
                try:
                    payload = json.loads(description)
                    if (not isinstance(payload, dict) or set(payload) != {"image_category", "summary", "items"}
                            or payload["image_category"] not in IMAGE_CATEGORIES
                            or not isinstance(payload["summary"], str) or not payload["summary"].strip()
                            or not isinstance(payload["items"], list)
                            or any(not isinstance(item, dict) or set(item) != set(ITEM_FIELDS)
                                   or any(not isinstance(item[key], str) or not item[key].strip() for key in ITEM_FIELDS)
                                   for item in payload["items"])):
                        raise ValueError("JSON không khớp schema image_category/summary/items.")
                    if choice["finish_reason"] == "stop":
                        result["status"] = "COMPLETED"
                    if args.image:
                        save(run / (Path(item["filename"]).stem + "_corpus_description.json"), payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    result["validation_error"] = str(exc)
                save(path, result)
                if result["status"] != "COMPLETED":
                    incomplete.append(item["filename"])
                    print(f'[warning] {item["filename"]}: JSON không hợp lệ hoặc bị cắt; xem {path}.', flush=True)
                else:
                    captions[image_id] = result
                print(f'[image] {item["filename"]}: {description}', flush=True)
                report()
        if args.image:
            manifest.update(status="INCOMPLETE" if incomplete else "COMPLETED", scope="selected_image",
                            captioned_images=len(captions), incomplete_images=incomplete,
                            elapsed_seconds=time.perf_counter() - started)
            save(run / "manifest.json", manifest)
            print(f"[output] {run}", flush=True)
            return
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
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if server_log is not None:
            server_log.close()
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
