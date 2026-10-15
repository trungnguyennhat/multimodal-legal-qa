"""Caption each train/dev/public-test image + question without using gold labels."""

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import socket
import subprocess
from pathlib import Path
from threading import Event, Thread
import time

from multimodal_legal_rag.data import DATA_FILES, DEFAULT_SOURCE, PRIVATE_TASK1_FILE, read_json_list
from .corpus_image_captioning import (ROOT, CORPUS_MODEL, PROJECTOR, SERVER, CORPUS_MAX_TOKENS,
                                     DESCRIPTION_SCHEMA, IMAGE_CATEGORIES, ITEM_FIELDS, save)

INPUT = ROOT / "data/processed/splits/train.json"
IMAGES = ROOT / "data/processed/images/train"
OUTPUT = ROOT / "artifacts/experiments/text_based_rag/query/query_image_captioning_train"
MAX_NEW_TOKENS = CORPUS_MAX_TOKENS
QUERY_PROMPT = """Bạn là chuyên gia phân tích hệ thống báo hiệu đường bộ Việt Nam.
Nhiệm vụ: Trích xuất mô tả chuẩn hóa từ HÌNH ẢNH để tìm kiếm điều luật, sau đó dùng CÂU HỎI để xác định đối tượng liên quan.

NGUYÊN TẮC ƯU TIÊN BẰNG CHỨNG:
- Quan sát ảnh trước khi xét câu hỏi. Hình dạng, màu sắc, chữ viết, biểu tượng và vị trí phải lấy từ ảnh; không lấy từ lời mô tả hay giả định trong câu hỏi.
- Câu hỏi có thể chứa phát biểu sai hoặc yêu cầu đánh giá đúng/sai. Không coi phát biểu đó là sự thật, không chép nó vào mô tả hoặc ý nghĩa của biển. Không trả lời câu hỏi đúng/sai trong lần trích xuất này.
- Không suy ngược hình dạng hoặc màu sắc từ tên, ý nghĩa hay chữ trên biển; kiểm tra riêng đường viền ngoài, nền, chữ và biểu tượng theo những gì nhìn thấy.
- Nếu chi tiết bị mờ, che khuất hoặc quá nhỏ, ghi "không đọc rõ" hoặc "không xác định từ ảnh" cho chi tiết đó; không tự điền chi tiết quen thuộc.
- Chỉ xuất nội dung tiếng Việt, giữ nguyên chữ/số đọc được trên biển.

BƯỚC 1 - PHÂN LOẠI HÌNH ẢNH (`image_category`):
Xác định hình ảnh thuộc đúng 1 trong 3 loại sau (đồng bộ với cơ sở dữ liệu luật):
- "SIGN_ILLUSTRATION": Ảnh chụp giao thông thực tế hoặc hình minh họa có chứa biển báo hiệu đường bộ.
- "TECHNICAL_DIAGRAM": Bản vẽ kỹ thuật đen trắng quy định kích thước hình học, thông số ký hiệu, hoặc sơ đồ khoảng cách/chiều cao lắp đặt.
- "MARKING_AND_SIGNAL": Ảnh chụp hoặc hình vẽ tập trung vào vạch kẻ đường, đèn tín hiệu, cọc tiêu, rào chắn hoặc người điều khiển giao thông.

BƯỚC 2 - QUY TẮC TRÍCH XUẤT & LỌC NHIỄU:
1. KHÔNG ĐOÁN MÃ BIỂN: Tuyệt đối không tự suy đoán mã số kỹ thuật của biển báo trừ khi mã số đó được viết rõ bằng chữ trong ảnh hoặc trong câu hỏi.
2. NẾU LÀ "SIGN_ILLUSTRATION":
   - Sử dụng từ vựng chuẩn về Nhóm biển, Hình dạng và Màu sắc.
   - Chép đúng chữ viết, con số, đơn vị (tấn, t, m, km, giờ...) thực sự đọc được trên mặt biển chính và biển phụ; phần không đọc được ghi "không đọc rõ".
   - Mỗi biển là một item. Ghi vị trí trong ảnh và quan hệ giữa các biển chỉ khi quan sát được. Hai biển ở gần nhau trong ảnh chưa chắc gắn cùng cột; nếu không rõ, không khẳng định cùng cột.
   - Ý nghĩa chỉ mô tả ở mức nhận diện được, không thêm điều kiện pháp lý hoặc ví dụ tình huống không có bằng chứng; nếu chưa đủ căn cứ, ghi "chưa xác định chắc chắn".
3. NẾU LÀ "TECHNICAL_DIAGRAM":
   - Không mô tả màu đen/trắng của nét vẽ kỹ thuật thành màu sắc thực tế của biển báo.
   - Liệt kê đúng tên hình khối hình học và các chữ cái ký hiệu kích thước/góc độ (như D, L, A, B, C, R, 45°...) xuất hiện trong hình.
4. NẾU LÀ "MARKING_AND_SIGNAL":
   - Mô tả rõ loại vạch kẻ (nét liền/nét đứt, đơn/đôi, màu trắng/vàng), màu đèn tín hiệu hoặc tư thế người điều khiển giao thông.
5. ĐÁNH GIÁ ĐỘ LIÊN QUAN (`is_relevant`):
   - Chỉ sau khi mô tả xong bằng chứng thị giác, đối chiếu câu hỏi để gán `is_relevant`; không sửa mô tả thị giác để khớp câu hỏi.
   - Đặt `true` cho đối tượng được hỏi trực tiếp hoặc quan sát rõ là điều khiển hướng đi/làn đường đang xét; bao gồm biển phụ thực sự đi kèm đối tượng đó.
   - Không coi mọi biển trong cùng cảnh là liên quan. Đặt `false` cho biển phía xa không được hỏi trực tiếp, biển thuộc hướng/làn khác, hoặc khi chưa đủ căn cứ xác định liên quan.

Câu hỏi của người dùng (chỉ dùng để xác định đối tượng liên quan, không dùng làm bằng chứng thị giác):
"{question_text}"

Trước khi xuất JSON, kiểm tra lại từng item có khớp đường viền, màu nền, chữ và vị trí thực sự trong ảnh không. Không đưa các bước phân tích vào output.

Hãy trả về kết quả duy nhất dưới định dạng JSON theo cấu trúc sau:
{
  "image_category": "SIGN_ILLUSTRATION | TECHNICAL_DIAGRAM | MARKING_AND_SIGNAL",
  "summary": "Tóm tắt 1 câu về bối cảnh giao thông hoặc nội dung kỹ thuật trong ảnh gắn với câu hỏi",
  "items": [
    {
      "identifier": "Nhóm biển + Vị trí (ví dụ: 'Biển báo cấm - phía trên', 'Biển phụ - ngay bên dưới') HOẶC Tên hình khối kỹ thuật",
      "visual_or_geometric_features": "Nếu là biển báo cụ thể: ghi Hình dạng + Màu sắc + Biểu tượng. Nếu là bản vẽ kỹ thuật: ghi Hình dạng hình học + Các thông số/ký hiệu kích thước đi kèm trên hình",
      "meaning_or_specification": "Ý nghĩa báo hiệu giao thông",
      "is_relevant": true
    }
  ]
}"""
QUERY_SCHEMA = deepcopy(DESCRIPTION_SCHEMA)
QUERY_SCHEMA["properties"]["items"]["items"]["required"].append("is_relevant")
QUERY_SCHEMA["properties"]["items"]["items"]["properties"]["is_relevant"] = {"type": "boolean"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train', 'dev', 'public_test', 'private_test'), default='train')
    parser.add_argument('--image', help='Run only this image ID or filename, saving a separate review run.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Regenerate every image + question caption and replace mapping/JSONL in the existing split directory.')
    args = parser.parse_args()
    INPUT = ROOT / f'data/processed/splits/{args.split}.json'
    OUTPUT = ROOT / f'artifacts/experiments/text_based_rag/query/query_image_captioning_{args.split}'
    IMAGES = ROOT / 'data/processed/images/train'
    if args.split == 'public_test':
        INPUT = DEFAULT_SOURCE / DATA_FILES['public_test']
        IMAGES = ROOT / 'data/processed/images/public_test'
        OUTPUT = ROOT / 'artifacts/public-test/text_based_rag/query_image_captioning'
    elif args.split == 'private_test':
        INPUT = PRIVATE_TASK1_FILE
        IMAGES = ROOT / 'data/processed/images/private_test'
        OUTPUT = ROOT / 'artifacts/private_test/text_based_rag/retrieval/query_image_captioning'
    rows = read_json_list(INPUT)
    if args.image:
        rows = [r for r in rows if r.get('image_id') == Path(args.image).stem]
        if not rows:
            raise ValueError(f'Không có query cho ảnh {args.image} trong {args.split}.')
        review = OUTPUT.parent / 'query_caption_review' if args.split == 'private_test' else ROOT / 'artifacts/experiments/text_based_rag/query/query_caption_review'
        OUTPUT = review / (
            Path(args.image).stem + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    query_ids, unique, mapping = set(), {}, []
    by_filename, caption_images = {}, set()
    for row in rows:
        query_id, source_image_id = row.get("id"), row.get("image_id")
        if (not isinstance(query_id, str) or not query_id or query_id in query_ids
                or any(char in query_id for char in ('/', '\\', ':')) or query_id in {'.', '..'}):
            raise ValueError(f"Query thiếu ID hoặc trùng ID trong {args.split}.")
        if (not isinstance(source_image_id, str) or not source_image_id
                or Path(source_image_id).name != source_image_id
                or any(char in source_image_id for char in ("/", "\\", ":"))):
            raise ValueError(f"image_id không hợp lệ: {query_id}")
        query_ids.add(query_id)
        filename = source_image_id + ".jpg"
        path = (IMAGES / filename).resolve()
        if path.parent != IMAGES.resolve() or not path.is_file():
            raise FileNotFoundError(f"Thiếu ảnh {args.split}: {filename}")
        if filename not in by_filename:
            image_id = "sha256:" + sha256(path.read_bytes()).hexdigest()
            by_filename[filename] = image_id
        image_id = by_filename[filename]
        question = row.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'Query thiếu câu hỏi: {query_id}')
        unique[query_id] = {'query_id': query_id, 'image_id': image_id, 'filename': filename,
                            'path': str(path), 'question_text': question}
        caption_file = (source_image_id if not args.image and image_id not in caption_images else query_id) + "_description.json"
        caption_images.add(image_id)
        mapping.append({"query_id": query_id, "source_image_id": source_image_id,
                        "filename": filename, "image_id": image_id,
                        "caption_file": "captions/" + caption_file})
    if not unique:
        raise ValueError(f"{args.split} không có ảnh để caption.")
    signature = sha256(json.dumps(mapping, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    settings = {"model": str(CORPUS_MODEL), "projector": str(PROJECTOR), "prompt": QUERY_PROMPT,
                "max_new_tokens": MAX_NEW_TOKENS, "crop": False, "temperature": 0, "seed": 2025,
                "context_size": 16384, "enable_thinking": False, "response_schema": QUERY_SCHEMA,
                "input_fields": ["image", "question"], "caption_unit": "query_id",
                "runtime": "llama.cpp-b11624-win-cuda-13.4", "libraries": {"requests": version("requests")}}
    for model_path in (CORPUS_MODEL, PROJECTOR, SERVER):
        if not model_path.is_file():
            raise FileNotFoundError(f'Thiếu model/projector/runtime: {model_path}')
    previous_path = OUTPUT / "manifest.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if not args.overwrite and (previous.get("train_images_signature") != signature or previous.get('settings') != settings
                                  or previous.get('input_sha256') != sha256(INPUT.read_bytes()).hexdigest()):
            raise ValueError("Query/ảnh/prompt/model đã thay đổi. Dùng --overwrite để sinh lại toàn bộ caption; "
                             "dùng --image để xem thử riêng.")
    (OUTPUT / "captions").mkdir(parents=True, exist_ok=True)
    paths = {item['query_id']: OUTPUT / item['caption_file'] for item in mapping}
    assert len({str(path).casefold() for path in paths.values()}) == len(unique), "Trùng tên checkpoint"
    print(f'[input] split={args.split}; queries={len(rows)}; image_files={len(by_filename)}; '
          f'unique_image_hashes={len(set(by_filename.values()))}; source={INPUT}', flush=True)
    print(f'[output] {OUTPUT}; mode={"overwrite-all-queries" if args.overwrite else "resume"}; '
          f'max_new_tokens={MAX_NEW_TOKENS}', flush=True)
    results, incomplete = {}, []
    legacy_captions = 0
    for query_id, path in paths.items():
        if path.exists() and not args.overwrite:
            saved = json.loads(path.read_text(encoding="utf-8"))
            item = unique[query_id]
            if saved.get("image_id") != item['image_id']:
                raise ValueError(f"Ảnh đã thay đổi: {path.name}")
            # Interrupted overwrites leave untouched image-only captions beside new query captions.
            if saved.get('query_id') is None and saved.get('settings', {}).get('caption_unit') != 'query_id':
                legacy_captions += 1
                continue
            if (saved.get('query_id') != query_id
                    or saved.get('question_text') != item['question_text'] or saved.get('settings') != settings):
                raise ValueError(f"Query/ảnh/cấu hình đã thay đổi: {path.name}")
            if saved.get("status") == "COMPLETED" and saved.get("description", "").strip():
                results[query_id] = saved
    if legacy_captions:
        print(f'[resume] {legacy_captions} caption theo ảnh cũ chưa được thay thế; sẽ sinh lại theo query.', flush=True)
    resumed = len(results)
    manifest = {"status": "IN_PROGRESS", "split": args.split, "input": str(INPUT),
                "input_sha256": sha256(INPUT.read_bytes()).hexdigest(),
                "train_images_signature": signature, "settings": settings,
                "queries": len(rows), "image_files": len(by_filename), "unique_image_hashes": len(set(by_filename.values())),
                "started_at": datetime.now(timezone.utc).isoformat(), "resumed_queries": resumed,
                "scope": "selected_image" if args.image else "full_split"}
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
    process = None
    server_log = None
    try:
        save(OUTPUT / "manifest.json", manifest)
        temporary = OUTPUT / "query_image_mapping.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for item in mapping:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        temporary.replace(OUTPUT / "query_image_mapping.jsonl")
        print(f"[resume] split={args.split}; giữ {resumed} query đã xong; caption theo ảnh + câu hỏi.", flush=True)
        report()
        if len(results) < len(unique):
            import requests
            from PIL import Image
            manifest["resources"] = {"backend": "llama.cpp CUDA", "gpu_layers": "all"}
            save(OUTPUT / "manifest.json", manifest)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            url = f'http://127.0.0.1:{port}'
            server_log = (OUTPUT / 'llama-server.log').open('w', encoding='utf-8')
            command = [str(SERVER), '-m', str(CORPUS_MODEL), '--mmproj', str(PROJECTOR),
                       '--host', '127.0.0.1', '--port', str(port), '-c', '16384', '-ngl', 'all',
                       '--parallel', '1', '--jinja', '--reasoning', 'off']
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.perf_counter() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'llama-server dừng; xem {OUTPUT / "llama-server.log"}')
                try:
                    if requests.get(url + '/health', timeout=2).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                if time.perf_counter() > deadline:
                    raise TimeoutError('Model chưa sẵn sàng sau 180s; xem llama-server.log.')
                time.sleep(1)
            phase = f"{args.split}-image-caption-inference"
            for query_id, item in unique.items():
                if query_id in results:
                    continue
                image_started = time.perf_counter()
                prompt = QUERY_PROMPT.replace('{question_text}', item['question_text'])
                if args.image:
                    (OUTPUT / (query_id + '_prompt.txt')).write_text(prompt, encoding='utf-8')
                # Some dataset .jpg files contain WebP; send lossless PNG bytes the runtime can decode.
                with Image.open(item['path']) as original, BytesIO() as buffer:
                    with original.convert('RGB') as image:
                        image.save(buffer, format='PNG')
                    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
                response = requests.post(url + '/v1/chat/completions', json={
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}},
                        {'type': 'text', 'text': prompt}]}],
                    'temperature': 0, 'seed': 2025, 'max_tokens': MAX_NEW_TOKENS,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'response_format': {'type': 'json_object', 'schema': QUERY_SCHEMA}}, timeout=(10, 900))
                if not response.ok:
                    raise requests.HTTPError(
                        f'Query {query_id}, ảnh {item["filename"]}: HTTP {response.status_code}: '
                        f'{response.text[:2000]}', response=response)
                generated = response.json()
                choice = generated['choices'][0]
                description = choice['message']['content']
                result = {'query_id': query_id, 'image_id': item['image_id'], 'filename': item['filename'],
                          'question_text': item['question_text'], 'description': description, 'settings': settings,
                          'status': 'INCOMPLETE', 'finish_reason': choice['finish_reason'],
                          'generated_tokens': generated['usage']['completion_tokens'],
                          'elapsed_seconds': time.perf_counter() - image_started}
                try:
                    payload = json.loads(description)
                    if (not isinstance(payload, dict) or set(payload) != {'image_category', 'summary', 'items'}
                            or payload['image_category'] not in IMAGE_CATEGORIES
                            or not isinstance(payload['summary'], str) or not payload['summary'].strip()
                            or not isinstance(payload['items'], list)
                            or any(not isinstance(sign, dict) or set(sign) != {*ITEM_FIELDS, 'is_relevant'}
                                   or any(not isinstance(sign[key], str) or not sign[key].strip() for key in ITEM_FIELDS)
                                   or type(sign['is_relevant']) is not bool for sign in payload['items'])):
                        raise ValueError('JSON query không khớp schema hoặc is_relevant không phải boolean.')
                    if choice['finish_reason'] == 'stop':
                        result['status'] = 'COMPLETED'
                    if args.image:
                        save(OUTPUT / (query_id + '_query_description.json'), payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    result['validation_error'] = str(exc)
                save(paths[query_id], result)
                if result['status'] != 'COMPLETED':
                    incomplete.append(query_id)
                    print(f'[warning] {query_id}: JSON lỗi/bị cắt; tiếp tục query sau.', flush=True)
                else:
                    results[query_id] = result
                print(f'[query] {query_id} | {item["question_text"]}\n{description}', flush=True)
                report()
        phase = "export-captions"
        temporary = OUTPUT / "image_captions.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for query_id in unique:
                value = json.loads(paths[query_id].read_text(encoding="utf-8"))
                assert value['query_id'] == query_id and value["status"] in {"COMPLETED", "INCOMPLETE"}
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        temporary.replace(OUTPUT / "image_captions.jsonl")
        manifest.update(status="INCOMPLETE" if incomplete else "COMPLETED", captioned_queries=len(results),
                        incomplete_queries=incomplete, elapsed_seconds=time.perf_counter() - started)
        save(OUTPUT / "manifest.json", manifest)
        report()
        print(f"[output] {OUTPUT}; status={manifest['status']}", flush=True)
    except BaseException as exc:
        manifest.update(status="FAILED", error=str(exc), captioned_queries=len(results), incomplete_queries=incomplete)
        save(OUTPUT / "manifest.json", manifest)
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
