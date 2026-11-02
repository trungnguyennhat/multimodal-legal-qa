# Multimodal Legal RAG

> **Cách cập nhật README:** Chỉ ghi trạng thái hiện tại của toàn dự án, kết quả đã có và lệnh chạy đúng với code hiện tại. Giữ riêng từng hướng nghiên cứu; không xóa hướng cũ khi thêm hướng mới. Lấy số liệu từ metric/manifest hoặc kết quả private test khi tôi báo, phân biệt benchmark cũ với output mới. Không ghi nhật ký thao tác hoặc thiết kế chưa triển khai; việc sắp làm nằm trong [CHECKPOINTS.md](CHECKPOINTS.md). Quy tắc chung: [AGENTS.md](AGENTS.md).

## Tiến độ và kết quả hiện tại

VLSP 2025 MLQA-TSR: 530 câu train gốc → **418 train + 112 dev**; **100 public**, **146 private**. Corpus schema 4: **402 article, 212 bảng, 787 file ảnh**. Có hai hướng riêng:

| Hướng / thành phần | Trạng thái và kết quả đã có |
| --- | --- |
| `text_based_rag`: ảnh → caption → BGE-M3; corpus + example retrieval | COMPLETED: corpus 830 chunk; train/dev/public/private đã có caption, embedding và retrieval |
| Caption-based retrieval dev, index hiện tại | F2: corpus **0.199681**; example **0.477725**; gated hybrid **0.488460** |
| Caption-based public | Retrieval F2 **0.604277**; QA accuracy **0.76**, 3-shot; 99 câu COMPLETED, 1 câu phục hồi đáp án, không default label |
| Private: submission cũ đã nộp | Benchmark F2 **0.599474**, QA accuracy **0.705479**; không phải điểm của output mới |
| Private hiện tại | Retrieval Task 1 đã COMPLETED và xuất lại 146 câu; chưa có điểm benchmark mới. Inference Task 2 có manifest INCOMPLETE |
| Hybrid: text thuần BGE-M3 | COMPLETED: bỏ 790 figure block, giữ text 212 bảng; 458 chunk 1.024/128; dev F2@5 **0.168180** |
| Hybrid: ảnh toàn khung DINOv2 ViT-B/14 | COMPLETED: 787 ảnh corpus → 228 citation; dev 112 câu / 61 ảnh; F2@5 **0.059457** |
| Hybrid: fusion text thuần + ảnh | COMPLETED: Weighted RRF, text/ảnh 0.5/0.5, k=60; F2@5 **0.140108**; 112 câu, mỗi câu 5 citation cuối |
| Hybrid example retrieval | COMPLETED: text example F2@5 **0.377727**; ảnh example **0.325044**; ranking ảnh corpus+example **0.275478** ở `query_retrieval_hybrid/` |
| YOLOE | Đã preview 11 ảnh VLSP + 12 ảnh chụp; chưa đưa crop vào retrieval, chưa có metric detection chuẩn |

QA caption-based dùng `relevant_articles` được cung cấp trong input, nên accuracy trên **không phải** accuracy của chuỗi retrieval → QA. Lỗi nhãn citation ngoài registry còn tạm hoãn xử lý; khi chấm vẫn giữ nguyên gold. Query embedding train+dev và bốn retrieval corpus/example đã COMPLETED. Ranking ảnh đã được tách thành bước riêng; bước text–ảnh cuối cần chạy lại để đọc ranking đã lưu. Các dòng text/ảnh/fusion chưa dùng example bank là baseline trước mở rộng; điểm example và ranking ảnh mới được ghi riêng.

## Môi trường và dữ liệu

Chạy PowerShell từ gốc project, dùng `.venv`, GPU CUDA. Có sẵn BGE-M3 ở `models/huggingface/hub/`, DINOv2 ở `models/huggingface/dinov2-base/`, Qwen3.5-9B Q4_K_M và projector ở `models/huggingface/Qwen3.5-9B-GGUF/`; llama.cpp ở `.venv/tools/llama.cpp-b11624/`.

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
# Chỉ cần khi chuẩn bị lại dữ liệu gốc.
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data validate
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data prepare
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data split --seed 2025 --dev-ratio 0.2
```

Dữ liệu nguồn: `data/raw/VLSP2025-MLQA-TSR/`; ảnh/split: `data/processed/`. Không commit dữ liệu, model, index, cache hoặc secrets; không sửa `.gitignore`. Tác vụ dài để người dùng tự chạy, log khoảng 30 giây; thiếu VRAM giảm batch xuống 1. `--overwrite` thay kết quả tại đúng output của bước đó.

## Hướng text_based_rag — caption-based, giữ riêng

Corpus preprocessing → caption → chunk → embedding (đã có kết quả; chỉ tái tạo khi cần):

```powershell
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_normalization
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_image_captioning --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_chunking --max-tokens 1024 --overlap-tokens 128 --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_embedding index --batch-size 1 --overwrite
```

Query caption có câu hỏi; embedding dùng câu hỏi + item caption liên quan. Dev dùng 418 train làm example bank; public/private dùng train + dev = 530 câu. Các ngưỡng gate đã chọn trên dev giữ cố định khi chấm test.

```powershell
$split="dev" # Caption/embedding: train, dev, public_test, private_test.
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split $split --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split $split --overwrite
# Retrieval: dev, public_test hoặc private_test; cần bank train/dev tương ứng đã embedding.
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split $split --method corpus --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split $split --method examples --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split $split --method hybrid --hybrid-strategy gated --example-weight 0.6 --final-top-n 5 --overwrite
```

Public embedding cần thêm `--allow-incomplete-captions` khi chấp nhận 3 caption bị cắt như lần chạy đã có. Không cần sinh lại caption nếu artifact vẫn hợp lệ. QA đọc luật gốc + caption corpus + caption query, có few-shot từ train:

```powershell
.\.venv\Scripts\python.exe -u -m text_based_rag.inference --split public_test --few-shot-count 3 --context-size 32768 --allow-incomplete-captions --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.inference --split private_test --few-shot-count 2 --context-size 65536 --overwrite
```

Output: dev/train ở `artifacts/experiments/text_based_rag/`; public ở `artifacts/public-test/text_based_rag/`; private ở `artifacts/private_test/text_based_rag/` (`retrieval/` và `inference/`). Submission private là `retrieval/submission_task1.json`, `inference/submission_task2.json`. Caption/inference có thể resume khi bỏ `--overwrite` và fingerprint giữ nguyên; embedding bị ngắt phải chạy lại.

## Hướng hybrid — text thuần + embedding ảnh riêng

Text chỉ dùng câu hỏi gốc và văn bản luật/bảng, DINOv2 chỉ nhận ảnh. Code đã hỗ trợ **418 train làm example bank, 112 dev để chấm**; query index chứa 530 câu, ảnh deduplicate theo SHA256 (preflight hiện có 298 ảnh duy nhất). Không dùng nhãn dev làm ứng viên. Model và corpus index hiện có vẫn hợp lệ: text `(458,1024)`, ảnh `(787,768)`; **không cần chạy lại corpus embedding**.

Luồng chạy đầy đủ từ gốc project; các bước embedding/retrieval và gộp ảnh đã có kết quả, hiện chỉ cần chạy lại lệnh cuối sau khi tách bước. Các lệnh ghi đè đúng output tương ứng:

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_embedding.py --split dev --batch-size 4 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_embedding.py --split dev --batch-size 4 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_retrieval.py --split dev --method corpus --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_retrieval.py --split dev --method examples --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_retrieval.py --split dev --method corpus --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_retrieval.py --split dev --method examples --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_hybrid.py --split dev --rrf-k 60 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/hybrid_retrieval.py --split dev --tune-on-dev --rrf-k 60 --final-top-n 5 --overwrite
```

Mỗi phương pháp ghi vào `query_retrieval_corpus/` hoặc `query_retrieval_example/` ở nhánh tương ứng. Example lấy top-5 train, gộp max similarity về citation; có thể trả 0–10 citation. Không trộn caption/choices/answer vào embedding, không fine-tune hoặc thêm model/file bank.

Bước `query_image_hybrid.py` gộp **ảnh corpus + ảnh example** bằng RRF 0.5/0.5, k=60, top-10, lưu riêng tại `full_image_method/query_retrieval_hybrid/` với predictions/per_query_metrics/metrics/manifest. Bước `hybrid_retrieval.py` đọc ranking ảnh đã lưu, gộp text corpus/example như trước rồi fusion text–ảnh; không gộp lại ảnh bên trong bước cuối. Ranking ảnh đã kiểm tra giống hoàn toàn kết quả gộp bên trong phiên bản trước trên cả 112 query. `--tune-on-dev` khảo sát trọng số text 0/0.25/0.5/0.75/1, ghi bảng khảo sát và điểm cấu hình 0.5 vào metrics.json, chọn F2@5 cao nhất (hòa ưu tiên text lớn hơn). Đây là chọn cấu hình trên dev, không phải kiểm chứng độc lập. Bỏ flag này và dùng `--text-weight 0.5` để chạy cố định.

Output dưới `artifacts/experiments/hybrid_rag/`: `text_only/`, `full_image_method/`; tên chung `corpus_embedding/`, `query_embedding_dev/`, `query_retrieval_corpus/`, `query_retrieval_example/`. Final ở `full_image_method/image_text_retrieval/`: **`final_citations.json`**, `predictions.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `manifest.json`. Giữ nguyên tên file; có trace train examples, ranking corpus+example từng nhánh và fallback khi example rỗng. Model, query index và bốn danh sách retrieval được giữ nguyên trong lần tách bước này; chưa ghi đè final sau khi đổi input. Không sửa hash để dùng lẫn output cũ/mới.

## YOLOE — preview hiện có

Code `src/hybrid_rag/yolo_e_train/yoloe_preview.py`; model ở `models/yolo_e_train/`, ảnh/crop/JSON ở `temp/sign_preview/yolo_e_train/` và `real_photos/`. Kết quả cũ có fingerprint khác code sau khi tách khỏi YOLO11n; kiểm tra output trước khi chạy lại.

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --device 0
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --real-photos --device 0
```

**Việc tiếp theo:** chạy lại lệnh `hybrid_retrieval.py` cuối cùng để đọc `query_retrieval_hybrid/`, kiểm tra metrics và cập nhật kết quả theo CHECKPOINTS. Không cần chạy lại embedding hoặc corpus/example retrieval chỉ vì lần tách bước này.
