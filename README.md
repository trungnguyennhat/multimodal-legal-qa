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
| Hybrid: fusion text thuần + ảnh, có example bank | COMPLETED: F2@5 **0.402659**, text/ảnh **0.75/0.25**, k=60, chọn trên dev; 112 câu, mỗi câu 5 citation cuối. Baseline corpus-only trước mở rộng: **0.140108** |
| Hybrid example retrieval | COMPLETED: text example F2@5 **0.377727**; ảnh example **0.325044**; corpus+example sau gộp: text **0.365551**, ảnh **0.275478**, lưu tại `query_retrieval_hybrid/` của mỗi nhánh |
| YOLOE preview | Đã preview 11 ảnh VLSP + 12 ảnh chụp; crop retrieval có kết quả ở nhánh riêng bên dưới, chưa có metric detection chuẩn |
| Hybrid: crop-image YOLOE → DINOv2 | COMPLETED: corpus **772 crop/vector**, query train+dev **1.512 crop/vector**; dev F2@5 corpus **0.091445**, example **0.430145**, image hybrid **0.378778**, final text–crop **0.425158** (text/ảnh 0.75/0.25 cố định) |

QA caption-based dùng `relevant_articles` được cung cấp trong input, nên accuracy trên **không phải** accuracy của chuỗi retrieval → QA. Lỗi nhãn citation ngoài registry còn tạm hoãn xử lý; khi chấm vẫn giữ nguyên gold. Cả full-image và crop-image đã COMPLETED các bước embedding, corpus/example retrieval, image hybrid và fusion text–ảnh. Các điểm corpus và điểm có example bank được ghi riêng; cấu hình chọn trên dev chưa phải test độc lập.

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

Luồng chạy đầy đủ từ gốc project đã có kết quả; chỉ chạy lại khi cần tái lập. Các lệnh ghi đè đúng output tương ứng:

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_embedding.py --split dev --batch-size 4 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_embedding.py --split dev --batch-size 4 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_retrieval.py --split dev --method corpus --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_retrieval.py --split dev --method examples --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_retrieval.py --split dev --method corpus --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_retrieval.py --split dev --method examples --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_text_hybrid.py --split dev --rrf-k 60 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_hybrid.py --split dev --rrf-k 60 --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/hybrid_retrieval.py --split dev --tune-on-dev --rrf-k 60 --final-top-n 5 --overwrite
```

Mỗi phương pháp ghi vào `query_retrieval_corpus/` hoặc `query_retrieval_example/` ở nhánh tương ứng. Example lấy top-5 train, gộp max similarity về citation; có thể trả 0–10 citation. Không trộn caption/choices/answer vào embedding, không fine-tune hoặc thêm model/file bank.

Bước `query_text_hybrid.py` gộp **text corpus + text example**, `query_image_hybrid.py` gộp **ảnh corpus + ảnh example**, cùng dùng RRF 0.5/0.5, k=60, top-10. Mỗi bước lưu predictions/per_query_metrics/metrics/manifest tại `query_retrieval_hybrid/` của nhánh tương ứng. Bước `hybrid_retrieval.py` chỉ đọc hai ranking đã lưu rồi fusion text–ảnh; k phải khớp với cả hai bước trước. Đã kiểm tra hai ranking giống kết quả gộp bên trong phiên bản trước trên cả 112 query. `--tune-on-dev` khảo sát trọng số text 0/0.25/0.5/0.75/1, ghi bảng khảo sát và điểm cấu hình 0.5 vào metrics.json, chọn F2@5 cao nhất (hòa ưu tiên text lớn hơn). Đây là chọn cấu hình trên dev, không phải kiểm chứng độc lập. Bỏ flag này và dùng `--text-weight 0.5` để chạy cố định.

Output dưới `artifacts/experiments/hybrid_rag/`: `text_only/`, `full_image_method/`; tên chung `corpus_embedding/`, `query_embedding_dev/`, `query_retrieval_corpus/`, `query_retrieval_example/`, `query_retrieval_hybrid/`. Final ở `full_image_method/image_text_retrieval/`: **`final_citations.json`**, `predictions.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `manifest.json`. Giữ nguyên tên file; có trace train examples, ranking corpus+example từng nhánh và fallback khi example rỗng. Final hiện tại đã đọc hai ranking lưu riêng; giữ nguyên toàn bộ kết quả làm mốc so sánh crop. Không sửa hash để dùng lẫn output cũ/mới.

## Hướng crop-image — YOLOE → DINOv2

Code ở `src/hybrid_rag/crop_image_method/`, output chỉ ở `artifacts/experiments/hybrid_rag/crop_image_method/`. Dùng YOLOE-v8l-seg + MobileCLIP trong `models/yolo_e_train/`, DINOv2 hiện có; CUDA, detection batch 1, embedding batch 4. 15 prompt cố định có dạng chung và ngoại lệ; cùng cấu hình cho corpus/query. Model không tải lại, package không cài thêm. Đã COMPLETED toàn bộ luồng: 787 ảnh corpus → 772 crop/vector; 418 train + 112 dev / 298 ảnh query → 1.512 crop/vector; 409 train example đủ điều kiện có crop.

Chạy hai bước crop trước, rồi xem `annotated/`, `crops/`, thống kê `no_detection` và ảnh chạm `max_det` trong manifest. Bbox có padding 5% mỗi cạnh, crop PNG; ảnh không có detection giữ mapping rỗng. Prompt được đặt tại `corpus_image_cropping.py`; điều chỉnh theo ảnh train, không theo gold dev. Hai bước phải dùng cùng conf/imgsz/prompt.

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/corpus_image_cropping.py --conf 0.25 --imgsz 960
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/query_image_cropping.py --split dev --conf 0.25 --imgsz 960
```

Khi crop đã đạt yêu cầu, embedding từng crop bằng DINOv2 CLS L2 768 chiều, resize/pad 518×518 như full-image; deduplicate tính toán theo RGB pixels, giữ đủ crop instance. Đo corpus trước:

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/corpus_image_embedding.py --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/query_image_embedding.py --split dev --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/query_image_retrieval.py --split dev --method corpus
```

Mỗi crop query tính max cosine theo citation trên toàn corpus crop → top-10 riêng; gộp các danh sách bằng RRF k=60, trọng số mỗi crop `1/m`, khử citation trùng → top-10 query. Predictions có ranking riêng từng biển, candidate union và bằng chứng; metric chính F2@5, đủ 112 câu kể cả no-detection. Chưa mở rộng citation sang điều luật không chứa ảnh; prompt detector không phải nhãn/mã biển đã xác minh.

Example và hai bước fusion đã COMPLETED; dưới đây là lệnh tương ứng (thêm `--overwrite` khi tái lập output đã có):

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/query_image_retrieval.py --split dev --method examples
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/query_image_hybrid.py --split dev --rrf-k 60
.\.venv\Scripts\python.exe -u src/hybrid_rag/crop_image_method/hybrid_retrieval.py --split dev --text-weight 0.75 --rrf-k 60
```

Example so hai tập crop bằng trung bình hai chiều của max cosine, lấy top-5 train rồi mới đọc nhãn; bank không có crop được báo riêng. Corpus/example gộp 0.5/0.5 vào `query_retrieval_hybrid/`; final đọc text-only đã lưu, text/ảnh 0.75/0.25 cố định, xuất `image_text_retrieval/final_citations.json`. Query thiếu ảnh dùng text ranking và ghi fallback. Không detect được biển thì không tự thay bằng toàn ảnh; mọi gold giữ nguyên khi chấm.

Kết quả dev F2@5: corpus **0.091445**, example **0.430145**, image hybrid **0.378778**, final **0.425158**; xem giá trị đầy đủ trong từng `metrics.json`. Final gồm 112 câu đúng thứ tự `dev.json`, mỗi câu 5 citation. Coverage gold corpus giảm từ 132/254 trước detect xuống 65/254 sau detect; crop không tự giải quyết điều luật không chứa ảnh. Mặc định không overwrite. Khi chạy lại, thêm `--overwrite` để **xóa và tạo lại đúng thư mục output của bước được chọn**; thay prompt/input/code cần chạy lại các bước phụ thuộc, không hỗ trợ resume batch. Mốc full-image final được kiểm tra input/output và chấm lại predictions để so sánh; code hash tại lần chạy cũ được giữ trong `baseline_reference`, không sửa fingerprint hay dùng baseline làm inference input. Crop chỉ đọc text/baseline, không ghi đè chúng.

## YOLOE — preview hiện có

Code `src/hybrid_rag/yolo_e_train/yoloe_preview.py`; model ở `models/yolo_e_train/`, ảnh/crop/JSON ở `temp/sign_preview/yolo_e_train/` và `real_photos/`. Kết quả cũ có fingerprint khác code sau khi tách khỏi YOLO11n; kiểm tra output trước khi chạy lại.

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --device 0
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --real-photos --device 0
```

**Việc tiếp theo:** phân tích detection làm mất coverage và image hybrid kéo giảm điểm example theo [CHECKPOINTS.md](CHECKPOINTS.md). Toàn bộ pipeline crop đã COMPLETED; chưa đổi prompt, mapping hoặc trọng số để cải thiện thêm. Kết quả trên là dev, chưa phải test độc lập.
