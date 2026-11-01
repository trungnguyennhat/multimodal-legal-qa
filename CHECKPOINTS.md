# Baseline retrieval bằng toàn ảnh

## Mục tiêu và trạng thái

Thiết kế đã được duyệt và CLI đã được triển khai. **Các bước download, embedding corpus/query và retrieval dev đã COMPLETED.** Model được chuyển sang `models/huggingface/dinov2-base/`; SHA256 xác nhận vector/prediction/metrics giữ nguyên. Manifest ghi thông tin chuyển đường dẫn và nguồn gốc trong `model_relocation`.

Luồng: **ảnh query → DINOv2 đóng băng → tìm ảnh corpus → gộp citation → evaluator dev**. Baseline này làm mốc so sánh với YOLOE crop sau này; chưa dùng detector, caption, nội dung câu hỏi, example retrieval, fusion hoặc QA.

- Encoder: `facebook/dinov2-base` (DINOv2 ViT-B/14), Transformers `AutoModel`.
- Dữ liệu hiện tại: 787 ảnh corpus, 112 câu dev dùng 61 ảnh query khác nhau.
- Code: `src/hybrid_rag/full_image_method/` (mỗi bước một file).
- Model: `models/huggingface/dinov2-base/`.
- Output thí nghiệm: `artifacts/experiments/hybrid_rag/full_image_method/`.
- Dependency hiện có trong `.venv`: torch, transformers, huggingface-hub, Pillow, NumPy; không thêm package.

## CLI và output

Chạy từ thư mục gốc project bằng PowerShell:

```powershell
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="0"
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/image_model_download.py

$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/corpus_image_embedding.py --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_embedding.py --split dev --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_retrieval.py --split dev
```

| File chạy | Output |
| --- | --- |
| `image_model_download.py` | `models/huggingface/dinov2-base/`: config, processor config, safetensors và manifest revision/hash |
| `corpus_image_embedding.py` | `full_image_method/corpus/`: `images.jsonl`, `vectors.npy`, `manifest.json` |
| `query_image_embedding.py --split dev` | `full_image_method/query_dev/`: `images.jsonl`, `queries.jsonl`, `vectors.npy`, `manifest.json` |
| `query_image_retrieval.py --split dev` | `full_image_method/retrieval_dev/`: `predictions.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `manifest.json` |

Ba thư mục thí nghiệm trong bảng nằm dưới `artifacts/experiments/hybrid_rag/`. Hugging Face lưu metadata tải xuống trong `models/huggingface/dinov2-base/.cache/`.

## Các checkpoint

### ☑ CP1 — Kiểm kê dữ liệu và xác minh mapping

- Dùng `image_manifest.jsonl` và `structured_corpus.jsonl` của preprocessing schema 4, trạng thái `COMPLETED`.
- Mapping `occurrences.article_key` sang `document_id/id`; kiểm tra block thuộc article và giữ đủ liên kết ảnh–citation.
- Kiểm tra đường dẫn, ảnh đọc được và SHA256; không bỏ qua ảnh lỗi.
- Dev dùng ảnh gốc trong `data/processed/images/train/`, không dùng caption.
- Kiểm kê corpus nằm trong manifest `corpus/`; số query/ảnh duy nhất nằm trong manifest `query_dev/`. Coverage gold và các citation ngoài registry được tổng hợp ở CP5.

**Đạt khi:** corpus có mapping hợp lệ cho 787 ảnh; dev có đủ 112 query. Không tự suy hoặc sửa citation từ tên ảnh.

### ☑ CP2 — Chuẩn bị encoder và preprocessing

- `image_model_download.py` resolve `main` thành commit SHA 40 ký tự ở lần đầu, lưu trước khi tải các file; chạy lại dùng revision đã ghi.
- Xác minh kích thước, SHA256 LFS upstream của weight và lưu SHA256 mọi file model. Model hoàn tất được kiểm tra và tái dùng mà không đổi manifest.
- Embedding chỉ đọc model local đã xác minh; không tự tải khi chạy embedding.
- EXIF transpose → RGB → resize bicubic giữ tỉ lệ vừa khung 518 × 518 → padding giữa khung màu `(124,116,104)` → rescale 1/255 và normalize theo mean/std trong processor config.
- Không center crop, augmentation hoặc kéo méo ảnh.
- CUDA, FP32, seed 2025, TF32 tắt, `eval()`, `inference_mode()`, trọng số đóng băng. Batch mặc định 4; lấy CLS cuối rồi chuẩn hóa L2 thành vector FP32 768 chiều.

**Đạt khi:** manifest model `COMPLETED`, revision/hash hợp lệ; manifest embedding ghi đủ cấu hình preprocessing. Thiếu model hoặc CUDA thì báo lỗi rõ ràng.

### ☑ CP3 — Embedding corpus và query dev

- Một vector cho mỗi ảnh corpus; query deduplicate theo SHA256, mỗi ảnh chỉ embedding một lần.
- Sắp xếp image ID ổn định, lưu `vector_row`; `queries.jsonl` ánh xạ từng query về vector, không đưa gold/câu hỏi vào encoder.
- Kiểm tra shape, dtype FP32, giá trị hữu hạn, norm L2 và đọc lại vectors đã lưu trước khi hoàn tất.
- Kiểm tra input hash lại trước khi đánh dấu `COMPLETED`.
- Manifest lưu hash input/code/model (gồm file bước và code embedding dùng chung), config, package versions, GPU, thời gian và peak CUDA memory.
- Heartbeat khoảng 30 giây: bước hiện tại, đã xử lý/tổng, %, elapsed và ETA theo bước; chưa biết tổng thì N/A.

**Đạt khi:** với dữ liệu hiện tại, corpus `(787,768)`, query `(61,768)`, mapping đủ 112 câu; manifest các bước `COMPLETED`.

### ☑ CP4 — Retrieval ảnh → citation

- Kiểm tra index hoàn tất, hash input/output/code và cùng config encoder/preprocessing.
- Cosine chính xác bằng tích ma trận; chưa dùng FAISS.
- Điểm citation = max cosine trên tất cả ảnh corpus thuộc citation. Gộp citation trước khi lấy top-k.
- Citation không có hình không tham gia ranking; không gán điểm giả.
- Tie-break citation theo `law_id`, `article_id`; ảnh đồng điểm chọn image ID nhỏ nhất theo thứ tự từ điển.
- Lưu top-10 citation mỗi query, score, rank, ID/path ảnh corpus tạo điểm cao nhất.

**Đạt khi:** đủ 112 prediction, không trùng citation trong từng danh sách, truy được ảnh bằng chứng. Các query chung ảnh nhận cùng thứ hạng.

### ☑ CP5 — Chấm điểm và tổng kết baseline

- Tái dùng `scores()` của evaluator hiện tại; giữ nguyên tập gold, kể cả citation ngoài registry.
- Macro/micro precision, recall, F1, F2 và hit rate tại k = 1, 3, 5, 10. Chỉ số chính: **macro F2@5**.
- Báo coverage trên lượt gold citation của từng query, số query có gold chưa được index bao phủ; tách citation ngoài registry và citation trong registry nhưng không có ảnh.
- Báo số ảnh query trùng byte với corpus và danh sách query liên quan.
- Lưu metrics từng câu, gồm gold không có ảnh trong index, để phân tích lỗi và so sánh với YOLOE.
- Cập nhật kết quả thực tế vào phần liên quan của README sau khi người dùng chạy; không ghi điểm dự kiến như kết quả đã có.

**Đạt khi:** `retrieval_dev/manifest.json` là `COMPLETED`, có metrics tổng hợp/từng câu và phân biệt được coverage với chất lượng xếp hạng. Không yêu cầu baseline phải tăng điểm.

**Kết quả đã có:** macro F2@5 = **0.05945680659531717**, hit rate@5 = **0.160714**, gold image coverage = **0.519685**; 787 ảnh corpus, 228 citation có hình, 112 query dev / 61 ảnh duy nhất, 0 ảnh query trùng byte corpus.

## Chạy lại, kiểm tra và giới hạn

- Các bước embedding/retrieval từ chối output đã tồn tại; thêm `--overwrite` chỉ thay output của bước được chọn. Không tự xóa dữ liệu hoặc output ở nhánh khác.
- Embedding bị ngắt chưa resume từng batch; chạy lại bước đó với `--overwrite`. File tạm hoặc output cũ trong run `FAILED` không được dùng làm index hoàn tất.
- Khi đổi code/model/preprocessing hoặc ảnh/input, chạy lại embedding liên quan rồi retrieval. Sửa file query không làm invalid index corpus; sửa code embedding dùng chung yêu cầu tạo lại cả hai index. Chỉ đổi batch size không đổi định nghĩa encoder, nhưng vẫn cần ghi cấu hình lần chạy mới.
- Thiếu VRAM: đóng model khác hoặc thử batch 2/1; không tự giảm độ phân giải hoặc chuyển CPU.
- Kiểm tra bằng CLI thực tế, manifest và evaluator; không thêm self-check/demo/test riêng. Sau khi chuyển model, đã xác minh hash và loader đọc model/index local; không chạy lại embedding hoặc retrieval.
- Không sửa `.gitignore`, split, text pipeline hoặc kết quả YOLOE; không commit model, index hoặc output sinh ra.
- Thời gian/VRAM của lần chạy nằm trong manifest tương ứng. Ảnh giống nhau cho cùng ranking dù câu hỏi khác; resize toàn cảnh có thể làm mất chi tiết biển nhỏ. Dev là tập phát triển, không phải kiểm chứng độc lập.

## Nhánh text thuần để ghép với ảnh

Code: `src/hybrid_rag/text_only/`; kết quả: `artifacts/experiments/hybrid_rag/text_only/`. Nhánh caption-based cũ được giữ riêng. Model BGE-M3 dùng lại trong `models/huggingface/hub/`, không tải hoặc copy model mới.

| Checkpoint | File chạy | Trạng thái / điều kiện |
| --- | --- | --- |
| Lọc corpus | `corpus_preprocessing.py` | `COMPLETED`: 402 article; bỏ 790 figure block; giữ text 212 bảng, không còn marker/block ảnh |
| Chia chunk | `corpus_chunking.py` | `COMPLETED`: 458 chunk, tối đa 1.024 token, overlap 128; tái dùng thuật toán chia text hiện tại |
| Embedding corpus | `corpus_embedding.py` | `COMPLETED`: BGE-M3 local, CLS/L2 1.024 chiều, CUDA BF16, không truncate |
| Embedding query | `query_embedding.py --split dev` | `COMPLETED`: chỉ nguyên văn câu hỏi; image ID là metadata; không caption/choices/answer |
| Retrieval dev | `query_retrieval.py --split dev` | `COMPLETED`: macro F2@5 0.1681801898478224; hit rate@5 0.375; top-10, gold giữ nguyên |

Lệnh chạy tuần tự và chạy lại nằm trong phần **Hybrid: nhánh text thuần** của README. Preprocessing dùng nguồn đã chuẩn hóa của `text_based_rag`, không sửa nguồn đó; bảng được chuyển thành text có thể chia nhỏ, figure trong bảng bị bỏ. Manifest kiểm tra hash và không ghi đè output nếu thiếu `--overwrite`.

## Fusion text thuần + ảnh toàn khung — COMPLETED

- Code: `src/hybrid_rag/full_image_method/hybrid_retrieval.py`.
- Output: `artifacts/experiments/hybrid_rag/full_image_method/hybrid_retrieval_dev/`.
- Input: top-10 citation của hai nhánh đã COMPLETED; kiểm tra hash, query/image ID, citation registry và đối chiếu lại metric từng nhánh với gold dev. Không dùng nhánh caption-based cũ.
- Weighted RRF, text/image 0.5/0.5, k=60, lấy 5 citation cuối. Chưa tối ưu trọng số theo dev.
- `final_citations.json`: 112 câu, mỗi câu 5 citation, giữ thứ tự input; `predictions.jsonl`: top-10, trace rank/score và bằng chứng từng nhánh.
- `metrics.json`, `per_query_metrics.jsonl`, `manifest.json`: P/R/F1/F2 và hit rate@1/3/5/10, so sánh hai nhánh, gold ngoài registry giữ nguyên, hash và cấu hình tái lập.
- Macro F2@5: text **0.1681801898478224**, ảnh **0.05945680659531717**, hybrid **0.14010798837660618**. Hybrid thấp hơn text; 7 câu cải thiện, 15 câu kém hơn, 90 không đổi. Chưa kết luận fusion tốt hơn.
- Chạy lại theo README; `--overwrite` chỉ thay output fusion. Không tải model hoặc chạy lại embedding.
