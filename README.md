# Multimodal Legal RAG

Project nghiên cứu VLSP 2025 MLQA-TSR, không phải công cụ tư vấn pháp lý production. Quy tắc làm việc: [AGENTS.md](AGENTS.md).

## Trạng thái

| Thành phần | Trạng thái |
| --- | --- |
| Stage 0–1: khởi tạo, dữ liệu và split | `COMPLETED` |
| Stage 2–5 cũ: đã gỡ retrieval/adapter/QA | `AWAITING_USER_TEST` |
| Stage 6–7: ablation, demo và đóng gói | `NOT_STARTED` |
| Chuẩn hóa, chunk và index corpus | `AWAITING_USER_TEST` |
| Caption corpus và query train; embedding query train | `COMPLETED` |
| Caption, embedding dev và retrieval dev từ kho train | `AWAITING_USER_TEST` |

## Luồng hiện tại

```text
Câu hỏi dev + caption ảnh → BGE-M3 → tìm trong 418 mẫu train
→ lấy relevant_articles của 5 hàng xóm → gom điểm → tối đa 5 citation
```

Citation nhận cosine cao nhất trong các hàng xóm hỗ trợ nó, hòa điểm theo citation ID. Chỉ trả citation tồn tại trong index luật. Không fine-tune; nhãn dev chỉ dùng đánh giá sau khi dự đoán. Code từ chối query ID trùng train/dev và loại ví dụ cùng ID/pixel ảnh trước tìm kiếm; chưa phát hiện gần trùng do resize/crop/nén lại. Đã phát hiện 2 cặp ảnh train/dev trùng pixel dù khác ID.

Kết quả dev ngày 14/10/2026: 112 query, macro **F2 = 0,4634**, precision = 0,2896, recall = 0,5818, hit rate = 77,68%; candidate recall = 0,6389. Evaluator chính thức cho F2 = 0,4634467334, khớp metric local trong sai số epsilon. Có 5 query phải loại ví dụ cùng ảnh từ kho train; đã kiểm tra nguồn hàng xóm và loại ảnh cho toàn bộ prediction. Có 5 gold citation dev thiếu trong index, vẫn giữ trong mẫu số recall. Đã tạo đủ 61 description nhưng 7 bị cắt ở 384 token (ảnh hưởng 15 query); giữ `INCOMPLETE` và dùng qua cờ cho phép bên dưới. Manifest embedding/retrieval `COMPLETED`; chờ người dùng kiểm tra kết quả.

## Chạy trên dev

Chạy PowerShell từ root. Cần `.venv`, model local Qwen3-VL-8B NF4/BGE-M3, ảnh trong `data/processed/images/train`, split và vector train/corpus hiện có. Caption/embedding cần GPU CUDA; retrieval dùng CPU. Không cần tạo lại đầu vào đã hoàn tất.

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split dev
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split dev --allow-incomplete-captions
.\.venv\Scripts\python.exe -u -m text_based_rag.example_citation_retrieval --split dev --neighbors 5 --overwrite
```

Caption resume ảnh đã hoàn tất. Cờ `--allow-incomplete-captions` cho phép dùng description không rỗng bị chạm giới hạn token, giữ trạng thái `INCOMPLETE`, không đánh dấu người dùng đã duyệt; bỏ cờ để yêu cầu mọi caption hoàn tất. Embedding từ chối output đã tồn tại; muốn tạo lại phải tự lưu bản cũ trước. **Retrieval `--overwrite` ghi đè kết quả cũ**, không thay caption/vector train. Bỏ cờ để từ chối ghi đè.

| Output dưới `artifacts/experiments/text_based_rag/` | Nội dung |
| --- | --- |
| `query_image_captioning_dev/captions/*_description.json` | Description và trạng thái từng ảnh |
| `query_image_captioning_train/`, `query_image_captioning_dev/` | Mapping query/ảnh, caption tổng và manifest |
| `query_embedding_train/`, `query_embedding_dev/` | Vector, mapping, text preview và manifest |
| `query_retrieval/` | `retrieval_results.jsonl`, `retrieval_preview.txt`, `metrics.json`, `per_query_metrics.jsonl`, `manifest.json` |

Mong đợi: 112 query dev, 61 ảnh, vector shape `(112, 1024)`, 112 dòng prediction, kho có 418 train, nhưng query dev có ảnh trùng train được loại các mẫu cùng ảnh trước tìm kiếm. Manifest embedding/retrieval `COMPLETED`; caption có thể `INCOMPLETE` nếu có ảnh chạm token. Metric chính là macro F2; precision/recall/hit rate và candidate recall dùng phân tích lỗi. Tác vụ dài in phase, processed/total, %, elapsed và ETA khoảng mỗi 30 giây.

Chế độ cũ vẫn có qua `example_citation_retrieval --split train`: loại query đang xét và cùng ID/pixel ảnh khỏi kho. Retrieval mặc định là dev; caption/embedding mặc định vẫn train để tương thích.

## Dữ liệu và source

Split: 418 train / 112 dev (seed 2025, nhóm theo ID ảnh); 100 public test có nhãn; 146 private test không có gold. Không dùng public/private để chọn cấu hình; sample submission không phải ground truth.

- `src/multimodal_legal_rag/data.py`: loader, validator và split.
- `src/text_based_rag/`: chuẩn hóa, caption, chunk, embedding, retrieval.
- `example_citation_retrieval.py`: tìm ví dụ train; `query_retrieval.py`: baseline tìm trực tiếp trong corpus.
- `data/processed/`: ảnh/split; `models/`: model local; `artifacts/`: output chạy.

Đã đổi thư mục output sang `artifacts/experiments/`; caption/vector train nằm trong `query_image_captioning_train/` và `query_embedding_train/`. Chỉ đổi vị trí, không chạy lại hay sửa nội dung artifact; đường dẫn trong manifest cũ là thông tin lịch sử. Đầu vào train/corpus đã có. Khi cần tái tạo, đặt biến môi trường như trên và chạy lần lượt qua `.\.venv\Scripts\python.exe -m`: `text_based_rag.corpus_normalization`, `text_based_rag.image_captioning`, `text_based_rag.corpus_chunking`, `text_based_rag.corpus_embedding index`, `text_based_rag.query_image_captioning --split train`, `text_based_rag.query_embedding --split train`. Các bước này không mặc định ghi đè output cũ; kiểm tra manifest và preview trước bước tiếp theo.

## Lỗi và giới hạn

- Import lỗi: dùng Python `.venv`, đặt `PYTHONPATH=src` và chạy từ root.
- Hash/mapping không khớp hoặc query ID train-dev trùng: kiểm tra nguồn/split; không bỏ validation.
- Citation nhãn thiếu trong index: báo và loại khỏi nguồn prediction, vẫn tính trong gold recall; không tự sửa nhãn.
- Thiếu model: tải BGE-M3 bằng `corpus_embedding download`, Qwen bằng `temp/download_vlm.py`, qua Python `.venv` trước inference.
- OOM: đóng ứng dụng GPU; embedding có thể giảm `--batch-size 1`.
- Caption INCOMPLETE có thể làm giảm chất lượng. Không suy diễn kết quả dev từ điểm train; chọn cấu hình nhiều lần trên dev cũng có thể quá khớp.

Không commit/phân phối lại dataset, weights, index, cache hoặc secrets; tôn trọng data agreement và giấy phép. Git chỉ rollback được file đã theo dõi. Nguồn: [tài liệu tham khảo](multimodal_legal_qa_vlsp2025_sources.md). Chờ người dùng kiểm tra trước khi chuyển trạng thái `COMPLETED`.
