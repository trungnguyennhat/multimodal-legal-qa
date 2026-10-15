# Multimodal Legal RAG

Pipeline nghiên cứu VLSP 2025 MLQA-TSR: chuyển ảnh thành mô tả văn bản, retrieval điều luật và trả lời câu hỏi. Quy tắc làm việc: [AGENTS.md](AGENTS.md).

## Trạng thái và kết quả

| Thành phần | Trạng thái | Kết quả |
| --- | --- | --- |
| Dữ liệu và split | `COMPLETED` | 530 câu train gốc → 418 train + 112 dev; 100 public, 146 private |
| Corpus preprocessing, caption, chunking, embedding | `AWAITING_USER_TEST` | 402 parent, 787 ảnh, 5.459 child; index `(5459, 1024)` |
| Caption và embedding query train | `COMPLETED` | 418 query; cùng encoder với corpus |
| Retrieval dev/public và inference public | `AWAITING_USER_TEST` | Metrics trong bảng dưới |
| Private retrieval và inference: file đã nộp benchmark | `COMPLETED` | Hai submission, mỗi file 146 câu |
| Bản sửa recovery đáp án inference mới nhất | `AWAITING_USER_TEST` | Chưa chạy kiểm tra bản sửa này |

| Split / phương pháp | F2 retrieval | Accuracy QA | Nguồn kết quả |
| --- | ---: | ---: | --- |
| Dev — corpus | 0.245940149229051 | — | `query_retrieval_corpus/metrics.json` |
| Dev — example | 0.4777248004425611 | — | `query_retrieval_example/metrics.json` |
| Dev — gated hybrid | 0.4918943282554603 | — | `query_retrieval_hybrid/metrics.json` |
| Public — gated hybrid / gold-citation QA | 0.6042771408924155 | 0.71 | Metrics local |
| **Private — submission đã nộp** | **0.5994742019171495** | **0.7054794520547946** | **Log benchmark do người dùng cung cấp** |

Subtask 2 dùng `relevant_articles` được cung cấp trong input QA, độc lập với citation dự đoán ở Subtask 1. Accuracy trên đây không phải accuracy của chuỗi retrieval → QA. Điểm private thuộc file đã nộp, không xác nhận chất lượng bản sửa recovery chưa chạy.

## Chuẩn bị

Chạy PowerShell từ thư mục gốc project. Dùng `.venv`, GPU CUDA, model Qwen3.5-9B Q4_K_M và BGE-M3 đã tải local; llama.cpp ở `.venv/tools/llama.cpp-b11624/`. Model Qwen ở `models/huggingface/Qwen3.5-9B-GGUF/`, gồm GGUF và projector cho caption ảnh.

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
```

Dữ liệu gốc ở `data/raw/VLSP2025-MLQA-TSR/`, ảnh/split đã xử lý ở `data/processed/`. Khi cần chuẩn bị từ dữ liệu gốc:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data validate
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data prepare
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data split --seed 2025 --dev-ratio 0.2
```

## 1. Corpus: preprocessing → caption → chunk → embedding

Preprocessing giữ cấu trúc điều luật, đoạn văn, HTML, bảng và hình. Qwen nhận ảnh cùng nội dung điều luật đã xử lý, trả JSON gồm loại hình và các item mô tả. Caption được ghép vào parent corpus theo hình.

Chunking giữ toàn điều luật làm parent; child gồm đoạn luật, hàng bảng và từng item ảnh, gắn parent/citation để ánh xạ ngược khi retrieval. Ngân sách 512 token, overlap 64 khi chia unit dài. BGE-M3 chỉ embedding trường `text`; metadata giữ trong mapping. Vector 1.024 chiều, CLS pooling và chuẩn hóa L2.

```powershell
# Chỉ chạy preprocessing khi thư mục data_preprocessing chưa tồn tại.
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_normalization

# Tái tạo các bước sau, ghi đè output tương ứng.
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_image_captioning --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_chunking --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_embedding index --overwrite
```

Output: `artifacts/experiments/text_based_rag/corpus/{data_preprocessing,corpus_image_captioning,corpus_chunking,corpus_embedding}/`.

Kết quả hiện có: 402 parent; 5.459 child gồm 3.138 đoạn luật, 731 hàng bảng, 1.404 caption item và 186 caption fragment. Chunk dài nhất 503 token; embedding `(5459, 1024)`. Một số caption bị cắt đã được chấp nhận và giữ dưới dạng fragment, có thể còn cú pháp JSON trong text.

## 2. Query: caption → embedding

Caption nhận ảnh và câu hỏi. Query embedding dùng câu hỏi cùng các item `is_relevant=true`, bỏ summary/item không liên quan và loại item trùng. Nếu không còn item phù hợp thì dùng riêng câu hỏi. Dùng cùng BGE-M3, pooling và normalization với corpus.

```powershell
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split train --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split train --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split dev --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split dev --overwrite
```

Train/dev output: `artifacts/experiments/text_based_rag/query/query_image_captioning_{split}/` và `query_embedding_{split}/`. Số vector tương ứng 418 và 112.

Public và private dùng cùng logic:

```powershell
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split public_test --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split public_test --allow-incomplete-captions --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split private_test --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split private_test --overwrite
```

Public output: `artifacts/public-test/text_based_rag/`; private: `artifacts/private_test/text_based_rag/retrieval/`. Có 100 vector public và 146 vector private. Cờ `--allow-incomplete-captions` chỉ dùng khi chấp nhận lấy các item hoàn chỉnh từ caption bị cắt.

## 3. Subtask 1: corpus + example → gated hybrid

Corpus retrieval tính cosine query–child rồi gộp về citation bằng điểm child cao nhất. Example retrieval tính cosine query–example rồi lấy gold citation của example gần nhất. Dev dùng 418 train làm example bank; public/private dùng toàn bộ 530 câu train gốc (train + dev). Cho phép ảnh trùng giữa các tập theo cấu hình đã dùng.

Hybrid dùng độ tương đồng example cao nhất để chọn gate:

| Gate | Candidate | Trọng số example / corpus | Kết quả |
| --- | --- | --- | --- |
| High: cosine ≥ 0.82 | Citation của top 3 examples | 0.90 / 0.10 | Giữ toàn bộ union; corpus chỉ xếp hạng trong pool |
| Middle: 0.70 ≤ cosine < 0.82 | Union top 5 corpus và citation top 5 examples | 0.60 / 0.40 | Top 5 citation |
| Low: cosine < 0.70 | Như middle | 0.30 / 0.70 | Top 5 citation |

Điểm example của citation là max cosine của example hỗ trợ; điểm corpus là max child cosine. Chuẩn hóa cosine bằng `(cosine + 1) / 2` trước fusion. Cấu hình chọn trên dev, giữ nguyên khi chạy public/private.

Chạy lần lượt ba nhánh, chọn split cần xử lý:

```powershell
$split="private_test" # Hoặc "dev", "public_test"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split $split --method corpus --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split $split --method examples --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split $split --method hybrid --hybrid-strategy gated --example-weight 0.6 --final-top-n 5 --overwrite
```

Mỗi split có `query_retrieval_corpus/`, `query_retrieval_example/`, `query_retrieval_hybrid/`. Dev nằm trong `artifacts/experiments/text_based_rag/query/`, public trong `artifacts/public-test/text_based_rag/`, private trong `artifacts/private_test/text_based_rag/retrieval/`.

Dev/public có prediction và metrics local. Private hybrid xuất file benchmark, giữ input và thêm `relevant_articles`:

```text
artifacts/private_test/text_based_rag/retrieval/submission_task1.json
```

Kết quả từng nhánh dev và hybrid public/private nằm trong bảng đầu README. Low gate chưa có mẫu dev/public trong các kết quả đã ghi nhận.

## 4. Subtask 2: inference từ citation được cung cấp

Input gồm toàn văn luật preprocessing theo `relevant_articles`, mô tả query đã lọc relevance và câu hỏi/options. Qwen chạy text-only, temperature 0, seed 2025, thinking off; output gồm `visual_matching`, `legal_rule_extraction`, `option_evaluation`, `final_answer`.

Private đọc `submission_task2_no_labels.json`, tái dùng caption Task 1 sau khi đối chiếu ID/ảnh/câu hỏi. Submission có đúng `id`, `image_id`, `question`, `question_type`, `relevant_articles`, `answer`; giữ citation và thứ tự input. Choices chỉ dùng trong prompt.

```powershell
# Public: inference và accuracy local.
.\.venv\Scripts\python.exe -u -m text_based_rag.inference --split public_test --context-size 32768 --allow-incomplete-captions --overwrite

# Private: inference và xuất submission.
.\.venv\Scripts\python.exe -u -m text_based_rag.inference --split private_test --context-size 32768 --overwrite
```

Output public: `artifacts/public-test/text_based_rag/inference/`; private: `artifacts/private_test/text_based_rag/inference/`. Các file chính là `answers/`, `inference_results.jsonl`, `manifest.json`, log server; public thêm `metrics.json`, private xuất:

```text
artifacts/private_test/text_based_rag/inference/submission_task2.json
```

Logic hiện tại: JSON lỗi/bị cắt sẽ lấy `final_answer` hợp lệ trong output hoặc gọi model chỉ lấy đáp án. Nếu cả hai cách lỗi, dùng nhãn dự phòng A/Đúng và ghi `answer_source=default_label`. Status giữ lại để chạy lại câu lỗi, không chặn export khi đã xử lý hết input. Nhãn dự phòng không phải đáp án model suy luận; số lượng được báo trong metrics/manifest. Bản sửa recovery này chưa được chạy kiểm tra; điểm benchmark đã nộp được ghi riêng phía trên.

## Chạy lại và xử lý lỗi

- Caption/inference: bỏ `--overwrite` để resume cùng input/config/code; embedding bị ngắt cần chạy lại với `--overwrite`.
- Khi code/input/prompt/model đổi, fingerprint có thể yêu cầu `--overwrite`. Khi caption/embedding đổi, chạy lại các nhánh retrieval phụ thuộc trước hybrid.
- Thiếu VRAM: đóng model/server GPU khác; embedding dùng `--batch-size 1`. Inference kiểm tra token trước, không tự cắt luật; private cần context lớn hơn mặc định16.384.
- Preprocessing không hỗ trợ ghi đè thư mục đã tồn tại; tái sử dụng output hiện có nếu không đổi nguồn luật. Thiếu dữ liệu/model hoặc lỗi khởi động server vẫn có thể dừng pipeline trước inference.

Đọc `manifest.json` và `metrics.json` của đúng bước để xem tiến độ/kết quả. Không commit hoặc phân phối dataset, model weights, index, cache và secrets. Tài liệu nguồn: [multimodal_legal_qa_vlsp2025_sources.md](multimodal_legal_qa_vlsp2025_sources.md).
