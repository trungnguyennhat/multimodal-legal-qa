# Multimodal Legal RAG

Pipeline nghiên cứu VLSP 2025 MLQA-TSR: chuyển ảnh thành mô tả văn bản, retrieval điều luật và trả lời câu hỏi. Quy tắc làm việc: [AGENTS.md](AGENTS.md).

## Trạng thái và kết quả

| Thành phần | Trạng thái | Kết quả |
| --- | --- | --- |
| Dữ liệu và split | `COMPLETED` | 530 câu train gốc → 418 train + 112 dev; 100 public, 146 private |
| Corpus normalization schema 4 | `COMPLETED` | 402 article, 212 bảng với một paragraph chung mỗi bảng; 787 file ảnh, 0 issue |
| Corpus chunking 1.024/128, embedding và corpus retrieval dev | `COMPLETED` | 830 chunk; index `(830, 1024)`; dev 112 query, F2@5 = 0.19968126105839815 |
| Caption và embedding query train | `COMPLETED` | 418 query; cùng encoder với corpus |
| Retrieval dev/public và inference public | `AWAITING_USER_TEST` | Metrics trong bảng dưới |
| Private retrieval và inference: file đã nộp benchmark | `COMPLETED` | Hai submission, mỗi file 146 câu |
| Inference bổ sung caption hình trong luật và recovery | `AWAITING_USER_TEST` | Public đã có metrics: 99 câu COMPLETED, 1 câu phục hồi đáp án; private chưa xác nhận bản mới |
| Few-shot QA public/private | `AWAITING_USER_TEST` | Public đạt 0.76; private đã kiểm tra ghép context, chưa chạy model |

| Split / phương pháp | F2 retrieval | Accuracy QA | Nguồn kết quả |
| --- | ---: | ---: | --- |
| Dev — corpus | 0.245940149229051 | — | `query_retrieval_corpus/metrics.json` |
| Dev — example | 0.4777248004425611 | — | `query_retrieval_example/metrics.json` |
| Dev — gated hybrid | 0.4918943282554603 | — | `query_retrieval_hybrid/metrics.json` |
| Public — gated hybrid / gold-citation QA kèm caption corpus + few-shot | 0.6042771408924155 | 0.76 | Metrics local; 100 COMPLETED, không recovery/default label |
| **Private — submission đã nộp** | **0.5994742019171495** | **0.7054794520547946** | **Log benchmark do người dùng cung cấp** |

Subtask 2 dùng `relevant_articles` được cung cấp trong input QA, độc lập với citation dự đoán ở Subtask 1. Accuracy trên đây không phải accuracy của chuỗi retrieval → QA. Điểm private thuộc file đã nộp, không xác nhận chất lượng code mới. Public: trước khi bổ sung caption corpus 0.71; có caption corpus 0.73; thêm few-shot 0.76.

Lỗi nhãn Subtask 1 (tạm hoãn xử lý): train gốc có 25 lượt citation không khớp ID corpus (19 train + 6 dev), public có 3; gồm lỗi định dạng, mã mơ hồ và mục D.11 nằm trong D.10.
Retrieval hiện loại nhãn example ngoài registry nhưng giữ nguyên gold khi chấm F2; khi quay lại cần xác minh ánh xạ và báo riêng điểm benchmark/điểm hiệu chỉnh, kể cả khi mở rộng multimodal.

## Hybrid: baseline embedding toàn ảnh

Code và thiết kế: `src/hybrid_rag/full_image_method/` (mỗi bước một file), [CHECKPOINTS.md](CHECKPOINTS.md). Trạng thái `COMPLETED`: đã tải DINOv2, embedding corpus/query và chấm dev. Baseline dùng toàn ảnh, chưa dùng YOLOE, caption, câu hỏi hoặc fusion.

Chạy từ gốc project trong `.venv` hiện có, không cần cài thêm package:

```powershell
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="0"
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/image_model_download.py

$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/corpus_image_embedding.py --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_embedding.py --split dev --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/query_image_retrieval.py --split dev
```

DINOv2 ViT-B/14 đóng băng, CUDA FP32, CLS/L2 768 chiều; resize giữ tỉ lệ và padding 518 × 518, không center crop. Model được pin commit/hash ở lần download đầu. Corpus hiện có 787 ảnh; 112 query dev dùng 61 ảnh duy nhất. Query chung ảnh nhận cùng ranking.

Model và metadata tải xuống ở `models/huggingface/dinov2-base/`. Output thí nghiệm ở `artifacts/experiments/hybrid_rag/full_image_method/`: `corpus/`, `query_dev/`, `retrieval_dev/`. Xem manifest `COMPLETED`; `retrieval_dev/metrics.json` báo P/R/F1/F2 và hit rate tại 1/3/5/10, chỉ số chính `macro_f2_at_5`, coverage gold và ảnh query trùng corpus. Gộp citation bằng max cosine của mọi ảnh thuộc citation trước khi lấy top-k; gold ngoài registry vẫn tính khi chấm.

Kết quả dev 112 câu: macro F2@5 **0.05945680659531717**, hit rate@5 **0.160714**; gold image coverage **0.519685**. Đã chuyển model khỏi artifacts sang `models/huggingface/dinov2-base/`, giữ nguyên vector/prediction/metrics; manifest cập nhật đường dẫn/hash và lưu nguồn gốc trong `model_relocation`. Không cần tải hoặc embedding lại chỉ vì lần chuyển này.

Embedding/retrieval không ghi đè mặc định; thêm `--overwrite` cho đúng bước cần chạy lại. Nếu bị ngắt hoặc đổi fingerprint, chạy lại bước liên quan và các bước phụ thuộc; chưa resume batch. Thiếu VRAM: batch 2/1. Có log 30 giây. Không commit model/index/output; `.gitignore` giữ nguyên.

## Hybrid: nhánh text thuần

Code ở `src/hybrid_rag/text_only/`, tách thành `corpus_preprocessing.py`, `corpus_chunking.py`, `corpus_embedding.py`, `query_embedding.py`, `query_retrieval.py`. Nhánh này dùng lại `text_based_rag/corpus/data_preprocessing/structured_corpus.jsonl` làm nguồn, bỏ toàn bộ figure/marker ảnh (kể cả trong bảng), giữ văn bản luật/bảng, tiêu đề và citation gốc. Không đọc caption hoặc ảnh; không dùng example retrieval. Query embedding chỉ dùng nguyên văn `question`, không choices/answer/caption; `image_id` chỉ là metadata để ghép với nhánh ảnh sau này.

Đã chạy preprocessing và chunking thực tế: **402 article**, bỏ **790 figure block**, giữ text **212 bảng**, tạo **458 chunk** với ngân sách **1.024 token / overlap 128**, không chunk nào vượt 1.024 token. Bảng được xem là text và có thể chia qua nhiều chunk. Model BGE-M3 dùng lại bản local trong `models/huggingface/hub/` (revision `5617a9f61b028005a4858fdac845db406aefb181`); không sao chép/tải model vào artifacts. Embedding và retrieval dev đã `COMPLETED`: macro F2@5 **0.1681801898478224**, hit rate@5 **0.375**.

Chạy từ thư mục gốc bằng `.venv` hiện có:

```powershell
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"

# Hai bước này đã COMPLETED; chỉ chạy lại khi cần tái tạo corpus/chunk.
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/corpus_preprocessing.py --overwrite
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/corpus_chunking.py --max-tokens 1024 --overlap-tokens 128 --overwrite

# Các bước này cũng đã COMPLETED; thêm --overwrite nếu cần chạy lại.
# BGE-M3, CUDA BF16, CLS/L2 1.024 chiều, không truncate.
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/corpus_embedding.py --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_embedding.py --split dev --batch-size 4
.\.venv\Scripts\python.exe -u src/hybrid_rag/text_only/query_retrieval.py --split dev
```

Output chỉ ở `artifacts/experiments/hybrid_rag/text_only/`: `data_preprocessing/`, `corpus_chunking/`, `corpus_embedding/`, `query_embedding_dev/`, `query_retrieval_dev/`. Xem `data_preprocessing/corpus_preview.txt` và `corpus_chunking/embedding_text_preview.txt` để kiểm tra text thực sự được giữ/embedding. Các bước có manifest/hash input/code/model/output, log khoảng 30 giây, thời gian và peak CUDA memory khi embedding. Không ghi đè mặc định; thêm `--overwrite` cho đúng bước cần chạy lại, chưa resume batch. CUDA cần hỗ trợ BF16; thiếu VRAM giảm batch xuống 2/1, không tự chuyển CPU.

Retrieval lấy max cosine child chunk về citation, lưu top-10 với `query_id`, `source_image_id`, score và `best_chunk_id`. Chấm macro/micro P/R/F1/F2, hit rate tại 1/3/5/10, chỉ số chính `macro_f2_at_5`; giữ nguyên gold kể cả ngoài registry. `predictions.jsonl`, `per_query_metrics.jsonl`, `metrics.json` nằm trong `query_retrieval_dev/`. Hiện chỉ làm dev 112 câu để khớp nhánh ảnh.

## Hybrid: fusion text thuần + ảnh toàn khung

Code: `src/hybrid_rag/full_image_method/hybrid_retrieval.py`. Đọc hai `predictions.jsonl` đã lưu trong `text_only/query_retrieval_dev/` và `full_image_method/retrieval_dev/`; kiểm tra manifest/hash, tập query ID, image ID và citation registry. Dùng union top-10 mỗi nhánh; Weighted RRF tính `text_weight/(rrf_k+text_rank) + (1-text_weight)/(rrf_k+image_rank)`, nhánh vắng đóng góp 0. Không cộng cosine của hai encoder, không dùng gold để tạo ranking hoặc tự tìm trọng số. Tie-break theo citation; trọng số 0/1 cho phép kiểm tra từng nhánh độc lập.

```powershell
# Đã chạy cấu hình này; --overwrite chỉ thay kết quả fusion trong cùng thư mục.
.\.venv\Scripts\python.exe -u src/hybrid_rag/full_image_method/hybrid_retrieval.py --split dev --text-weight 0.5 --rrf-k 60 --final-top-n 5 --overwrite
```

Output: `artifacts/experiments/hybrid_rag/full_image_method/hybrid_retrieval_dev/`. **`final_citations.json`** có 112 câu theo thứ tự dev, mỗi câu 5 `relevant_articles`; `predictions.jsonl` giữ top-10 và rank/score/đóng góp cùng bằng chứng của từng nhánh. Có thêm `per_query_metrics.jsonl`, `metrics.json`, `manifest.json`; không chạy lại embedding hoặc sửa kết quả hai nhánh.

Kết quả cố định text/image 0.5/0.5, RRF k=60: macro F2@5 **0.14010798837660618**, thấp hơn text riêng **0.1681801898478224** (ảnh riêng **0.05945680659531717**). So với text: 7 câu tốt hơn, 15 câu kém hơn, 90 câu giữ nguyên. Union top-10 bổ sung 28 lượt gold từ ảnh ngoài danh sách text, micro candidate recall **0.421260**; điều này chưa bảo đảm fusion xếp hạng đúng. Đây là baseline fusion trên dev, chưa tối ưu trọng số hoặc kiểm chứng độc lập. Muốn đổi trọng số/final cutoff phải ghi rõ cấu hình và giữ kết quả cũ trước khi dùng `--overwrite`.

## Hybrid: chuyển sang YOLOE

Đã dọn code, dữ liệu, checkpoint fine-tune và kết quả thử nghiệm YOLO11n. Chỉ giữ model pretrained `models/yolo_train/yolo11n.pt`; không còn run YOLO11n để resume hoặc đánh giá. Hướng detector tiếp theo là YOLOE; baseline text + ảnh toàn khung đã có fusion riêng ở trên, chưa đưa crop YOLOE vào retrieval hoặc QA.

### Preview YOLOE trên cùng 11 ảnh VLSP train

Code ở `src/hybrid_rag/yolo_e_train/yoloe_preview.py`. Đây là thử detector pretrained với text prompt, không fine-tune. Đã có model local và kết quả preview; chưa fine-tune hoặc đánh giá bằng box chuẩn. Dùng `yoloe-v8l-seg.pt` và `mobileclip_blt.ts` từ Ultralytics assets release v8.3.0; dependency Ultralytics giữ 8.4.182 và CLIP pin commit trong requirements riêng, không cần nâng cấp torch.

```powershell
.\.venv\Scripts\python.exe -m pip install -r src/hybrid_rag/yolo_e_train/requirements.txt
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py download
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --device 0
```

Download khoảng 707 MB thập phân (checkpoint ~107 MB, text encoder ~600 MB), lưu trong `models/yolo_e_train/`; hỗ trợ file `.part` và ghi SHA256 local. Không tự tải model khi preview. CLIP được cài bằng pip trong `.venv`, tokenizer đi kèm package; text encoder được nạp bằng đường dẫn local rõ ràng.

Preview đọc đường dẫn 11 ảnh từ `temp/sign_preview/yolo_e_train/image_001/predictions.json` đến `image_011/predictions.json` hiện có; không lấy mẫu mới hoặc phụ thuộc output YOLO11n. Output: `temp/sign_preview/yolo_e_train/image_001/` ... `image_011/`, mỗi thư mục có `original.jpg`, `annotated.jpg`, `sign_*.jpg`, `predictions.json`. Root output có manifest cấu hình/kết quả, download manifest và cache settings/matplotlib. Tất cả output của thử nghiệm này ở `temp/sign_preview/yolo_e_train/`, tách biệt với pipeline text RAG.

Mặc định `conf=0.25`, `imgsz=640`. Prompts gồm biển tròn/tam giác, chữ nhật xanh dương/xanh lá/trắng có chữ, biển phụ, thời gian và chỉ dẫn cao tốc. Dùng class-agnostic NMS IoU=0.7 để giảm box lặp giữa các prompt; chỉ dùng bounding box để crop, chưa dùng mask. JSON giữ prompt khớp từng box. Không coi các prompt là mã biển pháp lý hoặc kết quả OCR.

Có thể truyền `--conf`/`--imgsz`; output khác cấu hình hoặc code sẽ bị chặn để không trộn thí nghiệm. Chạy lại cùng cấu hình cho phép tái dùng ảnh đã hoàn thành; ảnh chưa có predictions JSON sẽ chạy lại. Không tự xóa output cũ: muốn đổi cấu hình cần giữ bản cũ sang tên khác trong temp trước. Code đã tách khỏi YOLO11n nên fingerprint khác kết quả cũ; kết quả cũ được giữ nguyên, chưa chạy lại bằng code mới. Có heartbeat 30 giây. Chưa có box chuẩn cho 11 ảnh nên đây là so sánh trực quan, không phải metric detection recall; vẫn cần kiểm tra bỏ sót biển phụ, biển lớn và nhận nhầm.

Thử thêm 12 ảnh chụp query train mới (không dùng lại 11 ảnh trước):

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --real-photos --device 0
```

Danh sách ở `temp/sign_preview/yolo_e_train/real_photos/selection.json`: seed 2026, lấy từ 202 ảnh có caption summary chứa “ảnh chụp”, loại ảnh đã preview, kiểm tra trực quan và thay một ảnh ghép quảng cáo bằng lượt rút ngẫu nhiên tiếp theo. Bộ này có ảnh đường phố và biển cao tốc, không phải mẫu đại diện thống kê toàn bộ query. Output riêng `real_photos/image_001/` ... `image_012/`, cấu trúc JPG/crop/JSON như trước. Dùng lại hai model local trong `models/yolo_e_train/`, không cần download lại; đã có kết quả preview cho 12 ảnh.

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

Preprocessing schema 4 gộp text liên tiếp trong cùng article thành một `paragraph`, giữ xuống dòng; chỉ ngắt tại ảnh, bảng hoặc hết article. Giữ `type="table"`; mỗi bảng có đúng một `paragraph` trong `blocks`, gộp các ô theo hàng/cột. Table không có trường `text` trực tiếp, các ô không còn paragraph riêng; `cells`/`grid` chỉ giữ vị trí, header và ô gộp. Ảnh/bảng lồng trong ô vẫn giữ block và vị trí bằng marker `<<BLOCK:block_id>>` trong paragraph chung. Không còn clause, reference hoặc nhận diện nhãn/chú thích/đơn vị; chữ gốc vẫn được giữ.

Chunking duyệt theo thứ tự nguồn trong từng article, ghép text và nguyên ảnh/bảng nếu vừa **1.024 token**, tính cả tiêu đề và overlap. Không vừa thì đóng chunk sớm; riêng ảnh/bảng vượt ngân sách được giữ nguyên trong một chunk riêng. **Overlap tối đa 128 token cuối phần nội dung chunk trước** được đưa vào đầu chunk sau, kể cả khi gặp ảnh/bảng; không lặp tiêu đề trong overlap, không truyền qua article. Text dài ưu tiên ngắt ở ranh giới câu/đoạn, chỉ chia trong câu khi cần. Ảnh dùng toàn bộ caption JSON gốc, không tách item; ảnh trong bảng được ghép tại marker tương ứng, toàn bảng vẫn là một đơn vị nguyên khối. Chunk dài vượt giới hạn encoder sẽ báo lỗi, không tự cắt nội dung.

Dùng lại caption hiện có bằng `image_id` và SHA256 của file ảnh; manifest chunking lưu hash normalization mới cùng manifest/hash caption cũ để truy nguồn. Đây là tái dùng caption đã sinh với context cũ, không phải sinh lại theo normalization mới. 27 caption thiếu JSON đã được người dùng chấp nhận vẫn được giữ nguyên và ghi issue. Hàm duyệt block, render bảng cho inference và kiểm tra ngân sách embedding đã được cập nhật tương thích; chưa sinh lại caption hoặc chạy inference. Đã chạy lại embedding corpus và corpus retrieval dev với chunk mới.

```powershell
# Từ thư mục gốc, sau phần Chuẩn bị; ghi đè output tương ứng.
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_normalization
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_chunking --max-tokens 1024 --overlap-tokens 128 --overwrite

# Tạo lại index và chấm corpus retrieval dev; đã chạy và ghi đè output cũ.
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_embedding index --batch-size 1 --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split dev --method corpus --overwrite
```

Output vẫn ở `artifacts/experiments/text_based_rag/corpus/`. Normalization ghi đè sáu file cũ trong `data_preprocessing/`; xem `corpus_preview.md`, `corpus_inventory.json`, manifest `COMPLETED` và `schema_version=4`. Chunking ghi đè các file cũ trong `corpus_chunking/`; xem `embedding_text_preview.txt` để kiểm tra nội dung thực tế, `chunks.jsonl` để xem số token/nguồn/ảnh/previous-next, `issues.jsonl` và `manifest.json` để xem cấu hình, hash và số chunk vượt ngân sách. Không thêm đường dẫn output mới.

Đã chạy normalization trên 402 article: 1.347 paragraph (1.135 bên ngoài bảng và 212 paragraph chung của bảng), 790 block figure ứng với 787 file ảnh, 212 bảng, 0 issue. Chunking đã chạy CLI: 402 parent. 830 chunk; 52 chunk ảnh/bảng vượt 1.024 token. dài nhất 7.076 token (giới hạn encoder 8.192); 27 issue ghi nhận caption thiếu JSON đã được chấp nhận. BGE-M3 dùng CLS pooling, chuẩn hóa L2, vector 1.024 chiều. Đã ghi đè index mới `(830, 1024)` bằng batch size 1, BF16, không truncate; embedding hoàn tất trong khoảng 34 giây. Corpus retrieval top 5 đã ghi đè kết quả dev 112 query, giữ nguyên query embedding và evaluator: precision 0.112500, recall 0.266901, F1 0.151245, **F2 0.199681**, hit rate 0.464286. F2 cũ 0.245940 giảm xuống 0.199681 (giảm 0.046259); đây là kết quả của toàn bộ thay đổi normalization/chunking, chưa tách riêng tác động từng thay đổi. Có 5 gold citation ngoài registry, vẫn tính vào recall như trước. Xem `query/query_retrieval_corpus/metrics.json`, `per_query_metrics.jsonl` và manifest để tái lập. Đã chạy lại retrieval example và gated hybrid dev với cùng query embedding: example F2 vẫn 0.4777248004425611; hybrid F2 từ 0.4918943282554603 xuống 0.48846026232139433 (112 query; high 97, middle 15, low 0). Example chỉ dùng corpus làm registry citation; chạy lại retrieval để cập nhật hash registry sau khi thay index, không tạo lại embedding train/query. Public gated hybrid đã chạy lại trên 100 query với index corpus mới: F2 vẫn 0.6042771408924155, hit rate 0.85 (high 88, middle 12, low 0). Đã tạo lại public query embedding để cập nhật hash manifest corpus; dùng caption hiện có, chấp nhận 3 caption INCOMPLETE như cấu hình cũ. Private đã tạo lại query embedding và chạy lại corpus/example/gated hybrid trên 146 query, ghi đè `artifacts/private_test/text_based_rag/retrieval/submission_task1.json`; không có private gold nên chưa chấm F2 mới. QA chưa chạy lại. Inference đọc corpus preprocessing hiện tại và ghép lại caption theo image_id; không yêu cầu hash preprocessing trong manifest caption cũ trùng bản mới.

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

Dev/public có prediction và metrics local. Private hybrid đã chạy lại với index 830 chunk và cấu hình gated giữ nguyên; query embedding, corpus/example/hybrid và submission Task 1 đã ghi đè. Cả 146 query hoàn tất; high 130, middle 15, low 1. Chưa nộp/chấm benchmark cho file mới, điểm private ở đầu README thuộc submission cũ; Task 2 chưa chạy lại. File benchmark giữ input và thêm `relevant_articles`:

```text
artifacts/private_test/text_based_rag/retrieval/submission_task1.json
```

Kết quả từng nhánh dev và hybrid public/private nằm trong bảng đầu README. Public hybrid mới có precision 0.398778, recall 0.728929, F1 0.497886, F2 0.604277; giữ nguyên cấu hình gated đã chọn trên dev và kho 530 example. Kết quả bằng lần chạy cũ; chỉ retrieval public đã được cập nhật, accuracy QA public 0.76 vẫn thuộc lần inference cũ. Low gate chưa có mẫu dev/public trong các kết quả đã ghi nhận.

## 4. Subtask 2: inference từ citation được cung cấp

Input gồm toàn văn luật kèm caption VLM của các hình theo `relevant_articles`, mô tả query đã lọc relevance và câu hỏi/options. Cả public/private đọc `corpus/data_preprocessing/structured_corpus.jsonl` rồi ghép các file `corpus/corpus_image_captioning/captions/*_description.json` theo `image_id` (đường dẫn dưới `artifacts/experiments/text_based_rag/`). Caption được đưa vào ngay tại block hình, kể cả hình trong bảng. Giữ chú thích gốc và tách nhãn mô tả do model sinh khỏi văn bản luật. Qwen chạy text-only, temperature 0, seed 2025, thinking off; output gồm `visual_matching`, `legal_rule_extraction`, `option_evaluation`, `final_answer`.

Bản cũ của cả public/private đọc corpus `data_preprocessing` và chỉ render chú thích gốc/tham chiếu hình, nên bỏ sót caption VLM dù retrieval đã dùng caption. Inference yêu cầu corpus captioning `COMPLETED`, caption hợp lệ, không trùng image_id và đủ caption cho mọi hình trong corpus hiện tại. Mục D.11 lồng trong D.10 được nhận diện cả khi tiêu đề nằm giữa paragraph đã gộp, và chỉ lấy nội dung từ tiêu đề D.11. Sau khi đổi preprocessing, dùng lại caption theo image_id; không đối chiếu hash corpus/image manifest cũ của bước caption. Caption được tái dùng vẫn sinh với context cũ; nếu thay nội dung ảnh dưới cùng image_id thì cần sinh lại caption. Hash input hiện tại vẫn được lưu để tái lập và kiểm tra thay đổi trong lúc chạy. Không cần chạy lại caption/embedding/retrieval nếu các artifact hiện có vẫn hợp lệ; chạy lại inference với `--overwrite`. Context dài hơn; chương trình kiểm tra đủ token trước khi inference và không tự cắt luật.

Private đọc `submission_task2_no_labels.json`, tái dùng caption Task 1 sau khi đối chiếu ID/ảnh/câu hỏi. Submission có đúng `id`, `image_id`, `question`, `question_type`, `relevant_articles`, `answer`; giữ citation và thứ tự input. Choices chỉ dùng trong prompt.

```powershell
# Public: inference và accuracy local.
.\.venv\Scripts\python.exe -u -m text_based_rag.inference --split public_test --few-shot-count 3 --context-size 32768 --allow-incomplete-captions --overwrite

# Private: inference và xuất submission.
.\.venv\Scripts\python.exe -u -m text_based_rag.inference --split private_test --few-shot-count 2 --context-size 65536 --overwrite
```

Output public: `artifacts/public-test/text_based_rag/inference/`; private: `artifacts/private_test/text_based_rag/inference/`. Các file chính là `answers/`, `inference_results.jsonl`, `manifest.json`, log server; public thêm `metrics.json`, private xuất:

```text
artifacts/private_test/text_based_rag/inference/submission_task2.json
```

Để kiểm tra lần chạy mới, `manifest.json` có `config.input_mode=gold_law_with_corpus_captions_and_query_caption_text_only`; mỗi citation trong `inference_results.jsonl` có `corpus_caption_block_ids` ghi các hình đã đưa caption vào prompt (rỗng nếu citation không có hình).

Đã kiểm tra bước chuẩn bị context bằng `prepare()` thực tế, không chạy model: private 146 câu, 143 câu có caption luật, 1.142 lượt hình; public 100 câu, 94 câu có caption luật, 380 lượt hình. Private dùng context 65.536 để có thêm chỗ cho caption; chưa xác nhận số token tối đa hoặc VRAM, preflight khi chạy sẽ kiểm tra chính xác.

Logic hiện tại: JSON lỗi/bị cắt sẽ lấy `final_answer` hợp lệ trong output hoặc gọi model chỉ lấy đáp án. Nếu cả hai cách lỗi, dùng nhãn dự phòng A/Đúng và ghi `answer_source=default_label`. Status giữ lại để chạy lại câu lỗi, không chặn export khi đã xử lý hết input. Nhãn dự phòng không phải đáp án model suy luận; số lượng được báo trong metrics/manifest. Public few-shot hiện có accuracy 0.76, 100 COMPLETED, không recovery/default label; điểm private đã nộp được ghi riêng phía trên.

### Few-shot QA — public/private

Toàn bộ code few-shot nằm trong `src/text_based_rag/inference.py`. Tái sử dụng BGE-M3 embedding đã có của split đang chạy và 530 câu train gốc (418 train + 112 dev). Private lấy query embedding/caption từ nhánh `retrieval/`, đối chiếu với input Task 2 trước khi chọn ví dụ. Lọc cùng `question_type` và có ít nhất một citation chung, xếp cosine giảm dần rồi lấy tối đa 1, 2 hoặc 3 ví dụ khác ảnh và không trùng câu hỏi/choices. Cho phép ảnh trùng giữa query và example bank theo chính sách hiện tại. Mỗi ví dụ gồm câu hỏi, caption đã lọc relevance, choices, citation và đáp án chuẩn (NFC); không sinh lời giải hay thêm luật của ví dụ vào context câu hiện tại. Chỉ dùng đáp án train gốc cho ví dụ, không dùng đáp án test.

Public đang có lệnh thử 3-shot; đã kiểm tra chuẩn bị 100 câu: 93 câu có 3 ví dụ, 1 câu có 2, 1 câu có 1, 5 câu không có ví dụ (trước preflight token runtime), chưa chạy model 3-shot. Chạy các lệnh ở trên với `--few-shot-count 2`; `1` dùng tối đa một ví dụ, mặc định `0` không dùng ví dụ. Mọi cấu hình đều ghi vào thư mục `inference/` hiện có của split, ghi đè khi dùng `--overwrite`, không tạo thư mục kết quả riêng. Kết quả few-shot public 0.76 đã được chuyển nguyên vẹn vào `artifacts/public-test/text_based_rag/inference/` và thay kết quả 0.73. Manifest giữ thông tin/hash của lần chạy đã tạo kết quả; sau khi gộp code, cần `--overwrite` nếu chạy lại. Bỏ `--overwrite` để resume các lần chạy bằng code mới khi cấu hình/code/input giữ nguyên. Không cần chạy lại caption/embedding/retrieval nếu artifact còn khớp manifest.

Preflight đếm token bằng runtime thực tế, bỏ ví dụ cosine thấp nhất khi thiếu context; không có ví dụ phù hợp hoặc bỏ hết thì dùng zero-shot. Nếu riêng context câu hiện tại quá dài vẫn dừng, không cắt luật/caption. Prediction lưu `few_shot_examples` (ID, cosine, citation chung, đáp án, nội dung), `few_shot_dropped_ids`, `few_shot_fallback`; metrics báo số ví dụ thực dùng và số bỏ do context. Manifest lưu cấu hình và hash nguồn. Vẫn có heartbeat 30 giây khi chạy.

Public đã chạy với 2 ví dụ: 94 câu có 2, 1 câu có 1, 5 câu không có ví dụ phù hợp; không bỏ ví dụ do context. Accuracy 0.76 (Multiple choice 50/65, Yes/No 26/35). Private đã kiểm tra chuẩn bị context với 2 ví dụ: 139 câu có 2, 1 câu có 1, 6 câu không có ví dụ phù hợp (trước preflight token runtime); chưa chạy model hoặc đánh giá trên dev.

## Chạy lại và xử lý lỗi

- Caption/inference: bỏ `--overwrite` để resume cùng input/config/code; embedding bị ngắt cần chạy lại với `--overwrite`.
- Khi code/input/prompt/model đổi, fingerprint có thể yêu cầu `--overwrite`. Khi caption/embedding đổi, chạy lại các nhánh retrieval phụ thuộc trước hybrid.
- Thiếu VRAM: đóng model/server GPU khác; embedding dùng `--batch-size 1`. Inference kiểm tra token trước, không tự cắt luật; private cần context lớn hơn mặc định16.384.
- Preprocessing tự ghi đè sáu file output hiện có; nếu lỗi, xem manifest `FAILED` và chạy lại cùng lệnh. Chunking yêu cầu schema 4 và dùng lại caption khi khớp ID/hash ảnh; captioning yêu cầu schema 4, nhưng context/code mới có thể yêu cầu sinh lại khi chạy captioning. Không chạy lại captioning chỉ để tạo chunk với caption hiện có.

Đọc `manifest.json` và `metrics.json` của đúng bước để xem tiến độ/kết quả. Không commit hoặc phân phối dataset, model weights, index, cache và secrets. Tài liệu nguồn: [multimodal_legal_qa_vlsp2025_sources.md](multimodal_legal_qa_vlsp2025_sources.md).
