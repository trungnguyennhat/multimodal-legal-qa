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

## Hybrid: chuẩn bị và fine-tune detector biển báo

Code: `src/hybrid_rag/yolo_train/detector_training.py`. Trạng thái: `READY_TO_RUN`; mới kiểm tra cú pháp và CLI, chưa tải đầy đủ dữ liệu/model hoặc chạy training/evaluation. Thiết kế tổng thể: [hybrid_plan.md](hybrid_plan.md). Bước này chỉ làm detector, chưa làm crop VLSP, image embedding hoặc fusion.

Dataset ngoài: [Traffic-sign-detection-VietNam](https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam), revision `bd948a572cefbea92a1c523a2784c1ea9cf1dbe1`; ZIP khoảng 905 MB, danh sách file có 8.125 ảnh train, 1.016 val và 1.016 test. Chưa kiểm tra nội dung ảnh/box đầy đủ. Đây là thiết lập dùng external detection data, khác điều kiện chỉ dùng dữ liệu VLSP.

Chạy từ thư mục gốc bằng PowerShell; dùng `.venv` có PyTorch/torchvision CUDA tương thích. Lệnh cài dưới đây thêm Ultralytics, không chủ động nâng cấp hoặc cài lại torch; nếu pip đề nghị thay torch/torchvision CUDA thì dừng và kiểm tra dependency trước.

```powershell
.\.venv\Scripts\python.exe -m pip install -r src/hybrid_rag/yolo_train/requirements.txt

# Tải dataset và pretrained YOLO11n; có thể dùng --target dataset hoặc model riêng.
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_train/detector_training.py download --target all

# Loại ảnh lỗi nhãn và bản trùng giữa các split (ưu tiên giữ test > val > train).
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_train/detector_training.py prepare --skip-invalid-images --deduplicate-splits

# Fine-tune: workers=0 cho Windows, không flip trái/phải, seed=2025.
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_train/detector_training.py train --name yolo11n_signs --epochs 100 --patience 20 --imgsz 640 --batch 8 --device 0

# Chọn ngưỡng trên val trước; tên mỗi lần evaluate phải khác nhau.
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_train/detector_training.py evaluate --weights models/yolo_train/yolo11n_signs/best.pt --split val --imgsz 640 --batch 8 --device 0

# Chỉ đánh giá test sau khi chốt cấu hình trên val.
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_train/detector_training.py evaluate --weights models/yolo_train/yolo11n_signs/best.pt --split test --imgsz 640 --batch 8 --device 0
```

`prepare` bỏ lớp 54/55 (đèn xanh/đỏ), gộp các lớp còn lại thành `traffic_sign`, giữ mọi box trong ảnh nhiều biển. Ảnh chỉ có đèn trở thành background cho bài toán biển. Biển cảnh báo có đèn giao thông phía trước vẫn được giữ. Thiếu nhãn/ảnh lỗi/box sai hoặc ảnh trùng byte giữa các split sẽ dừng; xem `artifacts/experiments/hybrid_rag/yolo_train/prepare/report.json` khi có báo cáo. Chưa dò ảnh gần trùng hoặc xác minh độ đầy đủ nhãn biển phụ. Không tự đổi split, sửa box hoặc tự coi file thiếu nhãn là background.

Lỗi dữ liệu đã xác minh bằng cách đọc toàn bộ 10.157 file nhãn trong ZIP: duy nhất `dataset/train/labels/8380.txt:4` có 6 trường (`12 0 0.63125 0.53984375 0.025 0.0455078125`) thay vì 5. Không tự đoán/xóa trường thừa. `prepare --skip-invalid-images` loại toàn bộ ảnh `8380` và mọi box của ảnh khỏi dữ liệu chuẩn bị, giữ nguyên ZIP nguồn; `report.json` và manifest lưu `invalid_images` (ảnh, nhãn, dòng gốc, lý do) cùng chính sách loại. Với lỗi hiện có, còn 8.124 ảnh train; val/test không đổi. Thống kê box chỉ tính các ảnh hợp lệ. Không có flag thì vẫn dừng sau khi ghi báo cáo, chưa tạo `prepared/`. Flag chỉ áp dụng lỗi định dạng/class/tọa độ nhãn; không bỏ qua ảnh hỏng, thiếu nhãn hoặc trùng giữa các split. Chưa chạy lại pipeline prepare sau bản sửa.

Dataset cũng có 28 bản ảnh trùng byte giữa các split. Thêm `--deduplicate-splits` để giữ bản ở split ưu tiên cao nhất theo `test > val > train`, loại toàn bộ ảnh/nhãn của bản ở split thấp hơn; không chuyển ảnh, gộp nhãn hay sửa ZIP nguồn. Nếu không có flag này thì vẫn dừng. Đọc và hash ảnh nguồn xác nhận: sau cả hai chính sách còn 8.097 train, 1.015 val, 1.016 test (loại 27 bản trùng train và 1 bản trùng val, ngoài ảnh train lỗi nhãn). Báo cáo/manifest giữ danh sách bản được giữ và bản bị loại trong `exact_cross_split_duplicates`, cờ `deduplicate_splits` và `duplicate_policy`; thống kê chỉ tính ảnh được giữ. Đây là split đã làm sạch, cần ghi rõ khi báo metric; chưa dò ảnh gần trùng. Chưa chạy lại lệnh prepare để tạo dữ liệu sau thay đổi này.

Đường dẫn:

Evaluate mặc định dùng tên split: `evaluate/val/` hoặc `evaluate/test/`, không cần `--name`. Có thể đặt `--name` riêng khi so nhiều lần chạy; thư mục đã tồn tại vẫn được bảo vệ khỏi ghi đè. Kết quả validation đã hoàn thành được đổi tên từ `yolo11n_signs_val_stopped/` thành `val/`; manifest giữ nguyên arguments/hash của lần chạy gốc để bảo toàn nguồn gốc kết quả.

- Dataset gốc tải về: `data/yolo_train/traffic-sign-detection-vietnam.zip`, `classid.csv`; dữ liệu một lớp ở `data/yolo_train/prepared/`, kèm `dataset.yaml` và manifest/hash từng ảnh/nhãn.
- Pretrained: `models/yolo_train/yolo11n.pt`. Sau train thành công, copy `best.pt` và `last.pt` sang `models/yolo_train/<name>/`. Ultralytics vẫn giữ checkpoint gốc trong output train.
- Output: `artifacts/experiments/hybrid_rag/yolo_train/`: `download/manifest.json`, `prepare/report.json`, `train/<name>/` và `evaluate/<name>/`. Train có `args.yaml`, `results.csv`, `weights/`, manifest; evaluate có metric precision/recall/mAP50/mAP50-95, tốc độ, prediction JSON/TXT và manifest. Manifest ghi hash input/code/checkpoint, phiên bản package, thời gian và peak CUDA memory. Settings Ultralytics cũng nằm trong output này.

Download tiếp tục file `.part` khi chạy lại, xác minh SHA256 ZIP theo LFS upstream; model ghi SHA256 local vì release upstream không công bố checksum. `prepare` không ghi đè thư mục đã có; nếu bị ngắt giữa bước ghi, kiểm tra/dọn thư mục `prepared/` thủ công trước khi chạy lại. Train/evaluate mới từ chối dùng lại thư mục output; dùng `--name` mới để giữ kết quả cũ.

Để dừng training rồi chạy tiếp với patience mới: nhấn Ctrl+C, chờ tiến trình cũ thoát và trả về prompt; không chạy hai tiến trình cùng `--name`. Giữ nguyên dataset và thư mục train, dùng:

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_train/detector_training.py train --resume --name yolo11n_signs --patience 8 --device 0
```

`--resume` đọc `artifacts/experiments/hybrid_rag/yolo_train/train/<name>/weights/last.pt`, khôi phục optimizer, EMA, lịch học và trạng thái early stopping theo Ultralytics 8.4.182. Chạy từ epoch tiếp theo sau epoch được lưu; batch của epoch đang dở không được khôi phục. Ví dụ checkpoint hết epoch 7 thì chạy lại epoch 8. Patience mới tính từ best epoch đã lưu, không đếm lại 8 epoch từ lúc resume; có thể dừng sớm ngay sau epoch tiếp theo nếu đã chờ quá lâu và không có cải thiện mới. Tổng epochs và cấu hình training khác giữ từ checkpoint; khi resume chỉ thay patience/device qua CLI này, không truyền `--weights` hoặc dùng `--epochs` để đổi tổng epoch.

Phải có ít nhất một checkpoint hoàn chỉnh. CLI từ chối dataset đã thay đổi, run đã COMPLETED hoặc checkpoint không còn optimizer/epoch (thường đã strip khi kết thúc). Resume tiếp tục ghi results.csv/checkpoint trong run cũ; manifest giữ các phiên trước trong `resume_history` và ghi `effective_training`, `start_epoch`, best epoch lúc bắt đầu. Thời gian trong results.csv của Ultralytics tính lại cho phiên resume, không coi là tổng thời gian toàn run. Resume chưa được chạy thực tế trong phiên sửa code; đã đối chiếu logic resume/early stopping với package đang cài.

Mỗi bước dài có heartbeat khoảng 30 giây: bước, số đã xử lý/tổng, phần trăm, elapsed và ETA của bước hiện tại; lúc khởi tạo chưa biết tổng thì ghi N/A. Training dùng FP32 (`amp=False`) để tránh bước AMP tự tải model kiểm tra; có thể tốn VRAM/thời gian hơn AMP. Thiếu VRAM: giảm `--batch 4` hoặc `2`; muốn thử biển nhỏ với `--imgsz 960` thì tạo lần chạy tên mới. Mặc định đánh giá `--conf 0.001 --iou 0.7` để tính AP; precision/recall Ultralytics báo ở điểm best-F1, không mặc định là chất lượng tại một ngưỡng deployment cụ thể. Không chọn ngưỡng bằng test.

Không commit dataset, trọng số, cache hoặc output sinh ra. `.gitignore` không được sửa; `models/` đã có rule ignore; `artifacts/experiments/` không được bảo vệ bởi rule `runs/`, cần kiểm tra file được stage trước khi commit. Kết quả detection trên dataset ngoài chưa xác nhận khả năng chuyển sang ảnh VLSP hoặc hình vẽ luật.

### Preview YOLOE trên cùng 11 ảnh VLSP train

Code ở `src/hybrid_rag/yolo_e_train/yoloe_preview.py`. Đây là thử detector pretrained với text prompt, không fine-tune. Chưa tải model hoặc chạy inference trong phiên tạo code. Dùng `yoloe-v8l-seg.pt` và `mobileclip_blt.ts` từ Ultralytics assets release v8.3.0; dependency Ultralytics giữ 8.4.182 và CLIP pin commit trong requirements riêng, không cần nâng cấp torch.

```powershell
.\.venv\Scripts\python.exe -m pip install -r src/hybrid_rag/yolo_e_train/requirements.txt
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py download
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --device 0
```

Download khoảng 707 MB thập phân (checkpoint ~107 MB, text encoder ~600 MB), lưu trong `models/yolo_e_train/`; hỗ trợ file `.part` và ghi SHA256 local. Không tự tải model khi preview. CLIP được cài bằng pip trong `.venv`, tokenizer đi kèm package; text encoder được nạp bằng đường dẫn local rõ ràng.

Preview đọc đúng đường dẫn ảnh trong `temp/sign_preview/image_001/predictions.json` đến `image_011/predictions.json`; không lấy mẫu mới. Output: `temp/sign_preview/yolo_e_train/image_001/` ... `image_011/`, mỗi thư mục có `original.jpg`, `annotated.jpg`, `sign_*.jpg`, `predictions.json`. Root output có manifest cấu hình/kết quả, download manifest và cache settings/matplotlib. Tất cả output của thử nghiệm này ở `temp/sign_preview/yolo_e_train/`, không đưa vào output training detector cũ.

Mặc định cùng `conf=0.25`, `imgsz=640` với preview YOLO11n. Prompts gồm biển tròn/tam giác, chữ nhật xanh dương/xanh lá/trắng có chữ, biển phụ, thời gian và chỉ dẫn cao tốc. Dùng class-agnostic NMS IoU=0.7 để giảm box lặp giữa các prompt; chỉ dùng bounding box để crop, chưa dùng mask. JSON giữ prompt khớp từng box. Không coi các prompt là mã biển pháp lý hoặc kết quả OCR.

Có thể truyền `--conf`/`--imgsz`; output khác cấu hình hoặc code sẽ bị chặn để không trộn thí nghiệm. Chạy lại cùng cấu hình cho phép tái dùng ảnh đã hoàn thành; ảnh chưa có predictions JSON sẽ chạy lại. Không tự xóa output cũ: muốn đổi cấu hình cần giữ bản cũ sang tên khác trong temp trước. Có heartbeat 30 giây. Chưa có box chuẩn cho 11 ảnh nên đây là so sánh trực quan, không phải metric detection recall; vẫn cần kiểm tra bỏ sót biển phụ, biển lớn và nhận nhầm.

Thử thêm 12 ảnh chụp query train mới (không dùng lại 11 ảnh trước):

```powershell
.\.venv\Scripts\python.exe -u src/hybrid_rag/yolo_e_train/yoloe_preview.py preview --real-photos --device 0
```

Danh sách ở `temp/sign_preview/yolo_e_train/real_photos/selection.json`: seed 2026, lấy từ 202 ảnh có caption summary chứa “ảnh chụp”, loại ảnh đã preview, kiểm tra trực quan và thay một ảnh ghép quảng cáo bằng lượt rút ngẫu nhiên tiếp theo. Bộ này có ảnh đường phố và biển cao tốc, không phải mẫu đại diện thống kê toàn bộ query. Output riêng `real_photos/image_001/` ... `image_012/`, cấu trúc JPG/crop/JSON như trước. Dùng lại hai model local trong `models/yolo_e_train/`, không cần download lại; không chạy inference trong phiên thêm code này.

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

Đã chạy normalization trên 402 article: 1.347 paragraph (1.135 bên ngoài bảng và 212 paragraph chung của bảng), 790 block figure ứng với 787 file ảnh, 212 bảng, 0 issue. Chunking đã chạy CLI: 402 parent. 830 chunk; 52 chunk ảnh/bảng vượt 1.024 token. dài nhất 7.076 token (giới hạn encoder 8.192); 27 issue ghi nhận caption thiếu JSON đã được chấp nhận. BGE-M3 dùng CLS pooling, chuẩn hóa L2, vector 1.024 chiều. Đã ghi đè index mới `(830, 1024)` bằng batch size 1, BF16, không truncate; embedding hoàn tất trong khoảng 34 giây. Corpus retrieval top 5 đã ghi đè kết quả dev 112 query, giữ nguyên query embedding và evaluator: precision 0.112500, recall 0.266901, F1 0.151245, **F2 0.199681**, hit rate 0.464286. F2 cũ 0.245940 giảm xuống 0.199681 (giảm 0.046259); đây là kết quả của toàn bộ thay đổi normalization/chunking, chưa tách riêng tác động từng thay đổi. Có 5 gold citation ngoài registry, vẫn tính vào recall như trước. Xem `query/query_retrieval_corpus/metrics.json`, `per_query_metrics.jsonl` và manifest để tái lập. Đã chạy lại retrieval example và gated hybrid dev với cùng query embedding: example F2 vẫn 0.4777248004425611; hybrid F2 từ 0.4918943282554603 xuống 0.48846026232139433 (112 query; high 97, middle 15, low 0). Example chỉ dùng corpus làm registry citation; chạy lại retrieval để cập nhật hash registry sau khi thay index, không tạo lại embedding train/query. Public gated hybrid đã chạy lại trên 100 query với index corpus mới: F2 vẫn 0.6042771408924155, hit rate 0.85 (high 88, middle 12, low 0). Đã tạo lại public query embedding để cập nhật hash manifest corpus; dùng caption hiện có, chấp nhận 3 caption INCOMPLETE như cấu hình cũ. Private đã tạo lại query embedding và chạy lại corpus/example/gated hybrid trên 146 query, ghi đè `artifacts/private_test/text_based_rag/retrieval/submission_task1.json`; không có private gold nên chưa chấm F2 mới. QA chưa chạy lại. Inference kiểm tra hash caption/normalization nên cần đồng bộ artifact trước khi chạy với corpus mới; không tự sửa manifest cũ để bỏ kiểm tra.

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

Bản cũ của cả public/private đọc corpus `data_preprocessing` và chỉ render chú thích gốc/tham chiếu hình, nên bỏ sót caption VLM dù retrieval đã dùng caption. Bản sửa yêu cầu corpus captioning `COMPLETED`, khớp hash corpus preprocessing và có caption cho từng hình được đưa vào context. Không cần chạy lại caption/embedding/retrieval nếu các artifact hiện có vẫn hợp lệ; chạy lại inference với `--overwrite`. Context dài hơn; chương trình kiểm tra đủ token trước khi inference và không tự cắt luật.

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
