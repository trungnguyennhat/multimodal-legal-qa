# Multimodal Legal RAG

Hệ thống nghiên cứu cho VLSP 2025 MLQA-TSR, không phải công cụ tư vấn pháp lý production.

## Tiến độ

| Stage | Nội dung | Trạng thái |
| --- | --- | --- |
| 0 | Khởi tạo và quản trị | `COMPLETED` |
| 1 | Dữ liệu | `COMPLETED` |
| 2 | Evaluator còn giữ; baseline retrieval đã xóa | `AWAITING_USER_TEST` |
| 3 | Visual retrieval và adapter đã xóa | `AWAITING_USER_TEST` |
| 4 | Hybrid retrieval đã xóa | `AWAITING_USER_TEST` |
| 5 | QA và kết quả đã xóa | `AWAITING_USER_TEST` |
| 6 | Thực nghiệm luận văn | `NOT_STARTED` |
| 7 | Demo và đóng gói | `NOT_STARTED` |
| Corpus text_based_rag | Chuẩn hóa văn bản, hình và bảng; chưa chunk/index | `AWAITING_USER_TEST` |
| Caption corpus text_based_rag | Mô tả ảnh luật bằng Qwen3-VL NF4, không crop | `AWAITING_USER_TEST` |

## Chuẩn hóa corpus text_based_rag — `AWAITING_USER_TEST`

Đã triển khai `src/text_based_rag/corpus_normalization.py`, dùng Python standard library và loader JSON đã có. Không cài dependency, không dùng GPU. Theo xác nhận của người dùng, hướng retrieval mới dùng Qwen3-VL-8B NF4 với prompt trong file hiện tại, không crop; bước này chỉ chuẩn hóa corpus, chưa chạy VLM, tạo caption, chia chunk hoặc embedding.

Schema phiên bản 2: mỗi record dẫn xuất giữ `id`, `title`, ID/tiêu đề văn bản, `article_key` theo vị trí nguồn và `blocks` theo thứ tự; không lưu lại `text` gốc hoặc thuộc tính HTML/CSS (`attributes`). Nội dung chuẩn hóa nằm trong các block văn bản và ô bảng. Block có ID duy nhất và `source_span` là khoảng ký tự `[start, end)` của vùng nguồn chứa block trong `text` ở law_db (vùng HTML có thể chứa nhiều block, không phải offset chính xác của từng từ). Đường dẫn/hash nguồn nằm trong manifest để đối chiếu mà không nhân bản nội dung gốc. HTML được đọc bằng HTMLParser; bỏ markup trang trí, giải mã entity, giữ ngắt đoạn, biểu diễn sub/sup bằng `_(...)`/`^(...)`. Văn bản ngoài HTML giữ ranh giới xuống dòng. File gốc không được ghi lại.

Hình lấy từ marker và thẻ img, kể cả trong ô bảng; giải mã URL, đối chiếu basename với thư mục ảnh local, không tải ảnh từ mạng. Manifest ảnh lưu SHA256 và mọi vị trí xuất hiện; file có cùng hash dùng cùng image_id. `description_vlm` hiện là null. Bảng giữ từng ô, tọa độ bắt đầu, rowspan/colspan, nội dung block trong ô và grid tham chiếu cell_id; null trong grid là ô thiếu, không phải số 0. `th` được đánh dấu header; không tự đoán header từ ô `td`. Bảng/chú thích/đơn vị không bị chuyển thành đoạn số liệu phẳng. `caption_original`/`unit_original` là chữ chú thích/đơn vị đã chuẩn hóa lấy từ nguồn, không chứa bản HTML gốc.

Nhãn hình được gắn khi đoạn ngay sau bắt đầu bằng “Hình ...”; tên bảng/đơn vị được tìm trong tối đa ba đoạn liên tiếp trước bảng. Đoạn chú thích vẫn được giữ trong blocks. Tham chiếu đơn giản “Hình/Bảng + số hoặc A.1” được giải quyết trong mục hiện tại trước, sau đó trong cùng văn bản nếu chỉ có một đích. Không chọn tùy tiện nếu có nhiều đích. Nhãn không tìm được, tham chiếu mơ hồ/không tìm được, HTML cần sửa cấu trúc và bảng bất thường được ghi vào issues. Chưa suy diễn tham chiếu khoảng, nhóm hình con, hoặc các cách viết phức tạp; kiểm tra bản preview cùng text gốc khi gặp các trường hợp này. Bộ kiểm tra tính toàn vẹn ID và đích liên kết chạy ngay trong command chuẩn hóa.

Điều kiện: file luật và ảnh đã được giải nén ở đường dẫn Stage 1, `.venv` có sẵn. Chạy từ root project:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -m text_based_rag.corpus_normalization
```

Output cố định tại `artifacts/results/text_based_rag/data_preprocessing/`. Thư mục normalization hiện có đã được đổi tên theo yêu cầu; không cần chạy lại. Command chuẩn hóa từ chối chạy nếu thư mục đích đã tồn tại để tránh ghi đè:

- `structured_corpus.jsonl`: một record dẫn xuất cho mỗi mục gốc, chỉ có nội dung chuẩn hóa trong blocks cùng ID và thông tin truy vết; không có text HTML gốc hoặc attributes trang trí.
- `image_manifest.jsonl`: danh sách ảnh local được tham chiếu, hash và vị trí sử dụng.
- `corpus_inventory.json`: số văn bản/mục, các loại block, số lần sử dụng ảnh và thống kê issue.
- `corpus_preview.md`: nội dung đọc được, ảnh, bảng theo từng ô và liên kết để đối chiếu.
- `issues.jsonl`: các trường hợp cần người dùng kiểm tra; không tự sửa nội dung luật.
- `manifest.json`: hash nguồn, đường dẫn, phiên bản schema, số lượng, thời gian và trạng thái chạy.

Mong đợi trên dữ liệu hiện tại: 2 văn bản, 402 record trong structured_corpus; ảnh lấy cả từ 761 marker và các thẻ img nên không cố định số ảnh manifest bằng 761. Bảng có thể lồng nhau nên số block table không nhất thiết bằng số marker TABLE. Terminal in tiến trình khoảng mỗi 30 giây và đường dẫn output cuối cùng. `manifest.status=COMPLETED` nghĩa là command hoàn tất, không có nghĩa các issue đã được duyệt; trạng thái bàn giao vẫn AWAITING_USER_TEST.

Mở `corpus_preview.md` để đối chiếu mục 12, B.24 và một bảng có ảnh; kiểm tra thứ tự, chữ/số/đơn vị, ảnh thuộc đúng ô, cell_id ở grid và references. Xem thống kê issue trong corpus_inventory rồi các chi tiết trong issues.jsonl. Nếu thiếu ảnh, kiểm tra thư mục `data/processed/images/law`; có thể truyền `--source` và `--image-dir` khi dữ liệu nằm ở vị trí khác. Lỗi import: đặt PYTHONPATH=src và dùng đúng Python .venv. Nếu chạy bị lỗi, thư mục giữ manifest FAILED; không dùng làm corpus hoàn chỉnh, sửa nguyên nhân, tự đổi tên thư mục data_preprocessing đang lỗi để lưu lại rồi chạy lại. HTMLParser chỉ sửa một số thẻ đóng tùy chọn; trường hợp HTML bất thường phải xem lại text gốc, không coi bản chuẩn hóa là tự động chính xác tuyệt đối.

Agent không chạy lại pipeline chuẩn hóa; chỉ đổi tên thư mục output hiện có theo yêu cầu. Bước caption đã được triển khai theo lệnh tiếp tục của người dùng.

Corpus trong data_preprocessing hiện có schema 2. Nếu cần chuẩn hóa lại, tự đổi tên thư mục đích hiện có để lưu bản cũ trước khi chạy cùng command; code không tự ghi đè hoặc xóa result. Mong đợi record không còn trường text cấp mục, block/table/cell không còn attributes; text trong block paragraph vẫn là nội dung luật đã chuẩn hóa. Trạng thái giữ AWAITING_USER_TEST.

## Caption ảnh corpus — `AWAITING_USER_TEST`

Code: `src/text_based_rag/image_captioning.py`. Đọc corpus schema 2 và image_manifest từ `artifacts/results/text_based_rag/data_preprocessing/`; kiểm tra hash ảnh, mô tả mỗi hash đúng một lần, rồi gắn `description_vlm` vào mọi block hình tương ứng, kể cả hình trong bảng lồng nhau. Nội dung luật và caption gốc được giữ riêng với mô tả do model sinh. Chưa chia chunk hoặc embedding.

Điều kiện: `.venv` và checkpoint Qwen3-VL NF4 local đã có từ thử nghiệm; GPU CUDA hỗ trợ BF16, ít nhất 11 GiB VRAM trống. Dùng `LAW_PROMPT` hiện tại trong module image_captioning.py do người dùng chỉnh. Sửa prompt pipeline tại module mới; sửa temp không tự thay prompt pipeline. Không crop, max_pixels=786432, max_new_tokens=1024, greedy, SDPA. Giới hạn allocator tối đa 12 GiB và chừa tối thiểu 2 GiB theo VRAM trống lúc bắt đầu; đây không phải bảo đảm chống OOM khi ứng dụng khác dùng GPU.

Chạy từ root project, chỉ một phiên caption cùng lúc:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -m text_based_rag.image_captioning
```

Output nằm trực tiếp trong `artifacts/results/text_based_rag/image_captioning/`, không có thư mục fingerprint. Chạy lại cùng lệnh sẽ tiếp tục các ảnh chưa hoàn tất. Bạn có thể sửa prompt hoặc giới hạn token rồi chạy tiếp: caption COMPLETED được giữ nguyên; ảnh chưa xong dùng prompt và giới hạn token hiện tại. Terminal báo số ảnh giữ lại và số còn cần xử lý. Chỉ dừng nếu corpus hoặc danh sách ảnh thay đổi. Không sửa file trong data_preprocessing.

- `captions/<tên_ảnh>_description.json`: checkpoint từng ảnh, ví dụ `image008_description.json`; chứa mô tả, trạng thái, token, thời gian và VRAM. Nếu hai ảnh khác đuôi có cùng tên, giữ thêm đuôi để phân biệt (ví dụ `image008.png_description.json`). Hash chỉ dùng bên trong để kiểm tra dữ liệu và cấu hình. Ghi qua file tạm rồi thay thế để tránh checkpoint bị ghi dở. Tại thời điểm cập nhật, thư mục image_captioning không còn tồn tại nên không có checkpoint cũ để đổi tên.
- `image_captions.jsonl`: tổng hợp caption sau khi toàn bộ ảnh hoàn tất.
- `structured_corpus.jsonl`: corpus mới có description_vlm tại các block hình; giữ nguyên cấu trúc và liên kết.
- `corpus_preview.md`: bản đọc có cả chú thích nguồn và mô tả VLM để đối chiếu ảnh.
- `manifest.json`: cấu hình/prompt đầy đủ, tài nguyên và trạng thái chạy. Chỉ dùng corpus khi status là COMPLETED.

Mong đợi với corpus hiện tại: 787 caption theo hash và 402 record corpus. Terminal in mô tả từng ảnh, tiến độ/ETA mỗi ảnh và khoảng mỗi 30 giây. Lần sau nếu caption đủ thì không cần nạp model để xuất lại corpus. Kiểm tra preview cùng ảnh gốc để đánh giá độ đúng; COMPLETED chỉ xác nhận xử lý đủ, không xác nhận mô tả chính xác.

Thiếu ảnh hoặc hash thay đổi: sửa dữ liệu và chuẩn hóa lại, không bỏ qua hình. Thiếu model local: kiểm tra cache checkpoint ghi bên dưới. Thiếu VRAM/OOM: đóng ứng dụng GPU rồi chạy lại cùng command; caption thành công đã lưu sẽ được bỏ qua. Caption rỗng/chạm giới hạn token: chương trình lưu checkpoint INCOMPLETE, báo cảnh báo rồi tiếp tục ảnh tiếp theo, không dừng terminal vì trường hợp này. Cuối lượt, nếu còn ảnh chưa xong, manifest có status INCOMPLETE và incomplete_images liệt kê tên ảnh; chưa xuất corpus hoàn chỉnh. Chạy lại cùng command sẽ thử lại các ảnh này, bỏ qua caption COMPLETED. Nếu vẫn chạm giới hạn, chỉnh prompt hoặc MAX_NEW_TOKENS trước khi chạy lại. Lỗi CUDA/OOM hoặc lỗi đọc/ghi file vẫn dừng và báo lỗi. Nếu đổi prompt hoặc MAX_NEW_TOKENS, chạy lại cùng command để tiếp tục, không cần chuyển thư mục. Ảnh INCOMPLETE được thử lại; caption COMPLETED không tự được mô tả lại. Các ảnh có thể dùng prompt khác nhau; manifest lưu lịch sử trong caption_settings và mỗi checkpoint giữ mã đối chiếu với lịch sử đó. Muốn làm lại một ảnh đã xong, tự xóa checkpoint tương ứng trước khi chạy. Lỗi import: dùng đúng Python .venv và PYTHONPATH=src.

Agent chưa chạy inference hoặc sinh output caption. Chờ người dùng tự chạy và xác nhận; trạng thái bàn giao AWAITING_USER_TEST.

## Thử mô tả ảnh bằng Qwen3-VL — `AWAITING_USER_TEST`

Theo yêu cầu ngày 10/10/2026, đã xóa hai cache Qwen2.5-VL-3B-Instruct và Qwen2.5-VL-7B-Instruct (gồm weight tải dở và lock tương ứng). Giữ model embedding/Visualized-BGE, dữ liệu và các output mô tả ảnh cũ. Script hiện chỉ dùng `unsloth/Qwen3-VL-8B-Instruct-bnb-4bit`, bản lượng tử hóa sẵn từ Qwen3-VL-8B-Instruct; không cần cài package Unsloth hoặc chạy remote code.

Agent đã tải hoàn tất checkpoint vào `models/huggingface/models--unsloth--Qwen3-VL-8B-Instruct-bnb-4bit/snapshots/10d842936325905934ea24e64325d774693f166e`, cả hai safetensors khớp kích thước và SHA256 LFS. Chưa chạy inference hoặc sinh caption Qwen3-VL; người dùng có thể chạy trực tiếp lệnh inference bên dưới, không cần tải lại.

### Chọn cấu hình và VRAM

RTX 5060 Ti có 16311 MiB VRAM (15,93 GiB); lúc kiểm kê còn 14416 MiB (14,08 GiB) trống. Index checkpoint ghi 8.767.123.696 tham số: tính BF16 2 byte/tham số là khoảng 17,53 GB (16,33 GiB) chỉ cho trọng số, chưa tính activations xử lý ảnh, KV cache và CUDA; không phù hợp yêu cầu chừa headroom khi chạy toàn model trên GPU này. Chọn NF4 4-bit double quant cho phần ngôn ngữ, giữ vision/embedding/lm_head BF16 theo config checkpoint. Đây là bản đóng gói bởi Unsloth, không phải checkpoint lượng tử hóa do Qwen phát hành.

Ước lượng lập kế hoạch cho inference một ảnh ngắn: khoảng 8–11 GiB, chưa đo trên máy này. Mức dùng thực tế phụ thuộc kích thước ảnh, số token và phiên bản thư viện. Script xử lý tuần tự một ảnh, SDPA, greedy, giới hạn ảnh tối đa 768 × 32 × 32 pixel, input tối đa 2048 token và output 384 token. Qwen3-VL dùng patch/merge theo bội 32; không giữ cấu hình bội 28 của Qwen2.5-VL. Hai prompt người dùng đã chỉnh và tọa độ crop được giữ nguyên. Thay preprocessing/model làm các số đo cũ không phải phép so sánh chỉ thay model.

Script lấy VRAM trống lúc bắt đầu, đặt ngân sách allocator PyTorch bằng `min(12 GiB, VRAM trống - 2 GiB)` và dừng nếu ngân sách dưới 9 GiB. Giới hạn này có thể làm PyTorch báo OOM trước khi GPU đầy; không tự offload sang CPU, không tự giảm ảnh hoặc đổi model. Đây không phải giới hạn toàn bộ VRAM: CUDA và ứng dụng khác có thể dùng thêm bộ nhớ. Cần theo dõi `nvidia-smi` khi chạy lần đầu. JSON lưu peak allocated, peak reserved và VRAM trống sau inference để đánh giá headroom thực tế; không bảo đảm hết OOM cho cấu hình ảnh/prompt tùy ý.

Nguồn: [model gốc Qwen](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [config NF4 của checkpoint được chọn](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-bnb-4bit/blob/main/config.json), [Qwen3-VL trong Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl).

### File và điều kiện

- `temp/describe_images.py`: inference local với Qwen3VLForConditionalGeneration, crop tùy chọn, output JSON/PNG theo timestamp; không tải model trong lúc inference.
- `temp/download_vlm.py`: chỉ tải checkpoint vào `models/huggingface`, không nạp model/GPU hoặc sinh caption; chạy lại để tiếp tục tải dở. Cố định revision `10d842936325905934ea24e64325d774693f166e` trong cả hai script. Metadata tải qua Hugging Face Hub; hai safetensors tải trực tiếp từ resolve URL của revision, tổng 7.226.399.236 byte (6,73 GiB), kiểm tra kích thước và SHA256 LFS trước đổi tên file `.incomplete`. Cách này tránh đường tải weight qua HEAD redirect bị đứng trên máy. Log mỗi 30 giây có số file, byte, %, elapsed và ETA ước lượng.
- `.venv`: Transformers 4.57.1, tokenizers 0.22.2, bitsandbytes 0.49.0, Accelerate 1.4.0, PyTorch 2.7.1+cu128 hiện có. Chỉ nâng Transformers/tokenizers để hỗ trợ Qwen3-VL; không nâng PyTorch. hf_xet 1.7.0 đã có từ lần thử trước, mặc định lệnh dưới dùng HTTP.
- Ảnh đã giải nén trong `data/processed/images/train` và `data/processed/images/law`; cần ít nhất 11 GiB VRAM trống. Không đọc câu hỏi, đáp án hoặc citation; chưa xây retrieval/OCR/detector.

### Lệnh PowerShell từ root project

Nếu cần thiết lập trên máy mới hoặc tiếp tục tải checkpoint:

```powershell
$env:PYTHONUTF8="1"
$env:HF_HUB_DISABLE_XET="1"
.\.venv\Scripts\python.exe -m pip install transformers==4.57.1 bitsandbytes==0.49.0
.\.venv\Scripts\python.exe -u temp\download_vlm.py
```

Người dùng tự chạy inference sau khi tải hoàn tất:

```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u temp\describe_images.py --train-image train_1_3.jpg --crop-train
```

Lệnh trên chạy ba đầu vào: train toàn cảnh, crop cả biển chính/phụ `(655, 0, 1168, 682)` và `law/image001.png`. Thay `--train-image` bằng `train_1_1.jpg` hoặc `train_1_2.jpg` để thử các ảnh trước; mặc định là ảnh thứ nhất. Ảnh thứ nhất có hai crop nên tổng bốn đầu vào. Không còn option `--model` trỏ tới model cũ đã xóa.

Mong đợi: log model NF4, phiên bản thư viện, VRAM trống/ngân sách, log tiến trình khoảng mỗi 30 giây; output `temp/qwen_descriptions_<timestamp>.json` có `completed: true` khi đủ đầu vào, kèm prompt, model revision, quantization config, token, thời gian và tài nguyên. PNG crop nằm cạnh JSON. Không ghi đè kết quả trước; không tự sửa caption sai. Mở ảnh và đối chiếu chữ/số, hình dạng, nét gạch chéo, chi tiết bị thiếu hoặc bịa thêm. Agent không chạy inference hoặc test theo yêu cầu người dùng; trạng thái giữ `AWAITING_USER_TEST` đến khi người dùng phản hồi.

Lỗi thường gặp: thiếu model local thì chạy downloader đến `100% completed`; mạng ngắt thì chạy lại downloader để tiếp tục file `.incomplete`. Nếu lỗi SHA256/kích thước, kiểm tra đúng file được báo; không dùng checkpoint lỗi, chỉ xóa file đó nếu cần tải lại. Import không có Qwen3VL thì cài đúng Transformers trong `.venv`. Thiếu VRAM/OOM thì đóng ứng dụng dùng GPU rồi chạy lại; giữ nguyên giới hạn allocator. Lỗi encoding thì đặt `PYTHONUTF8=1`. Crop lỗi kích thước thì kiểm tra đúng ảnh gốc (680×454, 800×522, 1200×900). Output chạm 384 token có cảnh báo có thể bị cắt.

Các kết quả Qwen2.5-VL-3B trước đây được giữ để tham khảo: `temp/qwen_descriptions_20261009T214816588067Z.json` (train_1_1), `temp/qwen_descriptions_20261009T215035817639Z.json` (train_1_2), `temp/qwen_descriptions_20261010T112649092456Z.json` (train_1_3). Bản 3B từng đọc đúng hai khung giờ trên ảnh thứ ba nhưng bỏ dấu `>` trước `29 CHỖ` và mô tả sai nét gạch đỏ; crop chưa sửa hết lỗi. Chưa có kết quả Qwen3-VL để kết luận cải thiện.

## Reset retrieval và QA — 09/10/2026

Theo yêu cầu người dùng, đã xóa code BM25 baseline, visual/hybrid retrieval, adapter fine-tune, code QA (gồm chọn evidence và inference), toàn bộ experiment/kết quả retrieval và QA dev/public/private, cùng prediction submission và ZIP private đã sinh. Không triển khai pipeline thay thế.

Giữ nguyên `models/` (weight, source model, Hugging Face cache), `.venv` và cấu hình môi trường, `data/`, loader/validator/split, evaluator F2/Accuracy, Git và tài liệu nền tảng. `bm25_evaluation.py` chỉ còn evaluator, không còn command `baseline`. `artifacts/experiments/note.md` chỉ là ghi chú công thức metric, không phải kết quả thí nghiệm.

Stage 2–5 ở trạng thái `AWAITING_USER_TEST` để xác nhận việc dọn dẹp, không phải đã triển khai lại. Retrieval và QA hiện chưa có pipeline. Kiến trúc mới sẽ được duyệt khi người dùng yêu cầu xây lại; không tự tiếp tục Stage 6.

### Kiểm tra dọn dẹp

Điều kiện: mở PowerShell tại root project.

```powershell
git status --short
Get-ChildItem src\multimodal_legal_rag
Get-ChildItem artifacts -Recurse
Test-Path src\multimodal_legal_rag\grounded_qa.py
Test-Path src\multimodal_legal_rag\visual_retrieval.py
Test-Path src\multimodal_legal_rag\hybrid_retrieval.py
Test-Path .venv\Scripts\python.exe
Test-Path models\Visualized_m3.pth
Test-Path models\huggingface
Test-Path data\processed
```

Mong đợi: ba lệnh kiểm tra source trả về `False`, bốn lệnh cuối trả về `True`. Không còn prediction, metric, adapter, evidence hay submission trong `artifacts/`; có thể còn thư mục rỗng và `note.md`. Nếu file bị khóa khi dọn, đóng process giữ file rồi báo lại đường dẫn; không xóa model hay environment. Agent chỉ kiểm kê filesystem, không chạy code/test/pipeline.

File nguồn còn dùng: `data.py` (dữ liệu), `bm25_evaluation.py` (evaluator) và `__init__.py` (package).

## Stage 0 — Bàn giao

### Đã hoàn thành

- Khởi tạo Git repository.
- Thêm quy trình stage-gate và user-test trong `AGENTS.md`.
- Thêm cấu trúc package Python tối thiểu tại `src/multimodal_legal_rag`.
- Thêm `.gitignore` để loại trừ dữ liệu, model, artifact, cache và secrets.
- Tạo virtual environment riêng tại `.venv`; các stage không cài package vào Python hệ thống.
- Ghi nhận môi trường hiện tại: Python 3.12.0, Git 2.46.2, NVIDIA GeForce RTX 5060 Ti 16 GB, driver 596.49.

Stage 0 chưa cài PyTorch hoặc thư viện ML. Dependency sẽ được cài vào `.venv` ở stage sử dụng đầu tiên để tránh khóa phiên bản khi chưa biết chính xác pipeline và dữ liệu.

### Điều kiện tiên quyết

- Mở PowerShell tại thư mục gốc project:

```powershell
cd "C:\Users\ADMIN\Documents\code\Personal Project\multimodal-legal-qa"
```

Nếu `.venv` chưa tồn tại trên máy khác, tạo và kích hoạt bằng:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Python hệ thống chỉ được dùng một lần để tạo `.venv`. Sau đó mọi lệnh Python của project phải dùng interpreter trong `.venv`.

### Cách tự kiểm tra

Chạy lần lượt:

```powershell
git status --short
.\.venv\Scripts\python.exe --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); import multimodal_legal_rag; print(multimodal_legal_rag.__version__)"
```

Kết quả mong đợi:

- `git status --short` liệt kê các file mới chưa commit; không xuất hiện thư mục dữ liệu, model hoặc cache.
- Python trong `.venv` in ra `Python 3.12.0` hoặc một phiên bản Python tương thích được cài sau này.
- GPU in ra `NVIDIA GeForce RTX 5060 Ti, 16311 MiB, 596.49` hoặc thông tin driver mới hơn.
- Lệnh import cuối cùng in ra `0.1.0`.

Sau khi cả bốn lệnh thành công, hãy báo: `Stage 0 test thành công`. Khi đó agent mới đánh dấu Stage 0 là `COMPLETED`; Stage 1 vẫn chỉ bắt đầu khi bạn yêu cầu tiếp tục.

### Lỗi thường gặp

- `git status` cảnh báo không đọc được global ignore tại `.config/git/ignore`: đây là cảnh báo quyền truy cập của Git, không làm hỏng repository; Stage 0 vẫn đạt nếu danh sách file mới được hiển thị bên dưới cảnh báo.
- Không tạo được `.venv`: kiểm tra `python --version`; Python hệ thống chỉ đóng vai trò tạo virtual environment ban đầu.
- PowerShell chặn `Activate.ps1`: không cần kích hoạt; dùng trực tiếp `.\.venv\Scripts\python.exe` như các command kiểm tra ở trên.
- `nvidia-smi` không được nhận diện: kiểm tra/cài NVIDIA driver. Việc này không ảnh hưởng lệnh import package nhưng cần thiết trước các stage chạy model trên GPU.
- Import thất bại: bảo đảm đang chạy từ đúng thư mục gốc project và thư mục `src\multimodal_legal_rag` tồn tại.

## Nguồn tham khảo

Danh sách nguồn ban đầu nằm tại [multimodal_legal_qa_vlsp2025_sources.md](multimodal_legal_qa_vlsp2025_sources.md).

## Stage 1 — Bàn giao dữ liệu

### Đã hoàn thành

- Tải official repository vào `data/raw/VLSP2025-MLQA-TSR` (thư mục này không được Git theo dõi).
- Thêm loader và CLI thuần Python standard library tại `src/multimodal_legal_rag/data.py`.
- Validator kiểm tra JSON schema, ID trùng, loại câu hỏi/đáp án, citation, ảnh thiếu, ZIP lỗi và header ảnh JPG/PNG.
- Lệnh `prepare` giải nén ảnh vào các thư mục ổn định dưới `data/processed/images` và bỏ metadata `__MACOSX`.
- Chia 530 mẫu gốc thành 418 train và 112 dev với seed `2025`. Việc chia nhóm theo `image_id` bảo đảm không có cùng ảnh ở cả hai tập.
- Giữ nguyên 100 mẫu public test có nhãn làm holdout. Private test có 146 mẫu nhưng repository không cung cấp gold labels.
- Lệnh `split` tự in `sample_overlap` và `image_overlap` để kiểm tra trực tiếp việc chia dữ liệu.

Không tự sửa các bất thường trong nhãn gốc. Validator báo chúng thành warning để Stage 2 xử lý bằng quy tắc chuẩn hóa có kiểm thử.

### Ghi chú sử dụng dữ liệu

Biểu mẫu nằm tại:

```text
data/raw/VLSP2025-MLQA-TSR/VLSP 2025 data agreement.docx
```

Theo xác nhận của chủ dự án, data agreement không phải điều kiện chặn việc triển khai. Biểu mẫu vẫn được giữ nguyên theo official repository để tham khảo. Không bán, cho mượn, công bố hoặc phân phối lại dữ liệu; khi sử dụng trong luận văn phải trích dẫn challenge paper.

`sample_submission/submission.zip` chứa prediction mẫu cho 146 mẫu private test, không phải gold labels:

- `submission_task1.json`: prediction retrieval mẫu.
- `submission_task2.json`: prediction QA mẫu; 24 output bị cắt như `C. Xe ô` hoặc `Sai. Biển`, phù hợp với baseline dùng `max_new_tokens=5`.

Hai file `submission_task1_no_labels.json` và `submission_task2_no_labels.json` là input không nhãn. Không được dùng sample submission làm ground truth hoặc báo cáo Accuracy/F2 như kết quả private test.

Ban tổ chức đã xác nhận qua email rằng nhãn private test không được cung cấp. Kết quả private chính thức chỉ được lấy bằng cách gửi file dự đoán tại [Codabench VLSP 2025 MLQA-TSR](https://www.codabench.org/competitions/9525/), mục **Post Submission**, theo từng subtask. Vì vậy quy trình đánh giá được cố định như sau:

- Train split: huấn luyện hoặc xây index.
- Dev split: chọn model, prompt, top-k, threshold và fusion weight.
- Public test có nhãn: báo cáo đánh giá local sau khi đã khóa cấu hình.
- Private test không nhãn: sinh submission và gửi Codabench; điểm trả về được lưu cùng file dự đoán và cấu hình đã dùng.

### Thiết lập trên máy mới

Chỉ chạy lệnh tải nếu `data/raw/VLSP2025-MLQA-TSR` chưa tồn tại:

```powershell
git clone --depth 1 https://github.com/sonlam1102/VLSP2025-MLQA-TSR.git data/raw/VLSP2025-MLQA-TSR
```

Sau đó đặt biến để Python tìm package trong `src`:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
```

Hai biến trên chỉ áp dụng cho cửa sổ PowerShell hiện tại. Stage 1 không cần cài thêm package vào `.venv`.

### Cách tự kiểm tra

Từ thư mục gốc project, chạy lần lượt:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data validate
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data prepare
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data split
.\.venv\Scripts\python.exe -m multimodal_legal_rag.data stats
```

Kết quả mong đợi:

- Validate: `ERRORS: 0`; warning phản ánh bất thường vốn có trong dữ liệu chính thức và được in rõ theo từng nhóm.
- Prepare: `train: 304`, `public_test: 90`, `private_test: 104`, `law: 1576`.
- Split: 418 train, 112 dev, 243 ảnh train, 61 ảnh dev, `sample_overlap: 0`, `image_overlap: 0`.
- Stats: 530 mẫu train, 100 mẫu public test có nhãn, 146 mẫu private test không có gold, 402 article records thuộc 2 văn bản luật và 761 tham chiếu ảnh trong corpus.
- Các split được lưu tại `data/processed/splits`; ảnh đã giải nén nằm tại `data/processed/images`. Cả hai đều không được Git theo dõi.

Người dùng đã chạy trực tiếp các lệnh và xác nhận kết quả thành công ngày 12/09/2026. Stage 1 đã `COMPLETED`; Stage 2 chỉ bắt đầu khi có lệnh riêng.

### Lỗi thường gặp

- `No module named multimodal_legal_rag`: chạy lại `$env:PYTHONPATH="src"` trong đúng cửa sổ PowerShell.
- Chữ tiếng Việt bị lỗi khi in: chạy lại `$env:PYTHONUTF8="1"`.
- `destination path ... already exists` khi clone: dữ liệu đã được tải; bỏ qua lệnh clone và chạy validator.
- Validator báo thiếu file: kiểm tra clone đã hoàn tất và không đổi cấu trúc thư mục official repository.
- Warning về định dạng ảnh, citation và ID trùng là bất thường đã ghi nhận của dữ liệu gốc, không phải lỗi command. Không được tự sửa trực tiếp dữ liệu nguồn.

