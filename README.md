# Multimodal Legal RAG

Đồ án tốt nghiệp về truy hồi và hỏi đáp đa phương thức cho luật giao thông Việt Nam, dựa trên VLSP 2025 MLQA-TSR. Đây là hệ thống nghiên cứu, không phải công cụ tư vấn pháp lý.

## Tiến độ

| Stage | Nội dung | Trạng thái |
| --- | --- | --- |
| 0 | Khởi tạo và quản trị | `COMPLETED` |
| 1 | Dữ liệu | `COMPLETED` |
| 2 | Evaluator và baseline | `COMPLETED` |
| 3 | Visual retrieval | `COMPLETED` |
| 4 | Hybrid retrieval | `COMPLETED` |
| 5 | Grounded legal QA | `AWAITING_USER_TEST` |
| 6 | Thực nghiệm luận văn | `NOT_STARTED` |
| 7 | Demo và đóng gói | `NOT_STARTED` |

Stage chỉ được đánh dấu `COMPLETED` sau khi người dùng chạy hướng dẫn kiểm tra và xác nhận thành công. Quy tắc đầy đủ nằm trong [AGENTS.md](AGENTS.md).

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

## Stage 2 — Bàn giao evaluator và baseline retrieval

### Đã hoàn thành

- Thêm evaluator khớp công thức chính thức: macro F2 (beta 2) cho retrieval và exact-match Accuracy cho QA.
- Chuẩn hóa Unicode NFC và khoảng trắng ở đáp án QA để `Đúng` và `Đúng` được so sánh nhất quán; không thay đổi nhãn nguồn.
- Thêm baseline BM25 text-only thuần Python standard library. Baseline chỉ dùng câu hỏi, lựa chọn và văn bản luật; ảnh được dành cho Stage 3.
- Gộp các article có cùng `(law_id, article_id)` thành một citation duy nhất, nên 402 article records tạo thành 398 citation có thể truy hồi.
- Mỗi lần chạy baseline lưu `config.json`, `predictions.json`, `metrics.json` và `resources.json` dưới `artifacts/experiments/<tên-thí-nghiệm>`.
- Cấu hình baseline mặc định là `top_k=5`, `k1=1.5`, `b=0.75`.

File chính: `src/multimodal_legal_rag/bm25_evaluation.py`. Không có dependency mới.

### Điều kiện tiên quyết

- Stage 1 đã tạo `data/processed/splits/dev.json`.
- Official corpus tồn tại tại `data/raw/VLSP2025-MLQA-TSR`.
- Chạy từ thư mục gốc project và đặt biến môi trường trong cửa sổ PowerShell hiện tại:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
```

### Cách tự kiểm tra

Chạy lần lượt:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation baseline
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\bm25-retrieval-dev\predictions.json
Get-ChildItem artifacts\experiments\bm25-retrieval-dev
```

Kết quả mong đợi:

- Baseline và evaluator in cùng một F2, `samples: 112`, `missing_predictions: 0`, `extra_predictions: 0`.
- Thư mục experiment có đúng bốn file `config.json`, `predictions.json`, `metrics.json`, `resources.json`.
- `artifacts/` vẫn bị Git bỏ qua và không được commit.

Để tự kiểm tra Accuracy bằng một file prediction QA, dùng:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task qa `
  --gold <đường-dẫn-gold.json> `
  --predictions <đường-dẫn-predictions.json>
```

### Lỗi thường gặp

- `No module named multimodal_legal_rag`: chạy lại `$env:PYTHONPATH="src"` trong cùng cửa sổ PowerShell.
- Thiếu `dev.json`: chạy lại lệnh `split` của Stage 1 trước khi chạy baseline.
- Thiếu corpus luật: kiểm tra official repository ở đúng đường dẫn Stage 1.
- `top-k phải trong khoảng ...`: dùng số nguyên dương không lớn hơn số citation trong corpus; mặc định là 5.
- Prediction thiếu sample được chấm 0 cho sample đó và được đếm tại `missing_predictions`; prediction thừa không góp vào điểm và được đếm tại `extra_predictions`.

Người dùng đã chạy trực tiếp các lệnh và xác nhận hoàn tất ngày 13/09/2026. Stage 2 đã `COMPLETED`; Stage 3 chỉ bắt đầu khi có lệnh riêng.

## Stage 3 — Bàn giao visual retrieval

### Đã hoàn thành

- Thêm visual retrieval tại `src/multimodal_legal_rag/visual_retrieval.py`, bám theo `BAAI/bge-visualized` với text encoder đa ngôn ngữ `BAAI/bge-m3`.
- Mỗi article được chia thành các text chunk tối đa 1.024 token, overlap 128 token. Mọi ảnh được tham chiếu tạo thêm một candidate ảnh + tiêu đề; text chunk và ảnh đều quy về citation gốc.
- Nội dung bảng HTML trong luật được giữ lại dưới dạng plain text trước khi chunk, không còn bị loại bỏ khỏi embedding.
- Query kết hợp ảnh, câu hỏi và các lựa chọn. Khi một article có nhiều ảnh, hệ thống lấy điểm candidate cao nhất rồi mới chọn `top_k`, nên citation trả về không bị trùng.
- Không còn cắt bỏ phần cuối article. Retrieval lấy điểm cao nhất trong toàn bộ text chunk và image candidate của cùng citation, nên căn cứ ở cuối văn bản vẫn có thể được truy hồi.
- Fine-tune một linear metric adapter 1024×1024 bằng multi-positive contrastive loss trên 418 mẫu `train.json`. Visualized-BGE gốc được đóng băng để vừa GPU 16 GB; khoảng 1,05 triệu tham số adapter được cập nhật.
- Command `train` đánh giá mỗi epoch trên 112 mẫu dev, lưu checkpoint có F2 dev cao nhất cùng `config.json`, `predictions.json`, `metrics.json`, `resources.json` và `history.json` dưới `artifacts/experiments/visual-bge-finetune`.
- Trong lúc chạy, CLI báo tiến trình cho encode corpus, train query, dev query và từng nhóm epoch, kèm phần trăm, thời gian đã chạy và ETA.
- Command `retrieve` bắt buộc nạp `adapter.pt`. Stage 4 dùng adapter này cho nhánh visual; Stage 5 dùng các citation do Stage 4 trả về, không dùng Visualized-BGE làm model sinh câu trả lời.
- Thêm query representation ablation qua `--query-mode`: `image-question-choices` (mặc định hiện tại), `image-question`, và `question-only`. Mode `question-only` không nạp ảnh query; mỗi mode phải huấn luyện adapter riêng và ghi vào experiment riêng.
- Đồng bộ training objective với evaluator bằng citation-level contrastive loss. Cấu hình Stage 3 được chọn là `image-question-choices`, citation-level loss, không thêm local image context và `top_k=5`; adapter nằm tại `artifacts/experiments/improvements/citation-level-loss/visual-bge-citation-loss/adapter.pt`.

Stage này chưa tạo cache embedding riêng. Chunking làm tăng số candidate và thời gian encode; chỉ thêm cache sau khi lần chạy thực tế xác nhận đây là nút thắt.

Kết quả zero-shot dev ngày 14/09/2026 được giữ làm mốc trước fine-tune: Visualized-BGE đạt F2 `0.1393` so với BM25 `0.2069`, mất khoảng 400 giây và dùng peak VRAM khoảng 12.36 GB. Visual có 12 hit riêng mà BM25 bỏ lỡ. Kết quả fine-tune phải được đo lại bằng command bên dưới; không giả định trước rằng fine-tune chắc chắn cải thiện dev.

### Điều kiện tiên quyết

Stage 1 phải có `data/processed/splits/dev.json`, ảnh đã giải nén dưới `data/processed/images`, và máy có đủ dung lượng cho model. Từ thư mục gốc project, cài dependency trong `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers==4.44.2 sentencepiece==0.2.0 timm==1.0.9 einops==0.8.0 ftfy==6.2.3
New-Item -ItemType Directory -Force models | Out-Null
git clone --branch v1.4.2 --depth 1 https://github.com/FlagOpen/FlagEmbedding.git models/FlagEmbedding
curl.exe -L "https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth?download=true" -o models\Visualized_m3.pth
```

Không chạy `pip install -e` cho `research/visual_bge`: bản upstream `v1.4.2` tạo metadata nhưng không expose được module `visual_bge`. CLI của project nạp trực tiếp source cố định đã clone và đặt Hugging Face cache trong `models/huggingface`.

`models/` bị Git bỏ qua. File `Visualized_m3.pth` khoảng 1.75 GB; xác minh file tải đúng bằng:

```powershell
(Get-FileHash models\Visualized_m3.pth -Algorithm SHA256).Hash.ToLower()
```

Kết quả mong đợi:

```text
d14e7e8f2618b80d3f4a3283c08f79a16f09ce37d4447dac24117b44df6bd069
```

Sau đó đặt biến môi trường cho cửa sổ PowerShell hiện tại:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
```

### Chạy baseline trước fine-tune

Command `zero-shot` dùng trực tiếp Visualized-BGE, không đọc hoặc tạo `adapter.pt`. Đây chỉ là mốc so sánh/ablation; Stage 4 trở đi vẫn dùng adapter đã fine-tune.

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.visual_retrieval zero-shot
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\visual-bge-zero-shot-dev\predictions.json
```

Kết quả nằm trong `artifacts/experiments/visual-bge-zero-shot-dev`. Vì code hiện tại đã chuyển sang chunk 1.024 token và giữ nội dung bảng, kết quả này có thể khác mốc zero-shot cũ F2 `0.1393`.

### Cách tự kiểm tra

Fine-tune trên train, chọn checkpoint có F2 cao nhất trên dev, rồi dùng evaluator độc lập để đối chiếu:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.visual_retrieval train
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\visual-bge-finetune\predictions.json
Get-ChildItem artifacts\experiments\visual-bge-finetune
```

Kết quả mong đợi:

- Trong quá trình chạy xuất hiện log dạng `[corpus] ...`, `[train-queries] ...`, `[dev-queries] ...` và `[training] 5/50 ...`.
- Training loss phải là số hữu hạn; log có loss và dev F2 theo epoch. Retrieval và evaluator in cùng F2 của `best_epoch`, `samples: 112`, `missing_predictions: 0`, `extra_predictions: 0`.
- Experiment có sáu file `adapter.pt`, `config.json`, `predictions.json`, `metrics.json`, `resources.json`, `history.json`.
- `resources.json` báo `device: "cuda"`, tên GPU và peak VRAM; `artifacts/` vẫn không được Git theo dõi.

Người dùng đã chạy trực tiếp baseline và các ablation, xác nhận Stage 3 hoàn tất ngày 15/09/2026. Stage 3 đã `COMPLETED`; Stage 4 chỉ bắt đầu khi có lệnh riêng.

### Query representation ablation

Giữ `top_k=5` đã được chọn trên dev. Kết quả hiện tại tại `artifacts/experiments/visual-bge-finetune` là baseline `image-question-choices`; không ghi đè thư mục này. Huấn luyện hai representation còn lại với cùng hyperparameter và output riêng:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.visual_retrieval train `
  --query-mode image-question `
  --top-k 5 `
  --output artifacts\experiments\improvements\query-representation\visual-bge-query-image-question

.\.venv\Scripts\python.exe -m multimodal_legal_rag.visual_retrieval train `
  --query-mode question-only `
  --top-k 5 `
  --output artifacts\experiments\improvements\query-representation\visual-bge-query-question-only
```

Sau mỗi lệnh, dùng evaluator độc lập:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\improvements\query-representation\visual-bge-query-image-question\predictions.json

.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\improvements\query-representation\visual-bge-query-question-only\predictions.json
```

Kết quả mong đợi: mỗi experiment có `adapter.pt`, `config.json`, `predictions.json`, `metrics.json`, `resources.json`, `history.json`; `config.json` ghi đúng `query_mode`; evaluator báo `samples: 112`, `missing_predictions: 0`, `extra_predictions: 0`. So sánh F2 với baseline `image-question-choices` là `0.4649128757684249`, rồi chọn duy nhất representation có F2 dev cao nhất. Không thay đổi hyperparameter khác trong ablation này.

Kết quả đã đo: `image-question` đạt F2 `0.4230114246`, `question-only` đạt `0.4108874375`, đều thấp hơn baseline. Vì vậy các artifact được giữ làm ablation tại `artifacts/experiments/improvements/query-representation`, còn cấu hình được chọn vẫn là `image-question-choices`.

### Citation-level loss improvement

Loss mặc định `candidate` được giữ để tái lập baseline. Thí nghiệm tiếp theo gộp score các chunk/ảnh bằng max theo citation trước khi tính contrastive loss, đồng bộ training objective với inference. Giữ query representation thắng là `image-question-choices` và `top_k=5`:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.visual_retrieval train `
  --loss-level citation `
  --query-mode image-question-choices `
  --top-k 5 `
  --output artifacts\experiments\improvements\citation-level-loss\visual-bge-citation-loss

.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\improvements\citation-level-loss\visual-bge-citation-loss\predictions.json
```

Kết quả mong đợi: experiment có đủ sáu artifact, `config.json` ghi `loss_level: "citation"` và `query_mode: "image-question-choices"`, evaluator báo đủ 112 mẫu. So sánh F2 với candidate-level baseline `0.4649128758`; chỉ chọn citation-level loss nếu F2 dev cao hơn. Nếu không, giữ adapter baseline và lưu experiment này làm ablation âm.

Kết quả đã đo: citation-level loss đạt F2 `0.4970959524`, cao hơn candidate-level baseline `0.4649128758`; precision tăng từ `0.2625` lên `0.2768` và recall tăng từ `0.6148` lên `0.6661`. Khi kiểm tra lại top-k với adapter mới, k3 đạt `0.4811664585`, k5 đạt `0.4970959524`, k7 đạt `0.4566970293`; vì vậy tiếp tục khóa `top_k=5`.

### Image local context improvement

Image candidate mặc định chỉ dùng title để giữ hành vi cũ. `--image-context-tokens 256` bổ sung tối đa 256 token ngay trước và 256 token ngay sau marker ảnh sau khi chuyển bảng HTML thành plain text. Thí nghiệm giữ nguyên query representation, citation-level loss và top-k đã chọn:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.visual_retrieval train `
  --loss-level citation `
  --query-mode image-question-choices `
  --image-context-tokens 256 `
  --top-k 5 `
  --output artifacts\experiments\improvements\image-local-context\visual-bge-local-context-256

.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\improvements\image-local-context\visual-bge-local-context-256\predictions.json
```

Kết quả mong đợi: experiment có đủ sáu artifact; `config.json` ghi `loss_level: "citation"`, `query_mode: "image-question-choices"`, `image_context_tokens_per_side: 256` và `top_k: 5`; evaluator báo đủ 112 mẫu. So sánh F2 với citation-level baseline `0.4970959524`. Chỉ chọn local context nếu F2 dev cao hơn; không thử thêm kích thước cửa sổ trước khi có kết quả này.

Kết quả đã đo: local context 256 đạt F2 `0.4936458544`, precision `0.2767857143`, recall `0.6579081633`; thấp hơn citation-level baseline F2 `0.4970959524`. Vì vậy Stage 3 khóa `image_context_tokens=0`; experiment được giữ làm ablation âm.

### Lỗi thường gặp

- `Thiếu dependency Stage 3`: chạy lại hai lệnh `pip install` ở phần điều kiện tiên quyết trong đúng `.venv`.
- `No module named 'visual_bge'` dù `pip show visual_bge` có kết quả: đây là lỗi editable install của upstream. Cập nhật code hiện tại và chạy lại; không cần cài editable.
- `destination path ... already exists` khi clone: bỏ qua clone nếu `models/FlagEmbedding/research/visual_bge` đã tồn tại và tiếp tục với bước tải weight.
- `Token indices sequence length ... 8730 > 8192` kèm `device-side assert`: phiên bản cũ encode toàn article. Code hiện tại chia article thành chunk 1.024 token; đóng terminal đã gặp CUDA assert, mở PowerShell mới rồi chạy lại.
- `Thiếu model weight` hoặc hash không khớp: xóa file tải dở và chạy lại lệnh `curl.exe`; không dùng file sai hash.
- Lỗi tải `BAAI/bge-m3`: lần chạy đầu cần Internet để tải config/tokenizer vào `models/huggingface`; kiểm tra kết nối và quyền ghi thư mục `models` rồi chạy lại.
- `CUDA out of memory`: đóng process đang dùng GPU rồi chạy lại. Baseline hiện encode tuần tự; không tăng batch size.
- `Thiếu fine-tuned adapter`: chạy `...visual_retrieval train` thành công trước khi dùng command `retrieve`.
- `device: "cpu"`: kiểm tra `nvidia-smi`, rồi xác nhận đã cài wheel PyTorch từ index `cu128`, không phải wheel CPU.
- `No module named multimodal_legal_rag`: chạy lại `$env:PYTHONPATH="src"` trong cùng cửa sổ PowerShell.

## Stage 4 — Bàn giao hybrid retrieval

### Đã hoàn thành

- Thêm `src/multimodal_legal_rag/hybrid_retrieval.py`, kết hợp nhánh example-based multimodal với nhánh corpus visual của Stage 3.
- Nhánh visual bắt buộc nạp adapter citation-level tốt nhất của Stage 3 tại `artifacts/experiments/improvements/citation-level-loss/visual-bge-citation-loss/adapter.pt`; không dùng zero-shot.
- Nhánh example tìm query train gần nhất trong chính không gian embedding đã fine-tune, rồi cộng similarity của các láng giềng cùng trích dẫn trước khi truyền citation sang query dev. Split theo `image_id` của Stage 1 ngăn ảnh train/dev trùng nhau.
- Giữ cấu hình visual đã khóa: query `image-question-choices`, không thêm local image context. Nhánh corpus lấy top 20 để tạo candidate pool.
- Chuẩn hóa min-max riêng từng nhánh và tìm `example_k` trong `1, 3, 5`, `example_weight` trong `0, 0.25, 0.5, 0.75`, `top_k` trong `3, 5, 7` trên dev.
- Lưu cấu hình, prediction, metric, resource, toàn bộ bảng tìm kiếm và metric độc lập của hai nhánh để tái lập.
- Phép cộng consensus đã được thử trên dev: cấu hình `example_k=3`, `example_weight=0.5`, `top_k=5` đạt F2 `0.5445245573`, precision `0.3089285714`, recall `0.7216836735`; cao hơn phép lấy maximum trước đó có F2 `0.5330363192`.

Không thêm model, dependency, FAISS hoặc cache embedding. BM25-RRF trước đó đạt tốt nhất khi `visual_weight=1.0`, tức không tạo cải thiện, nên được giữ tại artifact cũ làm ablation âm và không còn là pipeline Stage 4 chính.

### Điều kiện tiên quyết

- Stage 1 đã tạo `data/processed/splits/dev.json` và ảnh dưới `data/processed/images`.
- Dependency, model weight và source Visualized-BGE của Stage 3 vẫn tồn tại.
- Adapter được chọn của Stage 3 tồn tại tại đường dẫn mặc định nêu trên.
- Từ thư mục gốc project, đặt biến môi trường trong cửa sổ PowerShell hiện tại:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
```

### Cách tự kiểm tra

Chạy pipeline hybrid trên dev, sau đó đối chiếu bằng evaluator độc lập:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.hybrid_retrieval
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\hybrid-example-retrieval-dev\predictions.json
Get-ChildItem artifacts\experiments\hybrid-example-retrieval-dev
```

Kết quả mong đợi:

- Quá trình chạy có log `[model]`, `[corpus]`, `[train-examples]` và `[dev-queries]` khoảng mỗi 30 giây; đây là indexing/inference, không phải training.
- CLI hybrid và evaluator in cùng F2, `samples: 112`, `missing_predictions: 0`, `extra_predictions: 0`.
- Thư mục experiment có sáu file `config.json`, `predictions.json`, `metrics.json`, `resources.json`, `search.json`, `branch_metrics.json`.
- `config.json` ghi adapter citation-level của Stage 3, `query_mode: "image-question-choices"`, `image_context_tokens_per_side: 0` và cấu hình fusion được chọn.
- Với cùng dữ liệu và adapter đã nêu, cấu hình được chọn dự kiến là `example_k=3`, `example_weight=0.5`, `top_k=5`, F2 khoảng `0.5445245573`.
- `branch_metrics.json` cho phép so sánh corpus-only, example-only và hybrid ở cùng `top_k`; cấu hình chỉ được chấp nhận nếu hybrid vượt corpus-only F2 `0.4970959524` trên dev.
- `resources.json` báo `device: "cuda"`; `artifacts/` vẫn không được Git theo dõi.

Người dùng đã duyệt sum-consensus và xác nhận kết thúc Stage 4 ngày 16/09/2026. Confidence gating và dynamic top-k chỉ được thử trong bộ nhớ, không cải thiện đủ để chọn nên không được thêm vào pipeline. Stage 4 đã `COMPLETED`; Stage 5 chỉ bắt đầu khi có lệnh riêng.

### Lỗi thường gặp

- `Thiếu fine-tuned adapter`: kiểm tra đúng artifact citation-level đã được chọn ở Stage 3; không thay bằng adapter zero-shot hoặc candidate-level.
- `Thiếu dependency Stage 3`, model weight hay source Visualized-BGE: làm lại phần điều kiện tiên quyết của Stage 3 trong đúng `.venv`.
- `Thiếu ảnh`: chạy lại `data prepare` của Stage 1 và giữ `--query-images` mặc định là `data/processed/images/train` cho dev.
- `CUDA out of memory`: đóng process đang dùng GPU rồi mở PowerShell mới và chạy lại; pipeline encode tuần tự, không tăng batch size.
- Output đã tồn tại sẽ được ghi lại trong đúng experiment dev. Nếu cần giữ một lần chạy cũ, truyền `--output` sang một thư mục experiment mới.
- Hybrid không vượt F2 `0.4970959524`: giữ visual corpus-only làm cấu hình chính và ghi nhận example fusion là ablation âm; không tiếp tục chọn model bằng public/private test.

## Stage 5 — Bàn giao grounded legal QA

### Đã hoàn thành

- Thêm `src/multimodal_legal_rag/grounded_qa.py`, dùng `Qwen/Qwen2.5-VL-3B-Instruct` ở BF16 để trả lời từ ảnh câu hỏi, lựa chọn và evidence thuộc citation Stage 4.
- Citation chính được đọc từ `artifacts/experiments/hybrid-example-retrieval-dev/predictions.json`; Stage 5 không chạy lại hoặc thay đổi retriever.
- Evidence không được chọn bằng keyword. Pipeline dùng lại Visualized-BGE và adapter citation-level đã fine-tune ở Stage 3 để so khớp semantic giữa ảnh + câu hỏi + choices với các text chunk hoặc ảnh luật nằm trong từng citation.
- Các citation bất thường vốn có trong gold được resolve mà không sửa dữ liệu nguồn: hậu tố số `.0` ánh xạ về article số tương ứng, nhãn ghép như `22 B.15` xét cả hai article, và hai alias đã kiểm kê được ánh xạ tường minh `G1.1 → G.1`, `I.414 → E.14`. Artifact giữ cả nhãn nguồn và article corpus đã chọn; code không dùng fuzzy matching có thể nhầm article chỉ nhắc lại cùng ký hiệu.
- Mỗi citation giữ candidate có cosine similarity cao nhất. Text corpus vẫn dùng chunk 1.024 token, overlap 128 token; image candidate được truyền trực tiếp cho Qwen khi được chọn.
- Visualized-BGE được giải phóng trước khi nạp Qwen để hai model không cùng chiếm VRAM.
- So sánh đúng hai prompt `direct-answer` và `evidence-first` trên dev. Chọn Accuracy cao nhất; nếu hòa, giữ `direct-answer` vì đơn giản hơn.
- Kết quả chính dùng citation Stage 4. Một lượt riêng dùng gold citation được lưu làm oracle upper bound, không tham gia chọn prompt và không thay thế score chính.
- Generation là greedy, `max_new_tokens=8`; output chỉ chấp nhận `A/B/C/D` hoặc `Đúng/Sai`. Output không hợp lệ được ghi lại và tính sai, không tự thay bằng nhãn đoán.
- Các bước lâu in log khoảng mỗi 30 giây với số mẫu, phần trăm, elapsed và ETA. Đây là indexing/inference, không phải training.

Experiment mặc định nằm tại `artifacts/experiments/grounded-qa-dev` và gồm tám file:

```text
config.json
evidence.json
prompt_search.json
predictions.json
metrics.json
oracle_predictions.json
oracle_metrics.json
resources.json
```

`evidence.json` chỉ lưu citation, candidate index, modality và similarity để tái lập lựa chọn; không sao chép toàn bộ corpus luật.

### Điều kiện tiên quyết

- Stage 1 đã tạo `data/processed/splits/dev.json` và ảnh dưới `data/processed/images`.
- Dependency, source Visualized-BGE, weight và adapter citation-level của Stage 3 vẫn tồn tại.
- Stage 4 đã tạo `artifacts/experiments/hybrid-example-retrieval-dev/predictions.json` với top-5 citation cho đủ 112 mẫu dev.
- Windows native là môi trường chính. Ubuntu/WSL chỉ cần dùng nếu máy gặp lỗi CUDA/operator không xử lý được trên Windows.
- Cài dependency Stage 5 vào đúng `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade `
  transformers==4.49.0 `
  accelerate==1.4.0 `
  qwen-vl-utils==0.0.8
```

Không cài `bitsandbytes`, FlashAttention hoặc package vào Python hệ thống. Lần chạy đầu cần Internet để tải `Qwen/Qwen2.5-VL-3B-Instruct`; model được cache dưới `models/huggingface` và không được Git theo dõi.

Từ thư mục gốc project, đặt biến môi trường trong cửa sổ PowerShell hiện tại:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
```

### Cách tự kiểm tra

Chạy pipeline QA thực tế, sau đó đối chiếu prediction chính bằng evaluator Stage 2:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.grounded_qa
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task qa `
  --gold data\processed\splits\dev.json `
  --predictions artifacts\experiments\grounded-qa-dev\predictions.json
Get-ChildItem artifacts\experiments\grounded-qa-dev
```

Kết quả mong đợi:

- Có log `[evidence-indexing]`, `[evidence-selection]`, `[prompt-evaluation:direct-answer]`, `[prompt-evaluation:evidence-first]` và `[oracle-inference]`; không có bước nào được gọi là training.
- Pipeline và evaluator in cùng Accuracy chính, `samples: 112`, `missing_predictions: 0`, `extra_predictions: 0`.
- Mọi `answer` trong `predictions.json` là `A/B/C/D` cho Multiple choice hoặc `Đúng/Sai` cho Yes/No. `prompt_search.json` không có invalid output ở prompt được chọn.
- Mỗi evidence trong `evidence.json` thuộc đúng citation tương ứng; `modality` là `text` hoặc `image`, có `candidate_index`, `similarity`, citation nguồn và `corpus_law_id`/`corpus_article_id` đã resolve.
- `config.json` ghi adapter citation-level Stage 3, `selected_prompt`, `evidence_per_citation: 1`, BF16, SDPA và greedy decoding.
- `oracle_metrics.json` chỉ là upper bound. Accuracy chính để báo cáo pipeline nằm trong `metrics.json`.
- `resources.json` báo `device: "cuda"`, tên GPU, peak VRAM, thời gian và đúng phiên bản dependency.
- Thư mục experiment có đủ tám file nêu trên; `artifacts/` và model cache vẫn không được Git theo dõi.

Người dùng đã chạy và chấp nhận kết quả dev ngày 17/09/2026: `64/112`, Accuracy `0.5714285714`; oracle dùng gold citation đạt `75/112`, Accuracy `0.6696428571`. `direct-answer` được khóa làm prompt chính. Stage 5 đang `AWAITING_USER_TEST` cho lần đánh giá public test cuối cùng; Stage 6 chỉ bắt đầu khi có lệnh riêng.

### Lỗi thường gặp

- `Thiếu dependency Stage 5` hoặc `KeyError: 'qwen2_5_vl'`: chạy lại đúng lệnh cài ba package ở trên trong `.venv`; không dùng Transformers 4.44.2 cũ.
- Lỗi tải `Qwen/Qwen2.5-VL-3B-Instruct`: kiểm tra Internet và quyền ghi `models/huggingface`, rồi chạy lại. Không commit model cache.
- `Thiếu fine-tuned adapter Stage 3`: kiểm tra `artifacts/experiments/improvements/citation-level-loss/visual-bge-citation-loss/adapter.pt`; không thay bằng zero-shot hoặc candidate-level adapter.
- Thiếu prediction Stage 4: chạy lại command hybrid retrieval ở phần Stage 4 trước khi chạy QA.
- Citation không tồn tại, prediction thiếu/trùng ID hoặc thiếu ảnh: sửa đầu vào tương ứng; pipeline dừng trước khi nạp Qwen thay vì âm thầm bỏ qua.
- Gold báo `G1.1`, `I.414`, `9.0`, `22.0` hoặc `22 B.15`: đây là các nhãn không trùng trực tiếp ID article cấp cao trong corpus chính thức. Code hiện tại tự resolve có kiểm tra và ghi mapping vào `evidence.json`; không sửa JSON nguồn bằng tay.
- `Stage 5 yêu cầu CUDA`: kiểm tra `nvidia-smi` và wheel PyTorch CUDA 12.8 trong `.venv`. Pipeline BF16 này không có CPU fallback.
- `CUDA out of memory`: đóng process đang dùng GPU, mở PowerShell mới và chạy lại. Pipeline đã giải phóng retriever trước khi nạp Qwen; không cài quantization hoặc đổi kiến trúc trong Stage 5.
- `[Errno 22] Invalid argument: '/C:/...'`: phiên bản cũ truyền ảnh bằng `file://` URI nên `%20` trong đường dẫn Windows không được giải mã. Code hiện tại truyền native Windows path và đặt giới hạn pixel trên từng ảnh; cập nhật code rồi chạy lại.
- Cảnh báo Hugging Face về symlink Windows hoặc thiếu `hf_xet`: cache vẫn hoạt động và không làm sai kết quả; không cần chạy PowerShell bằng Administrator hoặc cài thêm package. Model đã tải xong sẽ được dùng lại từ cache.
- Có invalid output trong `prompt_search.json`: gửi lại raw output và log để sửa parser/prompt trong Stage 5; không chỉnh prediction thủ công.

### Đánh giá public test với cấu hình đã khóa

Output được tách riêng dưới `artifacts/public-test`, ngang cấp với `artifacts/experiments`. Không tìm lại tham số trên public test: retrieval giữ đúng `example_k=3`, weight `0.5`, top-5 đã chọn trên dev; QA chỉ chạy `direct-answer` và bỏ oracle.

Chạy lần lượt từng bước trong PowerShell từ thư mục gốc project:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
New-Item -ItemType Directory -Force artifacts\public-test
```

1. Chạy retrieval với cấu hình dev đã khóa:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.hybrid_retrieval `
  --dev "data\raw\VLSP2025-MLQA-TSR\dataset\public_test data\vlsp_2025_public_test.json" `
  --query-images data\processed\images\public_test `
  --example-ks 3 `
  --example-weights 0.5 `
  --top-ks 5 `
  --output artifacts\public-test\retrieval
```

2. Đánh giá retrieval:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task retrieval `
  --gold "data\raw\VLSP2025-MLQA-TSR\dataset\public_test data\vlsp_2025_public_test.json" `
  --predictions artifacts\public-test\retrieval\predictions.json
```

3. Chạy đúng một lượt QA, không prompt search và không oracle:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.grounded_qa `
  --input "data\raw\VLSP2025-MLQA-TSR\dataset\public_test data\vlsp_2025_public_test.json" `
  --query-images data\processed\images\public_test `
  --retrieval-predictions artifacts\public-test\retrieval\predictions.json `
  --prompt direct-answer `
  --skip-oracle `
  --output artifacts\public-test\qa
```

4. Đánh giá QA:

```powershell
.\.venv\Scripts\python.exe -m multimodal_legal_rag.bm25_evaluation evaluate `
  --task qa `
  --gold "data\raw\VLSP2025-MLQA-TSR\dataset\public_test data\vlsp_2025_public_test.json" `
  --predictions artifacts\public-test\qa\predictions.json
```

Mong đợi cả hai evaluator báo `samples: 100`, không thiếu/thừa prediction. Thư mục `qa` không có artifact oracle; public test chỉ dùng để báo cáo, không dùng kết quả này để đổi model, prompt hoặc tham số.

## Nộp kết quả private test lên Codabench

Ban tổ chức không phát hành nhãn private test. Điểm chính thức chỉ được lấy bằng cách nộp prediction tại [Codabench VLSP 2025 MLQA-TSR](https://www.codabench.org/competitions/9525/), phase **Post Submission**.

### File của Subtask 1 — Retrieval

Tạo `submission_task1.json` gồm đúng 146 mẫu theo thứ tự/input từ `submission_task1_no_labels.json`. Giữ nguyên `id`, `image_id`, `question` và thêm prediction vào `relevant_articles`:

```json
[
  {
    "id": "private_test_private_test_1",
    "image_id": "private_test_1_1",
    "question": "...",
    "relevant_articles": [
      {
        "law_id": "QCVN 41:2024/BGTVT",
        "article_id": "B.7"
      }
    ]
  }
]
```

### File của Subtask 2 — QA

Tạo `submission_task2.json` gồm đúng 146 mẫu. Giữ nguyên các field của `submission_task2_no_labels.json` và thêm `answer`:

```json
[
  {
    "id": "private_test_private_test_1",
    "image_id": "private_test_1_1",
    "question": "...",
    "question_type": "Multiple choice",
    "relevant_articles": [
      {
        "law_id": "QCVN 41:2024/BGTVT",
        "article_id": "22"
      }
    ],
    "answer": "A"
  }
]
```

Giữ nguyên `relevant_articles` đã có trong input Subtask 2. `answer` chỉ được là `A`, `B`, `C`, `D` đối với multiple choice hoặc `Đúng`, `Sai` đối với Yes/No; không kèm giải thích.

### Đóng gói

Đặt hai JSON trong `artifacts/submission`, sau đó chạy:

```powershell
Compress-Archive `
  -Path "artifacts\submission\submission_task1.json","artifacts\submission\submission_task2.json" `
  -DestinationPath "artifacts\submission\submission.zip" `
  -Force
tar -tf artifacts\submission\submission.zip
```

Kết quả `tar -tf` phải chỉ ra hai file ở ngay root ZIP, không nằm trong thư mục con:

```text
submission_task1.json
submission_task2.json
```

Command sinh và kiểm tra submission tự động sẽ được bổ sung tại các stage triển khai retrieval và QA. Không chỉnh JSON thủ công khi đã có command này.

### Thao tác trên Codabench

1. Đăng nhập và đăng ký/accept terms của competition nếu được yêu cầu.
2. Mở **Participate** và chọn phase **Post Submission**.
3. Chọn subtask tương ứng nếu giao diện tách riêng hai subtask.
4. Nhấn biểu tượng đính kèm, chọn `artifacts/submission/submission.zip` và thêm mô tả cấu hình, ví dụ `hybrid-rag-v1-seed-2025`.
5. Chờ trạng thái `Finished`, sau đó ghi lại F2 của Subtask 1 và Accuracy của Subtask 2.
6. Nếu scorer yêu cầu nộp riêng từng subtask, tạo ZIP chỉ chứa đúng JSON của subtask được chọn; không đổi tên JSON.

### Lưu kết quả cho luận văn

Mỗi lần nộp phải lưu cùng prediction và cấu hình:

```text
artifacts/experiments/<experiment-name>/
├── config.json
├── submission_task1.json
├── submission_task2.json
├── submission.zip
└── codabench_result.json
```

Nội dung tối thiểu của `codabench_result.json`:

```json
{
  "submitted_at": "YYYY-MM-DD HH:mm",
  "description": "hybrid-rag-v1-seed-2025",
  "task1_f2": null,
  "task2_accuracy": null
}
```

Chỉ nộp sau khi đã khóa cấu hình bằng dev/public test. Không dùng điểm private để tiếp tục chọn model, prompt, top-k, threshold hoặc fusion weight.
