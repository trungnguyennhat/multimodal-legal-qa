# Multimodal Legal RAG

Đồ án tốt nghiệp về truy hồi và hỏi đáp đa phương thức cho luật giao thông Việt Nam, dựa trên VLSP 2025 MLQA-TSR. Đây là hệ thống nghiên cứu, không phải công cụ tư vấn pháp lý.

## Tiến độ

| Stage | Nội dung | Trạng thái |
| --- | --- | --- |
| 0 | Khởi tạo và quản trị | `COMPLETED` |
| 1 | Dữ liệu | `COMPLETED` |
| 2 | Evaluator và baseline | `COMPLETED` |
| 3 | Visual retrieval | `AWAITING_USER_TEST` |
| 4 | Hybrid retrieval | `NOT_STARTED` |
| 5 | Grounded legal QA | `NOT_STARTED` |
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

Hãy gửi output của các lệnh trên. Stage 3 giữ trạng thái `AWAITING_USER_TEST` và chỉ chuyển thành `COMPLETED` sau khi bạn xác nhận chạy thành công.

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
