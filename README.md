# Multimodal Legal RAG

Đồ án tốt nghiệp về truy hồi và hỏi đáp đa phương thức cho luật giao thông Việt Nam, dựa trên VLSP 2025 MLQA-TSR. Đây là hệ thống nghiên cứu, không phải công cụ tư vấn pháp lý.

## Tiến độ

| Stage | Nội dung | Trạng thái |
| --- | --- | --- |
| 0 | Khởi tạo và quản trị | `COMPLETED` |
| 1 | Dữ liệu | `COMPLETED` |
| 2 | Evaluator và baseline | `NOT_STARTED` |
| 3 | Visual retrieval | `NOT_STARTED` |
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
