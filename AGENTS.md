# AGENTS.md

## Mục tiêu

Xây dựng Multimodal Legal RAG cho bộ dữ liệu VLSP 2025 MLQA-TSR theo từng bước, để người dùng hiểu cách làm, tự chạy và kiểm tra kết quả.

## Quy trình bắt buộc

1. Trước khi làm việc, đọc toàn bộ `AGENTS.md`.
2. Chỉ làm trong phạm vi người dùng yêu cầu. Trước khi đổi phạm vi hoặc kiến trúc đã duyệt, giải thích lý do và chờ người dùng đồng ý. Khi tạo mới file hay tạo 1 file output kết quả mới phải giải thích cho người dùng tại sao phải tạo và phải nhận được sự đồng ý, không tự ý thêm file code mới hoặc 1 path mới trong project.
3. Với tác vụ chạy lâu như training, captioning, embedding hoặc inference, cung cấp lệnh để người dùng tự chạy. Chỉ tự chạy khi người dùng yêu cầu rõ ràng.
4. Giữ `README.md` ngắn gọn và phản ánh tiến trình hiện tại: cách thực hiện, điều kiện tiên quyết, lệnh chạy và kết quả từng bước. Cập nhật phần liên quan khi có thay đổi để phiên làm việc mới biết project đang ở đâu; không biến README thành nhật ký mọi thao tác.
5. Sau khi thay đổi, giải thích ngắn gọn đã làm gì. Khi người dùng báo lỗi, phân tích và sửa trong phạm vi đó; cập nhật lỗi và cách xử lý vào phần liên quan của README nếu cần để chạy lại.
6. Không tự ý xóa/thêm file, thay đổi pipeline hoặc làm thêm việc ngoài yêu cầu. Không tự ý thay đổi nội dung .gitignore
7. Không tạo command `self-check`, demo hoặc test riêng chỉ để kiểm tra nội bộ. Hướng dẫn kiểm tra bằng CLI/pipeline thực tế và evaluator của bước đó; chỉ thêm test riêng khi người dùng yêu cầu rõ ràng.
8. Tác vụ chạy lâu phải in log khoảng mỗi 30 giây: bước hiện tại, số đã xử lý/tổng số, phần trăm, thời gian đã chạy và ETA.

## Nguyên tắc kỹ thuật

- Chọn giải pháp nhỏ nhất đáp ứng yêu cầu, đo lường và tái lập được. Không thêm abstraction hoặc dependency chỉ để dùng trong tương lai.
- Đặt tên file, module và thư mục theo chức năng, đủ cụ thể để phân biệt các bước. Tránh tên quá chung, tên theo số stage hoặc chi tiết cấu hình, như `evaluation.py`, `stage2.py`, `k5`.
- Lệnh và đường dẫn trong README phải khớp code hiện tại và chạy được từ thư mục gốc project.
- Cài và chạy package của project trong `.venv`, không dùng Python hệ thống. Trong README, ưu tiên `.\.venv\Scripts\python.exe` để không phụ thuộc việc kích hoạt môi trường.
- Lưu cấu hình, prediction, metric và thông tin tài nguyên của mỗi thí nghiệm đủ để tái lập.

## Dữ liệu và bảo mật

- Không commit dataset VLSP, model weights, vector index, cache, file môi trường hoặc secrets.
