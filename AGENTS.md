# AGENTS.md

## Mục tiêu

Xây dựng Multimodal Legal RAG cho bộ dữ liệu VLSP 2025 MLQA-TSR theo từng stage để người dùng có thể hiểu, tự chạy và xác nhận từng phần.

## Quy trình bắt buộc

1. Trước khi làm việc, đọc toàn bộ `AGENTS.md` và `README.md`.
2. Chỉ thực hiện stage mà người dùng yêu cầu rõ ràng. Không chuẩn bị trước code của stage sau.
3. Không đổi phạm vi hoặc kiến trúc đã duyệt nếu chưa giải thích và được người dùng đồng ý.
4. Cuối mỗi stage:
   - cập nhật bảng tiến độ trong `README.md` thành `AWAITING_USER_TEST`;
   - ghi những gì đã làm, điều kiện tiên quyết và các file chính;
   - ghi lệnh PowerShell chính xác để người dùng tự kiểm tra;
   - ghi kết quả mong đợi và cách xử lý lỗi đã biết;
   - dừng và chờ người dùng phản hồi.
5. Nếu người dùng báo command hoặc code lỗi:
   - phân tích output người dùng cung cấp và sửa nguyên nhân trong chính stage đó;
   - cập nhật command trong `README.md` nếu command thay đổi;
   - giữ trạng thái `AWAITING_USER_TEST` và yêu cầu người dùng chạy lại.
6. Chỉ chuyển stage hiện tại thành `COMPLETED` khi người dùng xác nhận đã kiểm tra thành công. Chỉ bắt đầu stage tiếp theo khi người dùng ra lệnh tiếp tục.
7. Sau khi viết hoặc sửa code, agent không tự chạy code, test, pipeline hay CLI của project, trừ khi người dùng yêu cầu rõ ràng trong tin nhắn hiện tại. Agent phải cung cấp lệnh PowerShell chính xác để người dùng tự chạy và kiểm soát output.
8. Agent không tự tạo, ghi đè, đổi tên hoặc xóa prediction, metric, experiment, index, cache hay artifact sinh ra khi chạy project. Các file output này chỉ được tạo bởi lệnh người dùng trực tiếp chạy, trừ khi người dùng yêu cầu agent thao tác rõ ràng.
9. Việc được yêu cầu triển khai code cho phép agent tạo/sửa source code và tài liệu trong đúng phạm vi stage; không đồng nghĩa với quyền tự chạy pipeline hoặc sinh output.
10. Không tạo command `self-check`, demo kiểm tra riêng hoặc test chỉ để kiểm tra nội bộ. Hướng dẫn người dùng kiểm tra bằng chính command chạy pipeline/CLI thực tế và evaluator của stage; chỉ thêm test riêng khi người dùng yêu cầu rõ ràng.
11. Mọi tác vụ có thể chạy lâu phải in log tiến trình ngắn gọn khoảng mỗi 30 giây, gồm bước hiện tại, số lượng đã xử lý/tổng số, phần trăm, thời gian đã chạy và ETA. Phân biệt rõ training với indexing/inference; không gọi một tác vụ là train nếu model không được cập nhật trọng số.
12. Từ Stage 4 trở đi, mọi nhánh visual retrieval phải dùng adapter đã fine-tune ở Stage 3; chỉ dùng Visualized-BGE zero-shot khi chạy ablation có ghi rõ. Stage 5 nhận citation từ retriever này qua Stage 4, không dùng retriever làm model sinh câu trả lời.

## Trạng thái stage

- `NOT_STARTED`: chưa triển khai.
- `IN_PROGRESS`: agent đang triển khai.
- `AWAITING_USER_TEST`: đã bàn giao, đang chờ người dùng tự kiểm tra.
- `COMPLETED`: người dùng đã xác nhận thành công.
- `BLOCKED`: không thể tiếp tục nếu thiếu dữ liệu hoặc quyết định từ người dùng.

## Nguyên tắc kỹ thuật

- Ưu tiên giải pháp nhỏ nhất có thể đo lường và tái lập; không thêm abstraction hoặc dependency để dùng trong tương lai.
- Đặt tên file, module và thư mục theo chức năng hoặc trách nhiệm nghiệp vụ, đủ cụ thể để không nhầm với thành phần ở stage sau (ví dụ `data.py`, `bm25_evaluation.py`, `bm25-retrieval-dev`); không dùng tên quá chung chung như `evaluation.py`, không đặt theo số stage hay chi tiết cấu hình như `stage2.py`, `stage2-bm25`, `k5`.
- CLI và đường dẫn ghi trong `README.md` phải khớp code hiện tại và chạy từ thư mục gốc project.
- Cài và chạy mọi package của project trong `.venv`; không cài dependency vào Python hệ thống. README ưu tiên command `.\.venv\Scripts\python.exe` để không phụ thuộc trạng thái kích hoạt shell.
- Mỗi thí nghiệm phải lưu cấu hình, prediction, metric và thông tin tài nguyên đủ để tái lập.
- Không dùng test/private split để chọn model, weight, threshold hoặc prompt.
- Citation do hệ thống trả về phải tồn tại trong corpus đã lập chỉ mục.
- Không mô tả hệ thống như công cụ tư vấn pháp lý production.

## Dữ liệu và bảo mật

- Không commit dataset VLSP, model weights, vector index, cache, file môi trường hoặc secrets.
- Tôn trọng data agreement và giấy phép của từng nguồn/model.
- Không phát hành lại dữ liệu VLSP trong repository này.

## Phạm vi stage

- Stage 0: khởi tạo Git, quy tắc làm việc, README và cấu trúc Python tối thiểu.
- Stage 1: thu thập, kiểm kê, loader, validator và train/dev split dữ liệu.
- Stage 2: evaluator F2/Accuracy và baseline retrieval.
- Stage 3: visual retrieval.
- Stage 4: hybrid retrieval và fusion giữa BM25 với visual retriever đã fine-tune.
- Stage 5: grounded legal QA trên citation do hybrid retriever đã fine-tune cung cấp.
- Stage 6: ablation, đo tài nguyên và phân tích lỗi.
- Stage 7: Streamlit, hoàn thiện CLI và tài liệu tái lập.
