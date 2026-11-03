# CHECKPOINTS — Fusion từ ranking text và ảnh đã lưu

> **Cách cập nhật CHECKPOINTS:** Đầu file chỉ giữ tóm tắt context đã hoàn thành. Phần dưới chỉ chứa bước chưa làm, theo thứ tự triển khai, có file cần sửa, input/output và điều kiện đạt. Khi bước hoàn tất, bỏ bước đó khỏi checklist, chuyển kết quả quan trọng vào tóm tắt/README. Không tích lũy lịch sử thao tác. Phân biệt rõ thiết kế chưa triển khai với lệnh/code đã chạy. Không tự thêm file kết quả, thư mục hoặc kiến trúc ngoài phạm vi đã duyệt.

## Context đã hoàn thành

- Hướng `text_based_rag` caption-based được giữ riêng: dev F2 corpus/example/hybrid 0.199681 / 0.477725 / 0.488460; public retrieval 0.604277, QA 0.76.
- Hybrid dùng text thuần BGE-M3 và ảnh toàn khung DINOv2, không caption hoặc YOLOE. Model/corpus index giữ nguyên; 418 train làm bank, 112 dev để chấm. Hai query bank và bốn corpus/example retrieval đã COMPLETED.
- Dev F2@5: text corpus 0.168180, text example 0.377727; ảnh corpus 0.059457, ảnh example 0.325044. Đây là điểm đọc từ kết quả thực tế hiện có.
- Đã tách `full_image_method/query_image_hybrid.py`: ảnh corpus + ảnh example → RRF 0.5/0.5, k=60, top-10. Đã chạy: F2@5 **0.2754776804218995**, 112 query; lưu `predictions.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `manifest.json` tại **`full_image_method/query_retrieval_hybrid/`**.
- Đã tách và chạy `text_only/query_text_hybrid.py`: text corpus + text example → RRF 0.5/0.5, k=60, top-10; F2@5 **0.36555129205447956**, 112 query. Lưu cùng bốn loại file tại **`text_only/query_retrieval_hybrid/`**.
- Đã kiểm tra cả hai ranking đã lưu giống hoàn toàn ranking tương ứng bên trong kết quả fusion phiên bản trước trên cả 112 query. Có fallback nếu example rỗng, hash input/output và metric kiểm chứng bằng evaluator; loader của bước cuối đọc được cả hai output.
- Bước cuối `full_image_method/hybrid_retrieval.py` chỉ đọc hai ranking text/ảnh đã lưu; khảo sát trọng số text/ảnh và top-5 cuối giữ nguyên. Output vẫn là **`full_image_method/image_text_retrieval/`**. Chưa ghi đè kết quả cuối sau lần tách bước này.

## Việc chưa làm

### ☐ 1. Chạy lại bước text–ảnh cuối

- Dùng lệnh cuối trong mục Hướng hybrid của README: `hybrid_retrieval.py --split dev --tune-on-dev --rrf-k 60 --final-top-n 5 --overwrite`.
- Input là `text_only/query_retrieval_hybrid/` và `full_image_method/query_retrieval_hybrid/`; bước cuối không gộp lại corpus/example. Cấu hình k của cả hai ranking phải khớp k của bước cuối; nếu đổi k thì chạy lại hai bước gộp trước.
- Giữ nguyên các file final/predictions/per_query_metrics/metrics/manifest trong `image_text_retrieval/`; không chạy lại model, query embedding hoặc bốn retrieval khi input không đổi.

### ☐ 2. Kiểm tra final và cập nhật tiến độ

- Manifest COMPLETED, hash khớp; final đúng 112 câu theo thứ tự dev, mỗi câu 5 citation không trùng. Predictions phải chứa hai ranking text/ảnh đã lưu, fallback và đóng góp từng nhánh.
- Metrics có corpus/example riêng, hai ranking sau gộp, điểm trọng số 0.5 ban đầu và bảng khảo sát trọng số text 0/0.25/0.5/0.75/1. Nếu chọn text weight=1 thì ghi rõ ảnh bị tắt; điểm chọn trên dev không phải test độc lập.
- Đọc số liệu thực tế cập nhật README rồi bỏ checkpoint đã hoàn tất khỏi đây; giữ riêng hướng caption-based. Chưa triển khai YOLOE crop, public/private hoặc QA cho hybrid này.
