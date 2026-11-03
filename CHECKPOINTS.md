# CHECKPOINTS — Phân tích kết quả YOLOE crop retrieval

> **Cách cập nhật CHECKPOINTS:** Đầu file chỉ giữ tóm tắt context đã hoàn thành. Phần dưới chỉ chứa bước chưa làm, có input/output và điều kiện đạt. Bỏ bước hoàn tất khỏi checklist, chuyển số liệu quan trọng vào tóm tắt/README; không tích lũy nhật ký. Phân biệt code đã triển khai với pipeline đã chạy. Không thêm file/path/kiến trúc ngoài phạm vi đã duyệt. Lệnh thực tế và trạng thái toàn dự án nằm trong README; quy tắc chung ở AGENTS.md.

## Context đã hoàn thành

- Giữ nguyên hướng caption-based: dev F2 corpus/example/hybrid 0.199681 / 0.477725 / 0.488460; public retrieval 0.604277, QA 0.76.
- Baseline full-image/text-only: dev F2@5 text corpus/example/hybrid 0.168180 / 0.377727 / 0.365551; ảnh corpus/example/hybrid 0.059457 / 0.325044 / 0.275478. Final full-image 0.40265941558682794, text/ảnh 0.75/0.25, chọn trên dev. Baseline/model/split được giữ nguyên.
- Đã triển khai 7 file CLI trong `src/hybrid_rag/crop_image_method/`; toàn bộ 8 bước crop, embedding, corpus/example retrieval, image hybrid và image/text fusion đã **COMPLETED**. Output nằm tại `artifacts/experiments/hybrid_rag/crop_image_method/` với tên folder đồng bộ full-image.
- Dữ liệu: 787 ảnh corpus → **772 crop/vector**, 373 ảnh không detection; 530 câu = 418 train + 112 dev / 298 ảnh query → **1.512 crop/vector**, 6 ảnh không detection. Có **409 train examples** có crop; dev không có câu bị rỗng crop. Registry gồm 402 record / 398 cặp citation duy nhất.
- Detector YOLOE-v8l-seg/MobileCLIP local, 15 prompt cố định, conf=0.25, imgsz=960, NMS không phân biệt prompt iou=0.7, padding 5% mỗi cạnh, PNG; không thay toàn ảnh khi detect rỗng. DINOv2 ViT-B/14 local, CLS L2 FP32 768 chiều, resize/pad 518×518, batch 4; giữ mapping từng crop và membership train/dev. Không thêm dependency hoặc tải model.
- Corpus: max cosine theo citation cho từng crop → top-10 riêng → RRF chuẩn hóa trọng số 1/m, k=60 → top-10 query. Example: trung bình hai chiều của max cosine giữa hai tập crop → top-5 train query → đọc nhãn train và gộp max về citation. Nhãn example thuộc toàn câu/ảnh, không coi là gold riêng từng biển.
- Dev F2@5 crop thực chạy: corpus **0.09144462382654213**, example **0.43014461411121807**, image hybrid **0.3787779929912646**, final text–crop **0.42515846825829445**. Corpus/example image hybrid 0.5/0.5, final text/ảnh 0.75/0.25 cố định, k=60; không tune trọng số mới trên dev.
- Đã tạo đủ `query_retrieval_example/`, `query_retrieval_hybrid/`, `image_text_retrieval/`; hai bước đầu có predictions/per_query_metrics/metrics/manifest, final thêm `final_citations.json`. Đã kiểm tra hash, evaluator và đủ 112 query. Final xuất đúng thứ tự `dev.json`, mỗi câu 5 citation hợp lệ không trùng.
- Trace giữ ranking riêng từng biển, candidate union, crop bằng chứng và trạng thái availability/fallback. Gold dev giữ nguyên. Final crop cao hơn full-image **0.022499052671466513** trên dev; chưa phải test độc lập. Coverage gold corpus giảm từ **132/254** trước detection xuống **65/254** sau detection, citation có ảnh/crop từ 228 xuống 100.
- Baseline final có source fingerprint từ phiên bản trước: chỉ dùng saved predictions/metrics đã kiểm tra input/output để so sánh, giữ hash gốc trong `baseline_reference`; input crop/text mới vẫn kiểm tra fingerprint nghiêm ngặt. Không sửa hash cũ hoặc ghi đè baseline.
- Lệnh chạy/tái lập ở mục Hướng crop-image trong README. `--overwrite` xóa/tạo lại đúng output bước chọn, không resume batch; thay prompt/input/code phải chạy lại các bước phụ thuộc. Không cần crop/embedding lại chỉ để xem các kết quả vừa hoàn tất.

## Việc chưa làm

### ☐ 1. Kiểm tra trực quan các lỗi detection và matching

- Dùng `corpus_cropping/`, `query_cropping_dev/`, annotated/crops và trace retrieval đã có. Kiểm tra biển nhỏ, ảnh phố nhiều biển, biển phụ, biển ghép, bbox lồng nhau và prompt khớp vật thể không phải biển báo.
- Xác định các citation/gold mất coverage vì ảnh corpus không detect được, so với lỗi truy nhầm crop/citation; không suy ra detection mAP khi chưa có bbox gold. Kiểm tra riêng biển đơn/corpus minh họa khác ảnh phố query.
- Điều kiện đạt: phân loại được các nhóm lỗi từ bằng chứng thực tế; chưa tự sửa prompt/model/mapping hoặc tạo bộ annotation/output phân tích mới.

### ☐ 2. Phân tích đóng góp corpus, example và fusion

- Đọc metrics/per-query metrics hiện có: corpus crop tăng so với full-image nhưng image hybrid **0.378778** thấp hơn example **0.430145**. Phân biệt citation đúng bị đẩy khỏi top-5 với false positive mới được thêm vào.
- Xem candidate-union recall, số crop được đại diện ở top-5, nhóm 0/1/nhiều crop và coverage. Crop-only chưa lọc biển theo câu hỏi, chưa mở rộng sang điều luật chung không chứa ảnh hoặc giải nghĩa kết hợp biển chính–phụ.
- Điều kiện đạt: giải thích được điểm tăng/giảm của từng nhánh trên dev; đề xuất thay đổi tiếp theo để người dùng duyệt trước khi sửa kiến trúc, trọng số, prompt hoặc data.

Caption/OCR, chọn biển theo câu hỏi, mask nền, fine-tune, mở rộng tham chiếu citation, tuning fusion mới, public/private và QA cho crop chưa triển khai. Giữ nguyên `.gitignore`, split, preview YOLOE và các baseline.
