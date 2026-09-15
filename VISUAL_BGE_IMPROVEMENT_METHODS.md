# Các hướng cải tiến Visualized-BGE có ROI cao

Không có thay đổi nào có thể bảo đảm tăng điểm trước khi đánh giá thực nghiệm. Danh sách dưới đây chỉ giữ những hướng có cơ sở rõ từ code, kết quả hiện tại hoặc thực hành retrieval đã được công bố; các hướng tốn công và có độ bất định cao đã được loại bỏ.

## 1. Tối ưu `top_k`

Code đang cố định `top_k=5`, trong khi số citation đúng của mỗi câu hỏi không giống nhau và F2 ưu tiên recall hơn precision. Thử các giá trị `top_k` khác nhau trên dev có thể cải thiện trực tiếp F2 mà không cần huấn luyện lại model.

Có thể dùng một `top_k` cố định đã chọn trên dev hoặc trả về số citation động dựa trên ngưỡng similarity. Phương án cố định đơn giản và ít nguy cơ overfit hơn.

## 2. Chọn lại cách biểu diễn query

Query hiện nối ảnh, câu hỏi và toàn bộ lựa chọn trả lời. Các lựa chọn sai có thể chứa từ khóa mạnh và kéo embedding về citation không liên quan.

Các biểu diễn đáng so sánh nhất là:

- Ảnh và câu hỏi.
- Ảnh, câu hỏi và các lựa chọn.
- Câu hỏi không kèm ảnh.

Có thể encode riêng `ảnh + câu hỏi` và `chỉ câu hỏi`, sau đó kết hợp similarity score. Cách này giữ tín hiệu của từng modality thay vì buộc model nén toàn bộ vào một biểu diễn duy nhất.

## 3. Huấn luyện và chấm điểm ở cấp citation

Loss hiện tại tối ưu trên 1.204 candidate, nhưng metric và output được tính trên 398 citation. Một citation có thể có nhiều text chunk và image candidate, nên số lần xuất hiện của các citation trong loss không đồng đều.

Có thể gộp điểm candidate thành điểm citation bằng `max` hoặc pooling top candidates rồi tính contrastive loss trên citation. Cách này đồng bộ objective huấn luyện với cách inference và evaluator hoạt động.

Citation-level loss cũng tránh coi mọi chunk và mọi ảnh của một gold article là positive mạnh, dù chỉ một candidate thực sự chứa căn cứ liên quan.

## 4. Ghép ảnh luật với local text context

Image candidate hiện chỉ gồm ảnh và title của article. Có thể bổ sung phần văn bản gần marker ảnh, chẳng hạn caption, đoạn mô tả đứng trước hoặc sau ảnh, và nội dung hàng bảng liên quan.

Local context giúp model liên kết đặc trưng thị giác với tên biển báo và thuật ngữ pháp lý cụ thể. Thay đổi này tận dụng dữ liệu sẵn có, không cần thêm OCR, layout model hoặc dependency mới.

## 5. Hard-negative mining

Với mỗi train query, lấy các citation không thuộc gold nhưng được model hiện tại xếp hạng cao làm hard negatives. Đây là những trường hợp model thực sự dễ nhầm và thường cung cấp tín hiệu học hữu ích hơn negative quá dễ.

Nguồn hard negative phù hợp gồm:

- Top results sai của Visualized-BGE hoặc adapter hiện tại.
- Top results sai của BM25.
- Citation có nội dung hoặc hình ảnh gần giống gold citation.

Mọi gold citation phải được loại khỏi negative set để tránh false negative. FlagEmbedding cũng sử dụng và cung cấp công cụ cho cách fine-tune với mined hard negatives.

## 6. Hybrid retrieval giữa Visualized-BGE và BM25

Kết quả hiện tại cho thấy Visualized-BGE và BM25 có các hit riêng, nên hai retriever cung cấp tín hiệu bổ sung cho nhau. Có thể kết hợp hai ranking bằng Reciprocal Rank Fusion hoặc score fusion đơn giản.

BM25 mạnh với từ khóa pháp lý chính xác; Visualized-BGE bổ sung tín hiệu hình ảnh và tương đồng ngữ nghĩa. Theo kiến trúc dự án, hướng này thuộc Stage 4 và nhánh visual phải sử dụng adapter đã fine-tune ở Stage 3.

## Tài liệu tham khảo

- [Visualized-BGE model card](https://huggingface.co/BAAI/bge-visualized)
- [Visualized-BGE: A Universal Multi-Modal Embedding Model](https://arxiv.org/abs/2406.04292)
- [BGE M3-Embedding](https://arxiv.org/abs/2402.03216)
- [FlagEmbedding hard-negative mining guide](https://github.com/FlagOpen/FlagEmbedding/blob/master/scripts/README.md)

