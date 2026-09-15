# Các hướng cải tiến Visualized-BGE có ROI cao

Không có thay đổi nào có thể bảo đảm tăng điểm trước khi đánh giá thực nghiệm. Danh sách dưới đây chỉ giữ những hướng có cơ sở rõ từ code, kết quả hiện tại hoặc thực hành retrieval đã được công bố; các hướng tốn công và có độ bất định cao đã được loại bỏ.

## 1. Tối ưu `top_k`

Code đang cố định `top_k=5`, trong khi số citation đúng của mỗi câu hỏi không giống nhau và F2 ưu tiên recall hơn precision. Thử các giá trị `top_k` khác nhau trên dev có thể cải thiện trực tiếp F2 mà không cần huấn luyện lại model.

Có thể dùng một `top_k` cố định đã chọn trên dev hoặc trả về số citation động dựa trên ngưỡng similarity. Phương án cố định đơn giản và ít nguy cơ overfit hơn.

### Kết quả và quyết định

- Với candidate-level adapter: k1 đạt F2 `0.3627`, k3 `0.4479`, k5 `0.4649`, k7 `0.4602`, k10 `0.4264`.
- Sau khi chuyển sang citation-level loss: k3 đạt `0.4812`, k5 `0.4971`, k7 `0.4567`.
- `top_k=5` thắng ở cả hai lần và được khóa. Tăng k làm recall tăng nhưng precision giảm nhanh hơn mức F2 bù được.
- Không tiếp tục vét k4/k6 để tránh chọn quá sát trên 112 mẫu dev.

### Hướng mở rộng

Fixed top-k chưa tương đương phương pháp dynamic top-k của đội mạnh. Đội `chmod+x` dùng số lượng biển báo liên quan để điều chỉnh k theo độ phức tạp của từng query. Chỉ nên thử dynamic top-k sau khi có traffic-sign detector hoặc score calibration đáng tin cậy; không dùng một similarity threshold tùy ý.

## 2. Chọn lại cách biểu diễn query

Query hiện nối ảnh, câu hỏi và toàn bộ lựa chọn trả lời. Các lựa chọn sai có thể chứa từ khóa mạnh và kéo embedding về citation không liên quan.

Các biểu diễn đáng so sánh nhất là:

- Ảnh và câu hỏi.
- Ảnh, câu hỏi và các lựa chọn.
- Câu hỏi không kèm ảnh.

Có thể encode riêng `ảnh + câu hỏi` và `chỉ câu hỏi`, sau đó kết hợp similarity score. Cách này giữ tín hiệu của từng modality thay vì buộc model nén toàn bộ vào một biểu diễn duy nhất.

### Kết quả và quyết định

- `image-question-choices`: F2 `0.4649`, precision `0.2625`, recall `0.6148`.
- `image-question`: F2 `0.4230`, precision `0.2357`, recall `0.5661`.
- `question-only`: F2 `0.4109`, precision `0.2321`, recall `0.5468`.
- Cả ảnh và choices đều mang tín hiệu hữu ích; loại choices làm mất từ khóa pháp lý/mô tả biển báo, còn bỏ ảnh giảm thêm tín hiệu thị giác. Stage 3 giữ `image-question-choices`.

### Hướng mở rộng

Thí nghiệm trên mới thay input của một joint embedding, chưa phải kiến trúc nhiều nhánh. Hướng mạnh hơn là encode text query/choices, ảnh toàn cảnh và crop biển báo bằng các encoder riêng, rồi fusion score đã hiệu chỉnh. Berry dùng Jina Embeddings cho text, C-RADIOv2-B cho ảnh và OWLv2 cho object feature; TechNova dùng kiến trúc hai nhánh và gme-Qwen2-VL-2B-Instruct. Đây là thay đổi lớn, phù hợp Stage 6 hơn là mở rộng Stage 3.

## 3. Huấn luyện và chấm điểm ở cấp citation

Loss hiện tại tối ưu trên 1.204 candidate, nhưng metric và output được tính trên 398 citation. Một citation có thể có nhiều text chunk và image candidate, nên số lần xuất hiện của các citation trong loss không đồng đều.

Có thể gộp điểm candidate thành điểm citation bằng `max` hoặc pooling top candidates rồi tính contrastive loss trên citation. Cách này đồng bộ objective huấn luyện với cách inference và evaluator hoạt động.

Citation-level loss cũng tránh coi mọi chunk và mọi ảnh của một gold article là positive mạnh, dù chỉ một candidate thực sự chứa căn cứ liên quan.

### Kết quả và quyết định

- Candidate-level loss: F2 `0.4649`, precision `0.2625`, recall `0.6148`.
- Citation-level loss: F2 `0.4971`, precision `0.2768`, recall `0.6661`.
- F2 tăng tuyệt đối `0.0322`, tương đối khoảng `6.9%`; precision và recall cùng tăng. Đây là improvement duy nhất được chọn trong Stage 3.
- Adapter được khóa tại `artifacts/experiments/improvements/citation-level-loss/visual-bge-citation-loss/adapter.pt`.

### Hướng mở rộng

Nếu tài nguyên cho phép, có thể thử query tower và corpus tower riêng, projection phi tuyến nhỏ, hoặc fine-tune một phần backbone thay vì dùng chung một linear adapter. Mọi phương án phải giữ citation-level evaluation và so sánh cùng split/seed; không tăng độ phức tạp nếu public-test result không cho thấy nhu cầu.

## 4. Ghép ảnh luật với local text context

Image candidate hiện chỉ gồm ảnh và title của article. Có thể bổ sung phần văn bản gần marker ảnh, chẳng hạn caption, đoạn mô tả đứng trước hoặc sau ảnh, và nội dung hàng bảng liên quan.

Local context giúp model liên kết đặc trưng thị giác với tên biển báo và thuật ngữ pháp lý cụ thể. Thay đổi này tận dụng dữ liệu sẵn có, không cần thêm OCR, layout model hoặc dependency mới.

### Kết quả và quyết định

- Cửa sổ 256 token mỗi phía đạt F2 `0.4936`, precision `0.2768`, recall `0.6579`.
- Citation-level baseline không context đạt F2 `0.4971`, precision `0.2768`, recall `0.6661`.
- Context không tăng precision và làm recall giảm nhẹ. Cửa sổ theo vị trí có thể chứa bảng hoặc quy định không trực tiếp mô tả ảnh, làm loãng title và đặc trưng thị giác.
- Stage 3 giữ `image_context_tokens=0`; không sweep 128/512 chỉ để vét dev score.

### Hướng mở rộng

Các đội mạnh không chỉ ghép một cửa sổ token. Hướng đáng thử về sau là chuyển bảng sang Markdown có cấu trúc, trích caption/đoạn mô tả theo semantic boundary, crop ảnh corpus để chỉ giữ biển báo, hoặc dùng vision-language model sinh mô tả biển rồi index mô tả đó. Những hướng này phải là ablation riêng vì dùng model/preprocessing mới.

## 5. Hard-negative mining

Với mỗi train query, lấy các citation không thuộc gold nhưng được model hiện tại xếp hạng cao làm hard negatives. Đây là những trường hợp model thực sự dễ nhầm và thường cung cấp tín hiệu học hữu ích hơn negative quá dễ.

Nguồn hard negative phù hợp gồm:

- Top results sai của Visualized-BGE hoặc adapter hiện tại.
- Top results sai của BM25.
- Citation có nội dung hoặc hình ảnh gần giống gold citation.

Mọi gold citation phải được loại khỏi negative set để tránh false negative. FlagEmbedding cũng sử dụng và cung cấp công cụ cho cách fine-tune với mined hard negatives.

### Đánh giá hiện tại và hướng mở rộng

- Stage 3 chưa triển khai mining riêng. Citation-level loss hiện so mỗi query với toàn bộ 398 citation; `logsumexp` đã tự nhấn mạnh các negative có score cao, nên mining có thể trùng phần lớn tín hiệu hiện có.
- Chỉ nên thử khi chuyển sang mini-batch/end-to-end fine-tuning, khi không còn đưa toàn corpus vào mỗi bước train, hoặc khi error analysis chỉ ra một nhóm citation dễ nhầm ổn định.
- Nếu thực hiện: mine bằng model đã khóa, loại toàn bộ gold citation, lưu danh sách negative và model/score nguồn, rồi train một experiment mới. Có thể kết hợp top sai của visual, BM25 và reranker; không dùng private-test feedback để mine hoặc chọn negative.

## 6. Hybrid retrieval giữa Visualized-BGE và BM25

Kết quả hiện tại cho thấy Visualized-BGE và BM25 có các hit riêng, nên hai retriever cung cấp tín hiệu bổ sung cho nhau. Có thể kết hợp hai ranking bằng Reciprocal Rank Fusion hoặc score fusion đơn giản.

BM25 mạnh với từ khóa pháp lý chính xác; Visualized-BGE bổ sung tín hiệu hình ảnh và tương đồng ngữ nghĩa. Theo kiến trúc dự án, hướng này thuộc Stage 4 và nhánh visual phải sử dụng adapter đã fine-tune ở Stage 3.

### Kết quả hiện tại và bước tiếp theo

- BM25 dev đạt F2 `0.2069`; visual citation-level đạt `0.4971`.
- Phân tích trước đó ghi nhận visual có các hit riêng mà BM25 bỏ lỡ, nên fusion có cơ sở thực nghiệm thay vì chỉ là giả thuyết.
- Stage 4 nên bắt đầu bằng Reciprocal Rank Fusion vì không cần hiệu chỉnh hai thang score khác nhau. Chỉ thử score fusion/reranker sau khi RRF tạo baseline đo được.

## Đối chiếu với các đội dẫn đầu VLSP 2025

Điểm Stage 3 `0.4971` là dev F2 trên 112 mẫu tự chia, không thể so trực tiếp với private leaderboard. Challenge công bố top 1–5 lần lượt đạt `0.6455`, `0.6114`, `0.5992`, `0.5790`, `0.5432` trên 146 mẫu private; đội hạng 6 đạt `0.4512`. Về trị số, kết quả hiện tại nằm giữa hạng 5 và 6, nhưng khác split và protocol.

Khoảng cách chính không phải do dự án triển khai sai cùng một mẹo nhỏ, mà do pipeline của các đội mạnh có thêm các thành phần lớn:

- Traffic-sign detection/cropping: LifeIsTough dùng YOLOE và Gemma-3-12B; Berry dùng OWLv2; Tanka_CDS fine-tune YOLOv8n. Pipeline hiện tại encode toàn ảnh nên background và nhiều object có thể làm loãng tín hiệu.
- Encoder/fusion mạnh hơn: Berry tách text, ảnh và object feature; TechNova dùng two-branch retrieval với gme-Qwen2-VL-2B-Instruct. Pipeline hiện tại dùng một Visualized-BGE frozen backbone và một linear adapter chung cho query/corpus.
- Reranking/graph: `chmod+x` dùng heterogeneous graph, Jina Reranker và dynamic top-k. Stage 3 hiện là single-stage dense retrieval.
- Preprocessing có cấu trúc: các đội mạnh chuẩn hóa text, đổi HTML table sang Markdown, lọc/crop ảnh; local context hiện tại chỉ là cửa sổ token quanh marker.

Visualized-BGE upstream được multimodal-pretrain trên hơn 500.000 instances và cung cấp downstream fine-tuning; Stage 3 chỉ fine-tune khoảng 1,05 triệu tham số adapter trên 413 mẫu hợp lệ để phù hợp GPU 16 GB. Vì vậy điểm hiện tại hợp lý với phạm vi và tài nguyên, nhưng không đại diện cho trần của end-to-end multimodal retrieval.

### Roadmap nếu cần tiến gần nhóm dẫn đầu

1. Stage 4: RRF giữa BM25 và citation-level visual adapter.
2. Khóa cấu hình bằng dev, đánh giá public test đúng một lần; chỉ so leaderboard sau khi nộp private Codabench.
3. Stage 6 ablation: detector/crop biển báo query và corpus — ưu tiên cao nhất.
4. Thử separate text/image/object branches và calibrated score fusion.
5. Rerank top-N citation bằng cross-encoder hoặc multimodal reranker.
6. Sau cùng mới cân nhắc dynamic top-k, graph retrieval hoặc partial/end-to-end fine-tuning.

Không dùng điểm private để tiếp tục chọn detector, model, fusion weight, threshold hay prompt. Nếu train final submission sau khi đã khóa cấu hình, có thể dùng lại toàn bộ 530 mẫu training theo protocol được ghi rõ, nhưng phải giữ public/private làm holdout.

## Tài liệu tham khảo

- [Visualized-BGE model card](https://huggingface.co/BAAI/bge-visualized)
- [Visualized-BGE: A Universal Multi-Modal Embedding Model](https://arxiv.org/abs/2406.04292)
- [BGE M3-Embedding](https://arxiv.org/abs/2402.03216)
- [FlagEmbedding hard-negative mining guide](https://github.com/FlagOpen/FlagEmbedding/blob/master/scripts/README.md)
- [VLSP 2025 MLQA-TSR challenge paper và phương pháp các đội](https://aclanthology.org/anthology-files/pdf/vlsp/2025.vlsp-1.48.pdf)
