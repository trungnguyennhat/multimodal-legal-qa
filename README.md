# Multimodal Legal RAG

Project nghiên cứu VLSP 2025 MLQA-TSR, không phải công cụ tư vấn pháp lý production. Quy tắc làm việc: [AGENTS.md](AGENTS.md).

## Trạng thái

| Thành phần | Trạng thái |
| --- | --- |
| Stage 0–1: khởi tạo, dữ liệu và split | `COMPLETED` |
| Stage 2–5 cũ: đã gỡ retrieval/adapter/QA | `AWAITING_USER_TEST` |
| Stage 6–7: ablation, demo và đóng gói | `NOT_STARTED` |
| Chuẩn hóa, chunk và index corpus | `AWAITING_USER_TEST` |
| Corpus chunking parent–child theo item ảnh, ghi đè output cũ | `AWAITING_USER_TEST` |
| Corpus embedding từ child chunks mới, hỗ trợ ghi đè index | `AWAITING_USER_TEST` |
| Query embedding strict-relevant train/dev, hỗ trợ ghi đè | `AWAITING_USER_TEST` |
| Caption corpus và query train; embedding query train | `COMPLETED` |
| Caption, embedding dev và dense corpus retrieval dev | `AWAITING_USER_TEST` |
| Example retrieval dev từ train | `AWAITING_USER_TEST` |
| Hybrid corpus + example retrieval dev | `AWAITING_USER_TEST` |
| Tổ chức thư mục experiments thành corpus/query | `AWAITING_USER_TEST` |
| Text-based retrieval public test | `AWAITING_USER_TEST` |
| Prompt corpus ảnh + điều luật, JSON phân loại hình với Qwen3.5: thử image002 | `COMPLETED` |
| Chạy lại toàn bộ caption corpus, ghi đè corpus_image_captioning | `COMPLETED` |
| Prompt query ảnh + câu hỏi, thử train_1_1 với Qwen3.5 | `AWAITING_USER_TEST` |
| Chạy lại caption query train/dev bằng prompt đã duyệt | `COMPLETED` |

Người dùng đã yêu cầu chấp nhận caption hiện có của corpus, train và dev, kể cả caption bị cắt. Đã cập nhật 27 caption corpus, 9 caption train và 2 caption dev từ `INCOMPLETE` sang `COMPLETED`, giữ nguyên description, finish_reason, lỗi validation nếu có; ghi `original_status=INCOMPLETE` và `accepted_by_user=true`. Manifest tương ứng hiện ghi đủ 787/787 ảnh corpus, 418/418 query train, 112/112 query dev và danh sách caption bị cắt đã chấp nhận; các JSONL tổng hợp đã có được đồng bộ. Đây là người dùng chấp nhận nội dung bị cắt, không phải model đã sinh đủ hoặc JSON bên trong đã được sửa. Không chạy lại inference.

Kiểm tra trạng thái đã chấp nhận trong PowerShell từ root:

```powershell
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_image_captioning\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\query\query_image_captioning_train\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\query\query_image_captioning_dev\manifest.json
```

Caption corpus hiện sẵn sàng cho chunker đọc và ghép theo image_id; chưa xuất lại corpus tổng hợp hoặc chạy chunking/embedding trong lượt chấp nhận này. Caption bị cắt có thể là JSON chưa đóng và thiếu một phần mô tả; code chunking hiện coi description là chuỗi nên không yêu cầu parse JSON. Nếu chuyển sang chunk theo item JSON thì phải xử lý riêng các caption đã chấp nhận bị cắt này, không tự bỏ nội dung hoặc sửa nghĩa.

## Tổ chức thư mục corpus/query — AWAITING_USER_TEST

Đã chuyển vị trí 11 thư mục hiện có dưới `artifacts/experiments/text_based_rag/`, không chạy lại pipeline, test, model, evaluator hoặc tạo prediction/embedding/caption mới. Đã đối chiếu SHA256 của toàn bộ 1.363 file trước/sau di chuyển, giữ nguyên nội dung, metrics và manifest.

```text
artifacts/experiments/text_based_rag/
├── corpus/
│   ├── corpus_chunking/
│   ├── corpus_embedding/
│   ├── corpus_image_captioning/
│   └── data_preprocessing/
└── query/
    ├── query_embedding_dev/
    ├── query_embedding_train/
    ├── query_image_captioning_dev/
    ├── query_image_captioning_train/
    ├── query_retrieval_corpus/
    ├── query_retrieval_example/
    └── query_retrieval_hybrid/
```

Đã cập nhật đường dẫn input/output trong cả 7 module `src/text_based_rag/`: corpus_normalization, corpus_image_captioning, corpus_chunking, corpus_embedding, query_image_captioning, query_embedding, query_retrieval. Thư mục review khi chạy sau này cũng theo nhóm corpus/query. Tên module và CLI giữ nguyên; public-test ở ngoài experiments giữ nguyên vị trí. Lệnh và link output trong README đã cập nhật.

Manifest/metadata đã sinh không bị viết lại: các đường dẫn tuyệt đối cũ là thông tin lịch sử, các key input_sha256 như corpus_embedding/chunks.jsonl vẫn là định danh artifact. Retrieval ánh xạ các key đó sang nhóm corpus/query mới để kiểm tra hash; caption_file tương đối vẫn hợp lệ trong thư mục đã chuyển. Việc di chuyển không đòi hỏi tạo lại caption, chunk hoặc vector. Hash source_sha256 cũ phản ánh code lúc tạo artifact, không được thay bằng hash code mới.

Điều kiện kiểm tra: đứng ở root project; không cần GPU/model. PowerShell chỉ đọc vị trí và metric hiện có:

```powershell
Get-ChildItem .\artifacts\experiments\text_based_rag\corpus -Directory
Get-ChildItem .\artifacts\experiments\text_based_rag\query -Directory
Get-Content .\artifacts\experiments\text_based_rag\query\query_retrieval_hybrid\metrics.json
```

Mong đợi 4 thư mục corpus, 7 thư mục query và F2 hybrid vẫn 0,4918943283. Nếu command cá nhân còn dùng đường dẫn cũ, thêm corpus/ hoặc query/ theo cây trên; không cần chạy lại stage để sửa vị trí. Lệnh pipeline cho những lần chạy sau nằm ở từng mục phía dưới, chỉ chạy khi muốn tái tạo kết quả. Chưa chạy pipeline để kiểm tra runtime sau di chuyển theo yêu cầu người dùng.

## Hybrid corpus + example retrieval dev — AWAITING_USER_TEST

Đã sửa `src/text_based_rag/query_retrieval.py` và tự chạy theo yêu cầu người dùng. Không thêm file Python/dependency. `--method hybrid` mặc định dùng `--hybrid-strategy gated`; `--hybrid-strategy fixed` vẫn tái lập bản fusion min–max/trọng số cố định trước đây. Kết quả mới đã ghi đè đúng 5 file trong `artifacts/experiments/text_based_rag/query/query_retrieval_hybrid/`: `retrieval_results.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `retrieval_preview.txt`, `manifest.json`. Corpus/example nguồn, split, caption và embedding không thay đổi; không chạy lại encoder hoặc dùng GPU.

Thiết kế gated đã triển khai:

| Gate từ max cosine query–train | Candidate | Example / Corpus | Cutoff |
| --- | --- | --- | --- |
| High: s_max >= 0,82 | Citation hợp lệ của top 3 examples; corpus không được thêm candidate | 0,90 / 0,10 | Giữ toàn bộ union top 3, không ép còn 5 |
| Middle: 0,70 <= s_max < 0,82 | Union citation từ top 5 examples và top 5 corpus đã lưu | 0,60 / 0,40 | Top 5 citation |
| Low: s_max < 0,70 | Union như middle | 0,30 / 0,70 | Top 5 citation |

Điểm example = max cosine của ví dụ hỗ trợ citation. Tính lại max child cosine corpus cho mọi candidate bằng vector hiện có, kể cả citation ngoài corpus top 5; không coi chúng là điểm 0. Chuẩn hóa cố định `(cosine+1)/2`, nhánh example không hỗ trợ đóng góp 0. Corpus chỉ rerank trong pool high đã khóa; trọng số 0,90/0,10 là reranking nhẹ, không phải chỉ phá hòa tuyệt đối. Vì high trả nguyên pool nên reranking corpus không làm đổi F2 của nhóm này. `sources=corpus` trong gated nghĩa là có điểm/evidence corpus được tính cho citation, không nhất thiết citation đã nằm trong top 5 corpus nguồn. Hòa điểm theo `(law_id,article_id)`. Giữ ảnh trùng train/dev theo yêu cầu. Cosine không phải xác suất đúng/OOD.

Điều kiện: hai nguồn retrieval COMPLETED, hash/result/query ID/text/encoder khớp; corpus index và vector/mapping dev hiện tại còn khớp nguồn. Có validation float32, shape, finite, L2, vector_row và citation ID. Nếu mismatch thì kiểm tra hoặc chạy lại nhánh nguồn được báo lỗi; không bỏ validation. Nếu output tồn tại phải dùng --overwrite. Mỗi tác vụ còn chạy có log khoảng 30 giây với bước, processed/total, %, elapsed/ETA. Đây là inference/evaluation CPU, không training.

Chạy lại từ root project:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --method hybrid --hybrid-strategy gated --split dev --example-weight 0.6 --final-top-n 5 --overwrite
if ($LASTEXITCODE -eq 0) {
    Get-Content .\artifacts\experiments\text_based_rag\query\query_retrieval_hybrid\metrics.json
}
```

`--example-weight` chỉ điều chỉnh gate middle; corpus weight middle = 1 - example weight. `--final-top-n` chỉ áp dụng middle/low. High/low thresholds và high/low weights giữ cấu hình đã duyệt, không thêm CLI chỉ để dùng trong tương lai. Muốn tái lập fixed baseline dùng cùng lệnh nhưng thay `--hybrid-strategy gated` thành `--hybrid-strategy fixed`; lệnh đó cũng ghi đè thư mục hybrid, cần chạy gated sau nếu muốn giữ kết quả gated.

Mong đợi 112 prediction, manifest COMPLETED; 97 high, 15 middle, 0 low với dữ liệu hiện tại. Kết quả lưu gate/s_max/trọng số từng query, điểm gốc/chuẩn hóa, source query IDs (high chỉ từ top 3), evidence chunk ID. Metrics có breakdown theo gate và baseline F2. Low không có mẫu dev, chưa kiểm chứng thực nghiệm. Candidate ngoài union đã lưu không được thêm mới ở middle/low dù scoring có tính toàn index.

Quá trình chọn cấu hình: thử trong bộ nhớ trước khi sửa source/output; giữ nguyên ngưỡng đề xuất 0,70/0,82, high top3/0,90, middle 0,60, low 0,30, cutoff 5 cho middle/low. Ba phép đối chiếu được thực hiện: example-only, gate-only (high top3, các nhóm khác giữ example union), gated fusion. Cấu hình đầu tiên đã vượt strongest baseline nên không chạy grid search hoặc chọn threshold theo query/gold. Sau đó mới sửa code, chạy fixed regression (F2 vẫn 0,4272186827), chạy gated pipeline và xác minh bằng chính hàm eval_retrieval trong baseline chính thức. Không dùng public/private test. Đây vẫn là development result trên dev đã được xem/phân tích, không phải đánh giá độc lập.

| Cấu hình | Macro precision | Macro recall | Macro F2 |
| --- | --- | --- | --- |
| Corpus-only | 0,135714 | 0,327912 | 0,245940 |
| Example-only top5 union | 0,254358 | 0,677870 | 0,477725 |
| Fixed hybrid 0,6/0,4, top5 | 0,250000 | 0,557249 | 0,427219 |
| Gate-only, không corpus reranking | 0,297077 | 0,636650 | 0,490423 |
| Gated hybrid mới | 0,303231 | 0,627721 | 0,491894 |

Official evaluator F2 gated = 0,49189432803341615; example = 0,47772480020099534; corpus = 0,24594014908550882. Sai khác epsilon với metric TP-form local dưới 1e-9. Gated tăng khoảng 0,01417 F2 (1,42 điểm phần trăm, ~2,97% tương đối) so với example. Phần lớn cải thiện đến từ candidate top3 ở high; phần thêm từ corpus ở middle so với gate-only là ~0,00147 F2. Recall giảm từ 0,677870 xuống 0,627721, hit rate giảm từ 0,839286 xuống 0,8125; không tuyên bố mọi chỉ số đều tăng hoặc đã tái lập mức tăng trong paper. Kết quả giữ AWAITING_USER_TEST để người dùng xem/xác nhận.

## Example retrieval dev từ train — AWAITING_USER_TEST

Đã sửa duy nhất file Python hiện có `src/text_based_rag/query_retrieval.py`, giữ nguyên tên file. CLI có `--method corpus|examples`, mặc định corpus. Không chạy retrieval/test. Theo yêu cầu bổ sung, đã đổi tên thư mục kết quả corpus dev hiện có từ query_retrieval/ thành query_retrieval_corpus/, giữ nguyên nội dung các file.

- Corpus dev chạy lần sau sẽ ghi vào `artifacts/experiments/text_based_rag/query/query_retrieval_corpus/`.
- Example dev ghi vào `artifacts/experiments/text_based_rag/query/query_retrieval_example/`.
- Corpus public test ghi vào `artifacts/public-test/text_based_rag/query_retrieval_corpus/`; nguồn query embedding public vẫn ở thư mục cũ.
- Thư mục kết quả corpus dev đã được đổi tên từ `query_retrieval/` thành `query_retrieval_corpus/`; nội dung giữ nguyên. `--overwrite` chỉ thay các file thuộc chế độ đang chạy ở thư mục mới.

Điều kiện: embedding dev/train hoàn tất, cùng BGE-M3/model weights/representation strict-relevant/L2; split và mapping/hash còn khớp. Corpus index hoàn tất để kiểm tra danh mục citation ID. Code vẫn từ chối train/dev trùng query ID. Theo yêu cầu người dùng, giữ toàn bộ train examples, kể cả ảnh trùng source image ID hoặc SHA256 với dev; không đổi split hoặc embedding. Khi hash/encoder/split không khớp, kiểm tra lại nguồn embedding được báo lỗi; không bỏ validation.

Cosine exact CPU giữa vector dev và train, chọn đúng 5 query train gần nhất trên toàn bộ train, kể cả ảnh trùng, hòa điểm theo query ID. Sau ranking mới đọc gold train để gộp citation, dedup theo `(law_id,article_id)`, không cắt union còn 5. Score citation là cosine cao nhất trong các ví dụ đã chọn hỗ trợ citation, kèm `source_query_ids`. Không dùng vector/text corpus để tính điểm, không fallback corpus hoặc fusion. Dev gold chỉ được đọc để đánh giá sau prediction; mẫu số predicted là số citation unique thực tế.

Theo quy tắc citation phải tồn tại trong corpus đã index, code chỉ đọc mapping corpus làm danh mục ID. Các gold train không có trong danh mục bị loại khỏi predicted citations, giữ nguyên trong gold của ví dụ được chọn để truy vết; danh sách được báo trong `excluded_train_citations` ở metrics/manifest và log. Không tự sửa nhãn. Thống kê trước triển khai có 15/117 train citation ID không hợp lệ; việc lọc này có thể giảm coverage. Gold dev ngoài nguồn vẫn tính vào mẫu số recall, không bỏ query. Đây là biến thể BGE-M3 của example retrieval trong [paper, mục 3.2.2](https://aclanthology.org/2025.vlsp-1.51.pdf), không tái lập encoder/weight gốc.

Theo yêu cầu mới nhất, đã bỏ lọc ảnh trùng trước top 5. Kiểm tra read-only trước đó thấy 5 query dev dùng ảnh trùng nội dung với 2 query train; các ví dụ này hiện được phép tham gia retrieval. Manifest ghi same_image_policy=allow_shared_images_as_requested. Metrics phản ánh split hiện tại có ảnh dùng chung giữa train/dev. Trạng thái vẫn AWAITING_USER_TEST; chưa chạy lại pipeline. Chỉ đọc metrics sau khi retrieval thành công.

Chạy từ root project:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --method examples --split dev --overwrite
if ($LASTEXITCODE -eq 0) {
    Get-Content .\artifacts\experiments\text_based_rag\query\query_retrieval_example\metrics.json
}
```

Mong đợi 112 query dev, mỗi query có 5 ví dụ trong `examples`, số citation trong `results` thay đổi theo union hợp lệ, manifest `COMPLETED`. Giữ 5 file output: `retrieval_results.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `retrieval_preview.txt`, `manifest.json`. Metrics gồm macro/micro precision/recall/F1/F2, hit_rate và `top_k_examples=5`; không gọi là F2@5 citations. Log tiến trình mỗi khoảng 30 giây nếu tác vụ còn chạy; đây là inference/evaluation CPU, không training. Nếu output tồn tại, dùng `--overwrite` để chạy lại nhánh đó.

Lệnh corpus dev cho lần chạy sau (không cần chạy lại lúc này):

```powershell
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --method corpus --split dev --overwrite
```

## Thiết kế cosine retrieval và đánh giá sau embedding mới — chưa triển khai

Phạm vi thực hiện đã được người dùng thu hẹp: chỉ dev, giữ nguyên schema/file output cũ, chưa làm train retrieval. Luồng cosine/max child score/top 5 hiện có dùng được với embedding mới; chỉ sửa `query_retrieval.py` để `submission_task1.json` chỉ được xuất cho public_test, không tạo thêm file khi chạy dev. Không triển khai metadata evidence/metric mới hoặc đề xuất train bên dưới. Agent chưa chạy retrieval/metric trong lượt bàn giao này. Để tự ghi đè 5 file dev hiện có:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split dev --overwrite
Get-Content .\artifacts\experiments\text_based_rag\query\query_retrieval_corpus\metrics.json
```

Điều kiện: corpus_embedding/query_embedding_dev manifest COMPLETED, vector/mapping/hash và encoder khớp, dev split không đổi. Kết quả tại `artifacts/experiments/text_based_rag/query/query_retrieval_corpus/`: ghi đè đúng `retrieval_results.jsonl`, `per_query_metrics.jsonl`, `metrics.json`, `retrieval_preview.txt`, `manifest.json`, giữ nguyên các trường hiện có. Mong đợi 112 query với 5 citation khác nhau/query, metric mới theo vector mới và manifest COMPLETED; không kỳ vọng metric bằng baseline cũ. Nếu hash/encoder mismatch thì kiểm tra artifact embedding, không bỏ validation; nếu output tồn tại phải dùng --overwrite. Đây là CPU retrieval/evaluation, không training. Trạng thái bàn giao AWAITING_USER_TEST; không đọc/run query train.

Người dùng xác nhận đã chạy xong embedding; đã đọc manifest để đối chiếu: corpus `COMPLETED`, 5.459 child/402 parent, input dài nhất 503 token; dev `COMPLETED`, 112 query, `question_relevant_items_v1`, input max 540, 3 question_only, 2 recovered; train `COMPLETED`, 418 query, cùng representation, input max 2.719, 16 question_only, 9 recovered. Cả ba cùng BGE-M3, 1.024 chiều, L2. Chỉ đọc manifest/code; chưa chạy cosine, evaluator, sửa retrieval hoặc ghi đè prediction.

Giữ nguyên toán học: ma trận score = query_vectors @ corpus_child_vectors.T vì cả hai vector được chuẩn hóa L2. Mọi child luật/bảng/item/fragment cùng dùng trong một phép tìm kiếm. Với mỗi citation `(document_id,id)`, lấy max score trong các child; giữ child có điểm cao nhất làm evidence, top 5 citation khác nhau. Không lấy top 5 child trước rồi dedup. Hòa điểm: citation lexicographic, child cùng điểm lấy hàng vector đầu tiên. Parent_id dùng để lookup parent, citation vẫn lấy ID thật của corpus, không dùng mã biển do Qwen sinh. Không cần embedding parent, train retriever hoặc thêm dependency/vector database. Ma trận float32 dev khoảng 2,3 MiB, train 8,7 MiB, nên giữ phép tính exact NumPy CPU hiện có, chưa cần batching/ANN.

Thay đổi code tối thiểu dự kiến trong `query_retrieval.py`:

1. Giữ validation status/hash vector/mapping, shape, dtype, finite, L2, unique ID, vector_row, model/revision/weights/pooling/input_prefix khớp. Bổ sung check representation query đúng strict-relevant, số lượng khớp manifest, hash chunks của corpus index khớp chunking hiện tại, split/caption nguồn khớp query embedding để tránh dùng vector cũ. Không đánh đồng việc encoder cùng tên với việc các artifact có cùng nguồn.
2. Đọc/check hash `organized_corpus.jsonl`, kiểm tra child.parent_id/article_key và `(document_id,id)` khớp parent; không chỉ kiểm tra caption có metadata parent_id. Không tạo hoặc sửa citation bằng heuristic. Không hard-filter corpus theo image_category/chunk_type và không thêm score weight/threshold 0,90 trong lần đánh giá đầu tiên.
3. Prediction giữ các key đang dùng và thêm evidence metadata của child thắng: parent_id, chunk_type, image_id/image_category, item_index, part_index, source_block_ids và provenance caption bị cắt nếu có. Query result giữ representation_mode/fallback/selected_item_count để đọc lỗi. Không copy toàn parent dài vào mỗi prediction; parent lookup qua parent_id và file parent đã hash. Không đưa nhãn gold vào bước ranking.
4. Đánh giá chính trên dev 112 query. Nếu chạy train 418 query thì thêm --split train, đọc đúng query_embedding_train/train.json, báo đây là diagnostic; train query không trở thành retrieval corpus hoặc dữ liệu cập nhật model. Thiết kế cũ từng giữ output dev `query_retrieval/`; hiện output corpus đã đổi thành `query_retrieval_corpus/`. Đề xuất train riêng `query_retrieval_train/`, không để hai split ghi đè nhau. Public-test không chạy hoặc tinh chỉnh trong bước này; caption/vector public cũ phải được nâng schema riêng nếu muốn dùng representation mới.
5. Primary metric là macro F2@5 theo đúng công thức eval_retrieval chính thức: mean(5*precision*recall/(4*precision+recall+1e-9)). Có thể giữ macro/micro P/R/F1/F2 conventional như diagnostic, nhưng phân biệt official F2 với biểu thức TP tương đương chỉ khác epsilon. Báo HitRate@5 đúng tên, không gọi là QA Accuracy. Gold citation thiếu trong corpus vẫn giữ mẫu số recall và được báo riêng; không tự sửa ID/nhãn hoặc loại query.
6. Giữ prediction JSONL, per-query metrics, metrics JSON, preview, submission_task1.json và manifest với hashes/config/runtime/resources. Per-query metric có TP/FP/FN và representation/fallback; metric chính vẫn trên toàn dev, không loại 3 question_only hoặc 2 recovered. Có thể báo thêm count/F2 cho question_only và recovered để quan sát, không dùng nhóm nhỏ này chọn threshold/model. Output ghi đè chỉ qua --overwrite và sau kiểm tra dữ liệu; không sinh self-check/demo script.

Các phần không cần đổi: `corpus_embedding.py`, `query_embedding.py`, encoder, cách chuẩn hóa, số chiều, cosine và max child aggregation. Chưa thêm reranker, fusion, category filtering, multi-vector query hoặc chọn k theo nhãn; giữ top_k=5 để so với baseline. Index parent–child đã được code max_chunk_score hiện tại hỗ trợ ở mức scoring, nên không viết thêm retriever riêng.

Đánh giá lần đầu đo hiệu quả của pipeline mới tổng thể (caption, chunking, query filtering cùng thay đổi), không kết luận riêng item chunking hay is_relevant gây ra mức tăng/giảm so với dev cũ F2≈0,1269. Nếu cần phân lập nguyên nhân thì thí nghiệm riêng phải giữ các thành phần còn lại cố định và được yêu cầu sau. Không dùng public/private test để chọn cấu hình. Đây là thiết kế, chưa triển khai hoặc chạy đánh giá; chờ người dùng duyệt bước sửa retrieval.

## Query embedding chỉ lấy item liên quan

Đã triển khai trong `src/text_based_rag/query_embedding.py` theo thiết kế đã duyệt. Tái sử dụng JSON decoder phục hồi item từ `corpus_chunking.caption_parts`; không thêm module/dependency. Agent chưa chạy embedding, test hoặc sinh/ghi đè output trong lượt bàn giao này. Các quy tắc và số liệu thiết kế phía dưới vẫn dùng để đối chiếu; chưa có kết quả F2 của representation mới.

Điều kiện: corpus embedding mới đã hoàn tất và hash input chunks khớp chunking hiện tại; caption train/dev đã hoàn tất/được chấp nhận, theo query ID với schema có is_relevant; model BGE-M3 và `.venv`/GPU CUDA như corpus embedding. Đóng tác vụ Qwen trước khi chạy. PowerShell từ root, chạy lần lượt:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split dev --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split train --overwrite
```

Lệnh ghi đè trong `artifacts/experiments/text_based_rag/query/query_embedding_dev/` và `query_embedding_train/`, giữ tên file cũ: `vectors.npy`, `queries.jsonl`, `query_text_preview.txt`, `manifest.json`. Input cuối gồm câu hỏi + category khi có item true + ba trường mô tả đã render; không chứa summary, item false, boolean hoặc JSON fragment. Metadata chọn/bỏ/trùng/parse_mode/fallback chỉ nằm ngoài trường text. Không sửa caption gốc hoặc vector corpus. File vector/mapping/preview mới được viết tạm, kiểm tra shape/L2/mapping và input hash trước khi thay file cũ; manifest IN_PROGRESS/FAILED ngăn retrieval dùng bộ output chưa hoàn tất. Nếu bị ngắt giữa bước commit có thể có một phần file mới; chạy lại lệnh --overwrite từ đầu, không hỗ trợ resume query embedding.

Mong đợi nếu input giữ nguyên: dev `(112,1024)`, train `(418,1024)`, float32 L2, manifest `COMPLETED`; giữ đủ mọi query. `question_only_queries` dự kiến 3 dev/16 train; `recovered_caption_queries` dự kiến 2 dev/9 train. Số selected_items/token cuối có thể giảm so với thống kê trước triển khai do loại bản lặp y hệt. Token tối đa 8192, không truncate hoặc bỏ query để né lỗi. Đây là inference, không training. Log khoảng mỗi 30s có bước, processed/total, %, elapsed và ETA; log representation hiển thị tổng question_only/selected_items.

Kiểm tra text thực tế và manifest sau khi chạy:

```powershell
Get-Content .\artifacts\experiments\text_based_rag\query\query_embedding_dev\query_text_preview.txt -TotalCount 45
Get-Content .\artifacts\experiments\text_based_rag\query\query_embedding_dev\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\query\query_embedding_train\manifest.json
```

Nếu corpus index chưa khớp chunk mới, chạy corpus_embedding index --overwrite trước. Nếu caption cũ không có query_id/is_relevant hoặc mapping/câu hỏi/hash không khớp, tạo/kiểm tra lại caption, không tự coi tất cả item là true. JSON bị cắt chỉ được phục hồi khi checkpoint có accepted_by_user và original_status=INCOMPLETE; các caption hiện có đã được chấp nhận nên không cần --allow-incomplete-captions. JSON đầy đủ nhưng sai schema/boolean sẽ báo query ID và không tự chuyển kiểu. OOM: chạy lại với --batch-size 1; input vượt 8192 thì báo ID/token để xử lý riêng, không tự cắt. Không đồng thời chạy caption hoặc ghi đè các input đang embedding. Chờ người dùng chạy/xác nhận, giữ `AWAITING_USER_TEST`.

Đề xuất theo yêu cầu người dùng: mỗi query vẫn có một vector BGE-M3 từ câu hỏi gốc + các item có `is_relevant` đúng kiểu boolean `true`. Giữ cùng model/revision/weights, CLS pooling, L2, BF16 inference, float32 storage, 1.024 chiều và input_prefix rỗng như corpus. Không thêm multi-vector/fusion/weight hoặc dependency. Query retrieval giữ giao diện một query ID–một hàng vector; không dùng train query như tập ví dụ để dự đoán hoặc cập nhật weights.

Đã thống kê read-only trên caption hiện có, không inference hoặc tạo output: train có 409 JSON hợp lệ + 9 JSON bị cắt, dev có 110 JSON hợp lệ + 2 bị cắt; các object hoàn chỉnh trong phần bị cắt được đọc để phân tích. Tổng object đọc được: train 1.559 (962 true, 597 false), dev 408 (226 true, 182 false). Không gặp object hoàn chỉnh thiếu/khác kiểu boolean hoặc thiếu trường mô tả. Có 16 query train, 3 query dev không có item true; 216 query train và 55 query dev có nhiều hơn một item true. Số này tính cả item lặp mà model đã sinh, chưa deduplicate.

Đo bằng tokenizer BGE-M3 local với câu hỏi + category + văn bản ba trường của item true, trước dedup: train median 187 token, p95 472, max 3.577; dev median 191, p95 435, max 683; không query nào vượt 8.192. Cách cũ đưa nguyên JSON có median 390/376 và max 6.066/4.588 token tương ứng train/dev. Độ dài thay đổi nhẹ theo nhãn/format triển khai, không suy ra F2 sẽ tăng từ việc giảm token.

Quy tắc thiết kế:

1. Luôn giữ nguyên câu hỏi để bảo toàn phủ định, câu đúng/sai, đơn vị, hướng/làn và yêu cầu pháp lý. Không trả lời/sửa câu hỏi hoặc đưa đáp án, choices hay relevant_articles/gold vào input.
2. Đọc caption qua mapping theo query ID, kiểm tra câu hỏi, image ID/hash và split khớp. Dùng trường thực tế `is_relevant`, không tìm key `relevant`. Kiểm tra JSON/schema trước khi lọc: `false` bị loại; không chuyển chuỗi `"true"`, số 1 hoặc field bị thiếu thành true. JSON mới không được chấp nhận bị cắt mà sai schema phải báo lỗi với query ID.
3. Render tiếng Việt các item được chọn theo format đồng bộ corpus: `Đối tượng`, `Đặc điểm`, `Ý nghĩa hoặc thông số`. Giữ nguyên vị trí, quan hệ biển chính–phụ trong nội dung Qwen sinh, không tự suy diễn thêm. Category render một lần khi có item được chọn. Không embedding cú pháp JSON, boolean, ID/hash/metadata. Summary toàn ảnh không đưa vào vector vì có thể kể cả đối tượng false và đưa nhiễu trở lại.
4. Nhiều item true được nối theo thứ tự nguồn trong cùng input; không gộp các vector bằng average/max và không gán trọng số câu hỏi/đặc điểm/ý nghĩa khi chưa có thí nghiệm. Nếu model lặp y hệt cả ba trường mô tả của một item trong cùng query, giữ một bản theo thứ tự đầu tiên; chỉ dedup chính xác sau strip, không gộp các biển chỉ vì tương tự hoặc cùng mã. Giữ vị trí source index để truy vết.
5. Không có item true: embedding câu hỏi gốc, ghi `representation_mode=question_only` và lý do. Không fallback sang toàn caption, summary hoặc item false. Vẫn giữ query trong output; không bỏ query và không tạo vector rỗng/zero.
6. Với 11 caption train/dev bị cắt đã được người dùng chấp nhận: đọc từng object hoàn chỉnh trong mảng items bằng JSON decoder, chỉ dùng object đủ ba trường chuỗi và `is_relevant is true`. Không sửa JSON, đoán field dang dở, dùng phần fragment thô hoặc lấy summary làm thay thế. Nếu không lấy được item true hợp lệ thì question_only, ghi riêng lý do truncated_no_recoverable_relevant_item. Checkpoint/raw caption gốc vẫn giữ nguyên. Chính sách query khác corpus: corpus giữ fragment để không mất dữ liệu nguồn, query không đưa fragment chưa xác định relevance vào vector tìm kiếm.
7. Đo lại token của input cuối bằng tokenizer model; giới hạn hard 8.192, không áp 512 như child corpus và không truncate. Nếu vượt, báo query ID và token count trước inference để có quyết định riêng; chưa thêm chia query/fusion. Corpus/query dùng cùng không gian vector nhưng không bắt buộc cùng độ dài. Model không cần instruction prefix cho query theo model card BGE-M3; giữ input_prefix hiện tại.
8. `queries.jsonl` giữ query_id, source image ID, caption_file/hash, vector_row, text và token_count; thêm representation_mode, selected_item_indices, các count item được chọn/bỏ/trùng và parse_mode để giải thích fallback. Manifest ghi chính sách strict true, loại summary, xử lý truncated, tổng question_only và hash source/config. Không sửa description hoặc trạng thái chấp nhận của caption để thực hiện lọc.

Ví dụ input cuối khi có item liên quan:

```text
Câu hỏi:
<câu hỏi nguyên văn>

Loại hình: SIGN_ILLUSTRATION

Đối tượng: <identifier của item true>
Đặc điểm: <visual_or_geometric_features>
Ý nghĩa hoặc thông số: <meaning_or_specification>
```

Đã sửa trực tiếp `query_embedding.py`, tái sử dụng load_encoder/embed_batch; thêm ghi đè output train/dev với file tạm và kiểm tra mapping/vector/hash trước commit như corpus embedding. Không dùng lại vector cũ chỉ vì query ID giống nhau. Caption public-test cũ không có flag không thể tự coi là true; code báo cần caption schema mới thay vì âm thầm lọc/giữ toàn bộ. Chưa chạy hoặc chỉnh artifact public test trong bước này.

Đánh giá dự kiến: giữ corpus index mới cố định, so query representation cũ và strict-relevant trên dev bằng evaluator F2@5/hit rate hiện có, ghi riêng nhóm question_only, truncated, nhiều item và lỗi Qwen đánh nhầm relevance. Chỉ lấy true có thể loại nhầm biển cần thiết; đây là rủi ro do relevance model sinh, không thể đảm bảo recall/F2 tăng. Không dùng public/private test chọn chính sách. Code đã sửa, chưa tạo vector hoặc chạy retrieval mới.

Nguồn: [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) xác nhận dense embedding 1.024 chiều, input tối đa 8.192 token và không cần thêm query instruction. Chính sách relevance/fallback ở trên là thiết kế cho dữ liệu project, chưa được xác nhận bằng kết quả retrieval.

## Embedding corpus từ child chunks mới

Đã sửa `src/text_based_rag/corpus_embedding.py` để `index --overwrite` thay index cũ, kiểm tra hash output chunking, số chunk/parent, khóa parent và citation, đo lại token bằng tokenizer BGE-M3 trước inference. Model chỉ nhận trường `text` từ `corpus_chunking/chunks.jsonl`; metadata ngoài text không được embedding, được giữ nguyên trong mapping. Parent đầy đủ được copy để lookup, không embedding toàn parent. Chưa làm sạch thêm nội dung 186 caption_fragment: cú pháp JSON nằm trong text fragment vẫn được embedding như dữ liệu đã chấp nhận. Không chạy lại chunking hoặc thay description.

Điều kiện: corpus_chunking manifest `COMPLETED` và các hash khớp; tokenizer/weights BGE-M3 revision đã pin tại models; `.venv` có torch CUDA BF16, transformers/numpy; ít nhất 5 GiB VRAM trống theo loader. Đóng tác vụ Qwen caption trước khi chạy. PowerShell từ root:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_embedding index --overwrite
```

Lệnh ghi đè `artifacts/experiments/text_based_rag/corpus/corpus_embedding/`: `vectors.npy`, `chunks.jsonl`, `organized_corpus.jsonl`, `manifest.json`. Vector/mapping/parent mới được viết vào file tạm, kiểm tra dữ liệu và input hash trước khi thay các file index cũ. Manifest `IN_PROGRESS`/`FAILED` ngăn retrieval đọc bộ index chưa hoàn tất; không đảm bảo thay cả ba file cùng lúc nếu process bị ngắt giữa bước commit. Không xóa thư mục index. Không hỗ trợ resume embedding; nếu bị ngắt thì chạy lại cùng lệnh từ đầu.

Mong đợi với chunking hiện tại: vectors shape **(5459, 1024)**, dtype float32, finite/L2; mapping 5.459 hàng, giữ parent/item/provenance và thêm `vector_row`; 402 parent. Input dài nhất 503 token, mọi chunk phải khớp token_count và <=512, thấp hơn input limit model 8192. Encoder dùng BF16, CLS pooling và L2 như trước; đây là inference/indexing, không training. Log mỗi khoảng 30s gồm phase, processed/total, %, elapsed, ETA. Manifest cuối `COMPLETED` có input/output hash và thông tin GPU/VRAM.

Kiểm tra sau khi chạy:

```powershell
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_embedding\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_embedding\chunks.jsonl -TotalCount 1
```

Nếu output tồn tại mà thiếu cờ, dùng `--overwrite`; hash/token/parent mismatch thì kiểm tra hoặc chạy lại chunking, không bỏ validation. Nếu OOM, đóng model GPU khác rồi chạy lại với `--batch-size 1`; không đổi model/pooling để né lỗi. Nếu thiếu weights, tải qua CLI `corpus_embedding download` trong `.venv` trước; không bật offline khi tải. Agent chưa chạy code/test/pipeline trong lượt bàn giao embedding; trạng thái `AWAITING_USER_TEST`. Query vectors và retrieval/prediction chưa được tái tạo theo caption mới trong bước này.

## Corpus chunking parent–child theo item ảnh

Người dùng đã duyệt và yêu cầu triển khai/chạy, ghi đè kết quả chunking cũ. Đã sửa trực tiếp `src/text_based_rag/corpus_chunking.py`, không thêm module/dependency. Pipeline đã chạy qua CLI thực tế, CPU, khoảng 10s; ghi đè các output hiện có dưới `artifacts/experiments/text_based_rag/corpus/corpus_chunking/`. Caption checkpoint và dữ liệu gốc không thay đổi. Embedding/index và query representation chưa được cập nhật hoặc chạy lại.

Kết quả: 402 parent, coverage 787 ảnh; 5.459 child gồm 3.138 `legal_text`, 731 `table_row`, 1.404 `caption_item`, 186 `caption_fragment`. Có 1.399 item-unit theo lần xuất hiện hình trước khi chia phần (một ảnh có thể xuất hiện nhiều block trong cùng điều), 27 fragment-unit từ caption bị cắt. Giới hạn cấu hình 512 token, overlap 64 chỉ khi phải chia một unit dài; token lớn nhất thực tế 503. Manifest `COMPLETED`; bảng bàn giao giữ `AWAITING_USER_TEST`. Pipeline đã kiểm tra unique chunk ID, parent hợp lệ, coverage block/ảnh, toàn bộ paragraph và description gốc còn ở parent, và mọi chunk trong ngân sách token. Có 148 issue/provenance note, gồm caption bị cắt được người dùng chấp nhận, fragment và nhóm dài bị chia; xem `issues.jsonl`.

Điều kiện: preprocessing schema 2 và caption manifest `COMPLETED`/đã được người dùng chấp nhận, đủ caption khớp image ID và hash nguồn; tokenizer BGE-M3 local trong models; `.venv` có transformers. Chạy từ root PowerShell để tái tạo/ghi đè:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_chunking --overwrite
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_chunking\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_chunking\embedding_text_preview.txt -TotalCount 45
```

Mong đợi manifest `COMPLETED` cùng thống kê phía trên nếu input/code/config giữ nguyên. Các file được thay: `chunks.jsonl`, `organized_corpus.jsonl`, `issues.jsonl`, `organized_corpus.txt`, `embedding_text_preview.txt`, `manifest.json`. Parent chứa text luật/bảng và description gốc; caption child chỉ chứa item đã render hoặc fragment, kèm metadata. Chunker dựng/kiểm tra kết quả trước khi thay file dữ liệu bằng temporary + rename; manifest ghi tiến trình/trạng thái để chặn dùng output chưa hoàn tất. Log mỗi khoảng 30s gồm số điều/tổng, %, elapsed và ETA. Nếu output tồn tại mà thiếu `--overwrite`, lệnh từ chối; nếu caption/source hash không khớp thì sửa nguồn, không bỏ validation. Context item dài quá ngân sách báo lỗi thay vì cắt bỏ tên/identity. Caption bị cắt không được tự hoàn thiện; nội dung model có thể sai dù chunk hợp lệ. Index cũ đang phản ánh corpus cũ, cần tái tạo embedding ở bước sau trước khi đánh giá retrieval mới.

Đã phân tích read-only bằng tokenizer BGE-M3 local, không inference hoặc sinh artifact: corpus có 402 điều, 787 ảnh; 760 description parse được JSON chứa 1.320 item; 27 description bị cắt không parse được JSON hoàn chỉnh nhưng đọc được 75 object item đầy đủ. Trong 760 JSON hợp lệ có 393 TECHNICAL_DIAGRAM, 298 SIGN_ILLUSTRATION, 69 MARKING_AND_SIGNAL. Với văn bản item gồm tiêu đề văn bản/điều, loại hình, identifier, đặc điểm và ý nghĩa/thông số: median 199 token, p95 318, max 1.412; 4/1.320 item vượt 512 token. Số liệu này chưa gồm nhãn hình và token thay đổi do lặp ngữ cảnh khi chia nhỏ. Description toàn ảnh có median 327, p95 1.315, max 4.085 token BGE-M3. Giới hạn input BGE-M3 local là 8.192 token, nhưng ưu tiên chunk theo ý nghĩa, không kéo chunk sát giới hạn model. Có 86 article ID xuất hiện ở nhiều văn bản: không dùng article_id đơn lẻ làm khóa parent.

Quy tắc đã triển khai:

1. Parent là toàn bộ điều luật đã chuẩn hóa cùng liên kết caption gốc. Giữ `article_key` hiện tại làm `parent_id`, đồng thời giữ `document_id`/`id` làm citation `(law_id, article_id)`. Không sinh citation mới từ mã biển Qwen nhận diện. Parent lưu trong `organized_corpus.jsonl`, không embedding toàn điều dài.
2. Các child có hai nguồn nội dung trong cùng index: văn bản pháp lý/bảng đã chuẩn hóa và caption theo item. Chỉ tách item ảnh sẽ bỏ mất điều kiện áp dụng, ngoại lệ hoặc thông số nằm trong văn bản/bảng. Văn bản luật giữ nhóm khoản/điểm, đoạn dẫn và ngoại lệ như hiện có; bảng giữ từng hàng cùng tiêu đề/đơn vị/header và ô gộp. Loại chunk có thể là `legal_text`, `table_row`, `caption_item`, `caption_fragment`.
3. Mỗi item ảnh hợp lệ là một child độc lập, không ghép nhiều biển/phôi vào một vector. Render tiếng Việt theo thứ tự: tiêu đề văn bản/điều; nhãn hình gốc nếu có; loại hình; đối tượng; đặc điểm; ý nghĩa/thông số. Không embedding JSON syntax, raw metadata hoặc toàn bộ summary đa đối tượng lặp trong từng item. Không suy đoán mã biển, không tự ghép một đoạn luật với item bằng ý nghĩa hoặc tên do model suy diễn. Summary/raw JSON đầy đủ vẫn giữ ở parent/nguồn để truy vết.
4. Giới hạn hard 512 token gồm mọi context và special token, đo bằng chính tokenizer embedding. Item không vượt ngân sách giữ nguyên, overlap 0 giữa các item độc lập. Item/nhóm luật/hàng bảng quá dài chia theo đoạn/câu, fallback token như hiện có, overlap 64 ở các phần của cùng nhóm; lặp tên đối tượng/loại hình/ngữ cảnh nhận diện ở từng phần. Không chia ngang từ item này sang item khác; không truncation để vừa input. Điều chỉnh chỉ sau đánh giá dev, không chọn cấu hình bằng public/private test.
5. Metadata giữ các trường hiện tại cho embedding/retrieval, thêm `parent_id`, `chunk_type`; caption child có `image_id`, `image_category`, `item_index` và `part_index` nếu phải chia. Mọi child giữ `source_block_ids` và source caption để truy vết. Item index là vị trí trong output model, không phải định danh một biển giao thông ngoài thực tế. Một item có thể mô tả nhiều vật hoặc bị Qwen nhận sai; chunking không được coi việc tách item là xác nhận nội dung đúng.
6. Với 27 caption người dùng đã chấp nhận bị cắt: dùng JSON decoder lấy từng object item hoàn chỉnh ở mảng items, chỉ chấp nhận đủ ba trường chuỗi mô tả. Không thêm dấu ngoặc, đoán/sinh lại trường còn thiếu hoặc biến object dang dở thành item hoàn chỉnh. Phần chưa đọc được giữ thành `caption_fragment`, chia bounded text và gắn provenance/trạng thái gốc bị cắt; nếu không lấy được item nào thì dùng toàn description làm fallback. Không bỏ nội dung nguồn; raw description luôn được lưu nguyên ở parent/checkpoint. Category lấy từ trường đã đọc được, thiếu thì metadata null, không tự phân loại lại. Tất cả item/fragment của cùng ảnh ánh xạ về điều luật thật trong preprocessing. Tách fragment nhằm giữ dữ liệu được người dùng chấp nhận, không coi đây là nội dung đầy đủ/chính xác hoặc JSON hợp lệ.
7. Khi dựng child luật và hàng bảng, không nhúng lại toàn caption JSON qua tham chiếu figure như code Organizer hiện tại; giữ liên kết block/image tới child ảnh. Điều này tránh cùng mô tả ảnh bị lặp ở nhiều vector và việc xếp hạng bị chi phối bởi chuỗi JSON dài. Parent vẫn có đầy đủ luật, bảng và caption; nhãn hình/tham chiếu pháp lý trong văn bản không bị xóa.
8. Retrieval giữ quy tắc đang có: tính cosine mọi child, điểm citation bằng max score của các child có cùng `(document_id,id)`, rồi top 5 citation khác nhau. Giữ child tốt nhất làm evidence và lookup parent khi cần đọc đầy đủ. Không lấy top 5 child rồi mới deduplicate, vì nhiều child cùng parent có thể chiếm hết kết quả. Chưa đặt weight theo chunk_type hoặc filter cứng image_category; query/model có thể phân loại sai. Max aggregation tránh cộng điểm chỉ vì phụ lục có nhiều item, nhưng vẫn có thiên lệch nhiều cơ hội khớp ở điều dài nên cần kiểm tra lỗi.

Phạm vi triển khai: đã sửa `corpus_chunking.py` để render item, fallback caption bị cắt và sinh metadata; giữ giao diện `chunks.jsonl`/`organized_corpus.jsonl`. `corpus_embedding.py` đã bổ sung ghi đè/validation theo hướng dẫn phía trên, vẫn embedding trường text và giữ metadata; chưa chạy index mới. `query_retrieval.py` giữ scoring hiện tại; chưa thêm metadata child vào evidence, chưa thay query representation hoặc dùng is_relevant để lọc. Không thêm module hay dependency chỉ để tổ chức parent–child.

Đánh giá sau khi được phép triển khai/chạy: coverage đầy đủ 402 parent/787 ảnh; mọi child có parent/citation tồn tại, source truy vết được, unique ID, token <=512; thống kê item đọc được/fallback/caption gốc bị cắt; không mất văn bản pháp lý/bảng/description nguồn. Sau đó đo F2@5, hit rate@5 trên dev bằng cùng query vector để tách ảnh hưởng của chunking, dùng evaluator hiện có. Các metric dev cũ dùng caption corpus/prompt cũ nên không phải so sánh kiểm soát riêng chunking; nếu muốn so riêng hai cách chia thì cả hai phải dùng cùng caption corpus mới và cùng query vector. Không cam kết similarity >0,90, hoặc F2 tăng trước khi đo.

Nguồn tham khảo: [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3) xác nhận embedding 1.024 chiều, input 8.192 token; [ParentDocumentRetriever chính thức](https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever) mô tả tìm child nhỏ rồi lookup parent lớn; chỉ tham khảo mô hình dữ liệu, không thêm LangChain. [Hướng dẫn BAAI về similarity](https://huggingface.co/BAAI/bge-base-en-v1.5) nhấn mạnh thứ tự tương đối và hiệu chỉnh threshold theo dữ liệu; không áp dụng phân phối score của BGE v1.5 như số đo cho BGE-M3.

## Ghi đè caption query train/dev bằng prompt đã duyệt

Tác vụ dev do agent khởi chạy đã được dừng theo yêu cầu; train chưa được khởi chạy. Code hỗ trợ ghi đè và resume, bàn giao để người dùng tự chạy các lệnh bên dưới. Nếu muốn tiếp tục dev đã chạy dở, bỏ `--overwrite` ở lệnh dev; giữ cờ để sinh lại từ đầu. Agent không chạy pipeline trong lượt bàn giao này.

Người dùng đã chấp nhận lần thử prompt ưu tiên bằng chứng ảnh. `query_image_captioning.py` có `--overwrite` để sinh lại mọi cặp ảnh–câu hỏi, thay caption/mapping/manifest/JSONL trong thư mục train/dev hiện có. Với mỗi ảnh, query đầu tiên dùng lại tên file `<source_image_id>_description.json` cũ để ghi đè description; các query còn lại có `<query_id>_description.json` riêng vì độ liên quan phụ thuộc câu hỏi. Mapping trỏ đúng caption của từng query; không dùng một description chung cho các câu hỏi khác nhau. Giữ schema và cấu hình model như lần thử đã duyệt.

Điều kiện: model/projector Qwen3.5 local, runtime llama.cpp trong `.venv`, requests, GPU đủ VRAM, hai split train/dev và ảnh train đã chuẩn bị. PowerShell từ root, chạy lần lượt để không nạp hai model cạnh tranh GPU:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split dev --overwrite
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split train --overwrite
```

Output: `artifacts/experiments/text_based_rag/query/query_image_captioning_dev/` và `query_image_captioning_train/`. Không có thư mục timestamp mới khi chạy toàn split. Mong đợi 112 caption dev và 418 caption train (theo query); JSON hợp lệ, `is_relevant` boolean, mapping không trùng đường dẫn, manifest `COMPLETED` khi mọi query hoàn tất. Xem kết quả:

```powershell
Get-Content .\artifacts\experiments\text_based_rag\query\query_image_captioning_dev\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\query\query_image_captioning_train\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\query\query_image_captioning_train\captions\train_1_1_description.json
```

`--overwrite` bỏ qua toàn bộ checkpoint cũ và thay từng caption bằng file tạm rồi rename; không cần tự xóa thư mục. Nếu bị ngắt hoặc có query `INCOMPLETE`, chạy lại split tương ứng **bỏ `--overwrite`** để giữ caption hoàn tất của đúng prompt/model và làm tiếp phần thiếu. JSON sai/bị cắt giữ `INCOMPLETE`, không tự coi là đã được duyệt. Server lỗi/OOM xem `llama-server.log`, đóng ứng dụng GPU khác; tiến trình in số lượng/tổng, %, elapsed và ETA mỗi khoảng 30s. Đây là inference. Không dùng nhãn/đáp án để sinh mô tả. Embedding/retrieval cũ chưa được cập nhật theo caption mới.

Đã sửa lỗi dev dừng sau 33/112 query: `train_149` dùng `train_3_7.jpg` nhưng bytes là WebP (`RIFF...WEBP`), runtime không giải mã được và không tìm thấy ffprobe nên trả HTTP 400. Query caption hiện dùng Pillow đã cài đọc ảnh theo định dạng thực, mã hóa PNG lossless trong bộ nhớ rồi gửi MIME `image/png`; không crop/resize, không thay ảnh nguồn hoặc tạo file ảnh mới. Lỗi HTTP tiếp theo sẽ hiển thị query ID, tên ảnh và response body của server. Agent chỉ sửa code/đọc log, chưa chạy lại pipeline. Giữ `AWAITING_USER_TEST`.

Tiếp tục dev bằng lệnh dưới đây để giữ 31 caption COMPLETED, thử lại 2 caption INCOMPLETE và làm các query còn thiếu:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split dev
```

Mong đợi query dùng ảnh WebP không còn lỗi decode HTTP 400, sau đó tiến trình tiếp tục đến 112/112. Nếu manifest vẫn `INCOMPLETE` vì chạm giới hạn token, gửi các query ID trong `incomplete_queries`; không dùng `--overwrite` chỉ để tiếp tục vì sẽ sinh lại toàn bộ caption. Train chưa được agent chạy; khi dev hoàn tất, dùng lệnh train `--overwrite` phía trên để thay caption train cũ.

Đã sửa tiếp logic resume khi lần `--overwrite` bị ngắt: thư mục có thể chứa đồng thời checkpoint mới theo query và file caption theo ảnh từ model cũ chưa được ghi đè. Code trước đây báo `Query/ảnh/cấu hình đã thay đổi: train_3_7_description.json` vì file này chưa có `query_id`. Hiện code kiểm tra hash ảnh trước, nhận diện caption cũ không có query ID/đơn vị caption theo query để sinh lại, giữ các checkpoint mới COMPLETED khớp query/câu hỏi/settings; mismatch của checkpoint mới vẫn báo lỗi. Không đánh dấu caption cũ là hoàn tất cho prompt mới. Dùng lại lệnh dev **không có `--overwrite`** phía trên; mong đợi qua bước đọc checkpoint, giữ 31 caption mới hoàn tất, thử lại 2 caption chưa hoàn tất và thay các caption cũ còn lại. Agent chưa chạy code/test/pipeline hoặc sửa artifact; trạng thái vẫn `AWAITING_USER_TEST`.

## Thử prompt query ảnh + câu hỏi trên train_1_1

Đã sửa `src/text_based_rag/query_image_captioning.py` dùng prompt query mới nhất người dùng cung cấp, gửi ảnh + `question` vào Qwen3.5-9B Q4_K_M qua llama.cpp; không đưa gold citation, lựa chọn đáp án hoặc đáp án đúng vào model. Output JSON có `image_category`, `summary`, `items`; mỗi item có ba trường mô tả đồng bộ corpus và `is_relevant` kiểu boolean. Ràng buộc schema khi sinh và kiểm tra lại JSON/boolean khi lưu.

Theo yêu cầu chạy lại ngày 15/10/2026, prompt query đã bổ sung ưu tiên bằng chứng ảnh: quan sát hình dạng/màu/chữ/vị trí trước khi xét câu hỏi, không coi phát biểu trong câu hỏi là sự thật, không suy ngược hình thái từ tên biển, cho phép ghi “không đọc rõ/không xác định”, chỉ khẳng định cùng cột khi quan sát được. Câu hỏi đặt sau phần hướng dẫn quan sát; chỉ dùng để gán độ liên quan. Với đối tượng phía xa không được hỏi hoặc chưa đủ căn cứ liên quan, gán `false`. Không thêm ví dụ riêng về biển STOP vào prompt; không đổi model, schema, temperature, seed, context, ảnh hoặc crop.

Caption mới phụ thuộc cả câu hỏi, nên checkpoint/mapping theo `query_id`, không dùng chung một caption cho mọi câu hỏi trên cùng ảnh. `query_embedding.py` đã thêm kiểm tra query ID và câu hỏi khi đọc caption kiểu mới; vẫn đọc được artifact caption cũ. Chưa thay đổi cách embedding lọc theo `is_relevant`: trường này hiện được giữ trong chuỗi JSON mô tả, chưa dùng để loại item trước embedding. Toàn bộ input train/dev/public test vẫn theo các đường dẫn hiện có; caption cũ khác prompt/model/mapping bị từ chối, không tự ghi đè.

Điều kiện: `.venv` có requests, corpus module `corpus_image_captioning.py`, model/projector local tại `models/huggingface/Qwen3.5-9B-GGUF/`, runtime `.venv/tools/llama.cpp-b11624/llama-server.exe`, GPU còn đủ VRAM, train split và ảnh `data/processed/images/train/train_1_1.jpg`. Chạy PowerShell từ root:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split train --image train_1_1
```

Ảnh này có 4 query `train_66`–`train_69`; lệnh chạy riêng từng cặp ảnh–câu hỏi và tạo thư mục timestamp mới dưới `artifacts/experiments/text_based_rag/query/query_caption_review/`. Mỗi query có `<query_id>_query_description.json` thuần JSON, `<query_id>_prompt.txt` và `captions/<query_id>_description.json`; có mapping, manifest, JSONL tổng hợp và log runtime. Không ghi đè caption/vector train hiện có. Mong đợi 4 caption JSON hợp lệ, category thuộc 3 loại, từng `is_relevant` là boolean; manifest `COMPLETED` khi tất cả caption hoàn tất, không đồng nghĩa nội dung đã được duyệt. Cấu hình giữ như corpus: context 16384, output tối đa 4096 token, temperature 0, seed 2025, tắt thinking; server tự dừng khi kết thúc. Tác vụ in bước, số lượng/tổng, %, elapsed và ETA khoảng mỗi 30 giây và sau mỗi query.

Lần thử đã chạy: [train_66](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T050638576830Z/train_66_query_description.json), [train_67](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T050638576830Z/train_67_query_description.json), [train_68](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T050638576830Z/train_68_query_description.json), [train_69](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T050638576830Z/train_69_query_description.json). Thời gian khoảng 25s, 4/4 JSON hợp lệ, manifest `COMPLETED`. Chất lượng chưa đạt đồng đều: train_67/train_69 mô tả STOP thành hình tròn nền trắng chữ đỏ trong khi ảnh là bát giác nền đỏ chữ trắng; train_68 đưa cả biển phía xa vào và đánh dấu tất cả item `true`, có lẫn chữ Trung Quốc trong mô tả hình dạng. Không sửa tay output hoặc đổi prompt sau lần thử; chờ người dùng kiểm tra.

Lần thử tiếp theo với prompt ưu tiên bằng chứng ảnh: [train_66 mới](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T051107977986Z/train_66_query_description.json), [train_67 mới](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T051107977986Z/train_67_query_description.json), [train_68 mới](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T051107977986Z/train_68_query_description.json), [train_69 mới](artifacts/experiments/text_based_rag/query/query_caption_review/train_1_1_20261015T051107977986Z/train_69_query_description.json). Khoảng 26s, 4/4 JSON hợp lệ, manifest `COMPLETED`; output cũ giữ nguyên. Quan sát thủ công: train_67/train_69 mô tả STOP đúng bát giác nền đỏ chữ trắng; train_68 chỉ gán `true` cho biển tam giác, các biển nhỏ `false`. Tuy nhiên train_66 đảo màu STOP thành nền trắng chữ đỏ; model còn nhận sai biển nhỏ thành cấm dừng xe/xe máy, khẳng định cùng cột thiếu căn cứ và train_69 vẫn có chữ “八角”. Do đó prompt mới cải thiện một số điểm nhưng chưa ổn định về thị giác, chưa thể kết luận chỉ prompt là nguyên nhân hay chất lượng chung đã tăng. Dùng cùng command phía trên để tự chạy lại; chưa chạy toàn split/embedding/retrieval, giữ `AWAITING_USER_TEST`.

Lỗi đã biết: thiếu model/runtime thì kiểm tra các đường dẫn trên; server dừng/OOM xem `llama-server.log` và đóng ứng dụng GPU khác; JSON sai/bị cắt giữ `INCOMPLETE`. Không dùng command toàn split để ghi đè artifact cũ: khi bị báo khác cấu hình, chỉ dùng `--image` để xem riêng hoặc tự lưu thư mục cũ trước. Chưa chạy lại caption toàn split, embedding hoặc retrieval.

## Chạy lại toàn bộ caption corpus trong thư mục cũ

Đã đổi tên module thành `corpus_image_captioning.py` và đổi tên thư mục output hiện có từ `image_captioning/` sang `corpus_image_captioning/`, không sinh lại nội dung. Import của corpus chunking/embedding và query caption/embedding/retrieval đã cập nhật; chunking đọc đúng thư mục mới. Lệnh module cũ `text_based_rag.image_captioning` không còn sử dụng. Trạng thái vẫn `AWAITING_USER_TEST`.

Người dùng đã duyệt kết quả thử prompt phân loại trên image002. Đã sửa `src/text_based_rag/corpus_image_captioning.py` để `--overwrite` sinh lại tất cả caption bằng Qwen3.5-9B Q4_K_M, ảnh + văn bản điều luật và prompt phân loại đã duyệt, ghi đè trực tiếp dưới `artifacts/experiments/text_based_rag/corpus/corpus_image_captioning/`. Agent chưa chạy lệnh này.

Điều kiện: corpus chuẩn hóa schema 2 đã hoàn tất, ảnh corpus còn nguyên, `.venv` có requests, model/projector và runtime llama.cpp CUDA b11624 tại các đường dẫn mô tả bên dưới; GPU còn đủ VRAM. Chạy PowerShell từ root:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_image_captioning --overwrite
```

`--overwrite` bỏ qua toàn bộ caption cũ, kể cả caption COMPLETED, để sinh lại. Mỗi caption được thay thế bằng file tạm rồi rename sau inference/validation. Không tạo thư mục timestamp hoặc file prompt/response/JSON riêng, không thêm lịch sử metadata; chỉ cập nhật các output hiện có: `captions/*_description.json`, `manifest.json`, `image_captions.jsonl`, `structured_corpus.jsonl`, `corpus_preview.md`. Giữ manifest và cấu hình/trạng thái hiện có để chunking tiếp tục kiểm tra nguồn và từ chối corpus chưa hoàn tất. `description` chứa chuỗi JSON với `image_category`, `summary`, `items` đúng schema. Không xuất thêm bản sao các trường này vào metadata caption. Chunk/vector/index và prediction cũ chưa được tái tạo bởi lệnh này.

Mong đợi mọi ảnh duy nhất trong corpus có caption mới, manifest `COMPLETED`, JSON hợp lệ, các figure trong corpus tổng hợp có `description_vlm` mới. Đây là inference, không training; tiến trình in bước, số lượng/tổng, %, elapsed, ETA khoảng mỗi 30 giây và sau từng ảnh. Xem kết quả:

```powershell
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_image_captioning\manifest.json
Get-Content .\artifacts\experiments\text_based_rag\corpus\corpus_image_captioning\captions\image002_description.json
```

Nếu bị ngắt giữa chừng hoặc có caption JSON lỗi/chạm giới hạn token: chạy lại cùng lệnh **bỏ `--overwrite`** để chỉ làm tiếp ảnh thiếu/chưa COMPLETED của đúng prompt/model hiện tại. Nếu prompt/model/config đã thay đổi thì phải dùng `--overwrite`. Khi có ảnh chưa hợp lệ, manifest giữ `INCOMPLETE` và không thay thế corpus tổng hợp; không dùng bản tổng hợp cũ trong lúc đó. Khi thiếu model/runtime hoặc llama-server dừng, kiểm tra đường dẫn và VRAM; có thể dùng CLI `--image image002.jpg` bên dưới để xem log riêng. JSON hợp lệ vẫn cần người dùng xem chất lượng. Chờ người dùng chạy và phản hồi trước khi chuyển thành `COMPLETED`.

## Xem thử prompt corpus mới trên image002

Đã sửa `src/text_based_rag/corpus_image_captioning.py`: dùng prompt phân loại hình mới nhất người dùng cung cấp ngày 15/10/2026, chèn văn bản điều luật tương ứng từ corpus chuẩn hóa vào `{law_article_text}`, gửi cùng ảnh đến Qwen3.5-9B Q4_K_M qua llama.cpp. Văn bản gồm tên văn bản, ID/tên điều và toàn bộ text trong các block của điều, kể cả ô bảng; không đưa HTML hoặc caption VLM cũ vào prompt. Nếu một ảnh thuộc nhiều điều, ghép các điều tương ứng và giữ nguồn. Output JSON có `image_category` (đúng một trong `SIGN_ILLUSTRATION`, `TECHNICAL_DIAGRAM`, `MARKING_AND_SIGNAL`), `summary` và `items`; mỗi item có `identifier`, `visual_or_geometric_features`, `meaning_or_specification`. Schema được ràng buộc khi sinh và kiểm tra lại sau sinh. Caption metadata vẫn giữ `description` dạng chuỗi JSON để tương thích bên đọc corpus hiện có. Query caption chưa chuyển runtime.

Điều kiện: `.venv`, requests đã cài trong `.venv`, corpus chuẩn hóa schema 2, model/projector trong `models/huggingface/Qwen3.5-9B-GGUF/` và runtime CUDA chính thức llama.cpp **b11624** tại `.venv/tools/llama.cpp-b11624/`. Runtime và model/projector đã tải và kiểm tra SHA256; không cài vào Python hệ thống. Code mở server riêng trên loopback, tự dừng server khi kết thúc; chỉ chế độ `--image` ghi log runtime riêng. Cấu hình: context 16384, max output 4096 token, temperature 0, seed 2025, tắt thinking, không crop ảnh. Đây là inference.

Chạy PowerShell từ root để xem riêng ảnh này bằng chính CLI caption:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.corpus_image_captioning --image image002.jpg
```

Mỗi lần dùng `--image` tạo thư mục có timestamp riêng trong `artifacts/experiments/text_based_rag/corpus/corpus_caption_review/`, không ghi đè caption/corpus/index cũ và không xuất corpus đầy đủ. Trong đó có JSON thuần `image002_corpus_description.json`, prompt thực tế `image002_description_prompt.txt`, response model, `captions/image002_description.json`, `manifest.json` và `llama-server.log`. Mong đợi JSON hợp lệ, category thuộc ba loại trên, summary không rỗng và đủ 3 trường dạng chuỗi không rỗng cho mỗi item; JSON đúng schema không chứng minh mô tả đúng ảnh. Khi JSON sai/bị cắt, caption giữ `INCOMPLETE`, lưu response để xem; khi runtime lỗi, xem `llama-server.log`. Chế độ toàn corpus dùng `--overwrite` theo hướng dẫn phía trên để thay caption cũ có cấu hình khác.

Lần thử prompt cũ ngày 15/10/2026: [JSON model sinh](artifacts/experiments/text_based_rag/corpus/corpus_caption_review/image002_20261014T222918297490Z/image002_corpus_signs.json). Model gán sai mã P.106a–P.106e và biểu tượng xe tải/xe máy. Giữ nguyên output này để so sánh.

Lần thử prompt phân loại mới ngày 15/10/2026: [JSON kết quả mới](artifacts/experiments/text_based_rag/corpus/corpus_caption_review/image002_20261014T224202536700Z/image002_corpus_description.json), [prompt thực tế](artifacts/experiments/text_based_rag/corpus/corpus_caption_review/image002_20261014T224202536700Z/image002_description_prompt.txt). Timestamp folder là UTC. Model phân loại `TECHNICAL_DIAGRAM`, trả 4 item (tròn, bát giác, tam giác, chữ nhật/vuông), không sinh mã P.106 hoặc biểu tượng xe. Các số D/B/A, L/B/R/C và màu viền đỏ/trắng có căn cứ trong Bảng 1 của Điều 12 gửi kèm; màu không được quan sát từ ảnh đen trắng. Giới hạn: chưa tách tam giác xuôi/ngược và chưa nêu góc 45° trên ảnh. JSON hợp lệ, manifest `COMPLETED`, toàn lệnh khoảng 19s; đây chỉ là trạng thái inference/validation, chưa được người dùng duyệt nội dung. Chưa chạy lại toàn corpus/chunk/index; giữ `AWAITING_USER_TEST` và chờ phản hồi.

## Luồng hiện tại

```text
Câu hỏi dev + caption ảnh → BGE-M3 → cosine với mọi chunk trong corpus luật
→ lấy điểm chunk cao nhất cho mỗi citation → top 5 citation khác nhau
```

Retrieval chỉ dùng nội dung corpus luật để xếp hạng; nhãn dev chỉ được đọc để đánh giá sau khi dự đoán. Không fine-tune và không dùng nhãn/query mẫu train để tạo prediction. Hòa điểm theo citation ID, rồi thứ tự chunk trong mapping.

Theo yêu cầu ngày 15/10/2026, đã xóa hai module chạy example retrieval và fusion, sửa `src/text_based_rag/query_retrieval.py` sang dev, chạy lại inference CPU và ghi đè các file kết quả trong `query_retrieval/` bằng corpus-only. Caption/vector train/dev và corpus hiện có được giữ nguyên. Output thí nghiệm fusion cũ trong `example_corpus_fusion/` chỉ là artifact lịch sử, không được sử dụng bởi luồng hiện tại.

Kết quả corpus-only trên **112 query dev**: macro **F2 = 0,1269011141**, precision = 0,0678571429, recall = 0,1733630952, F1 = 0,0939432591; hit rate@5 = **30,36%**. Có 5 gold citation thiếu trong index, vẫn giữ trong mẫu số recall. Đã tạo đủ 61 description dev nhưng 7 bị cắt ở 384 token (ảnh hưởng 15 query); giữ `INCOMPLETE` và dùng qua cờ cho phép bên dưới. Manifest retrieval `COMPLETED`; trạng thái bàn giao vẫn `AWAITING_USER_TEST` đến khi người dùng xác nhận.

Đã đối chiếu prediction mới với hàm `eval_retrieval` chính thức trong `data/raw/VLSP2025-MLQA-TSR/retrieval.py`: F2 = `0.1269011140299031`, khớp evaluator local trong sai số epsilon. Chỉ nạp hàm evaluator để không khởi tạo model visual của baseline gốc.

## Chạy corpus retrieval trên dev

Chạy PowerShell từ root. Cần `.venv`, `data/processed/splits/dev.json`, vector/mapping/manifest `query_embedding_dev/` và `corpus_embedding/` đã hoàn tất. Retrieval dùng CPU và vector BGE-M3 hiện có, không cần GPU hay tải model lại. Không cần embedding/query train.

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --overwrite
```

**`--overwrite` thay thế kết quả corpus trong `query_retrieval_corpus/`.** Bỏ cờ để từ chối ghi đè nếu thư mục đã tồn tại. Code kiểm tra hash, encoder, shape/L2, mapping, split và prediction trước khi bắt đầu ghi output.

Đọc kết quả đã chạy:

```powershell
Get-Content .\artifacts\experiments\text_based_rag\query\query_retrieval_corpus\metrics.json
Get-Content .\artifacts\experiments\text_based_rag\query\query_retrieval_corpus\retrieval_preview.txt -TotalCount 35
```

| Output dưới `artifacts/experiments/text_based_rag/query/query_retrieval_corpus/` | Nội dung |
| --- | --- |
| `retrieval_results.jsonl` | 112 prediction, mỗi query có 5 citation cùng cosine và chunk luật tốt nhất |
| `retrieval_preview.txt` | Câu hỏi, metric, gold citation và nội dung chunk để đọc thủ công |
| `metrics.json` | Macro/micro precision, recall, F1, F2, hit rate và gold thiếu index |
| `per_query_metrics.jsonl` | 112 dòng metric theo query |
| `manifest.json` | Method `dense_corpus_retrieval`, split `dev`, cấu hình, encoder, hash, thời gian và tài nguyên |

Mong đợi 112 dòng prediction và 112 dòng metric, mỗi prediction có đúng 5 citation khác nhau tồn tại trong index, manifest `COMPLETED`. Metric được tính lại từ embedding đang có; số liệu baseline phía trên là lịch sử trước khi đổi caption/chunk/query representation. Tác vụ dài in phase, processed/total, %, elapsed và ETA khoảng mỗi 30 giây. Đây là inference, không phải training.

## Chạy text-based retrieval trên public test

Đã mở rộng ba module hiện có `query_image_captioning.py`, `query_embedding.py`, `query_retrieval.py` với `--split public_test`; không thêm file code. Giữ nguyên Qwen caption 384 token, BGE-M3, cosine, max chunk score và top 5 như dev; không chọn cấu hình bằng nhãn public test. Đây là inference, không fine-tune. Nhãn chỉ dùng đánh giá sau ranking.

Cần `.venv`, model Qwen/BGE-M3 local, GPU CUDA BF16 cho caption/embedding, corpus index đã hoàn tất, JSON `data/raw/VLSP2025-MLQA-TSR/dataset/public_test data/vlsp_2025_public_test.json` và ảnh `data/processed/images/public_test/`. Chạy từ root bằng PowerShell:

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split public_test
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split public_test --allow-incomplete-captions
.\.venv\Scripts\python.exe -u -m text_based_rag.query_retrieval --split public_test
```

Tất cả output public test nằm trong `artifacts/public-test/text_based_rag/`: caption trong `query_image_captioning/`, vector/mapping trong `query_embedding/`; retrieval chạy lần sau lưu trong `query_retrieval_corpus/` với `retrieval_results.jsonl`, `retrieval_preview.txt`, `per_query_metrics.jsonl`, `metrics.json`, `manifest.json` và `submission_task1.json` (JSON array theo định dạng evaluator chính thức).

Đã chạy ngày 15/10/2026: 100 query, 89 ảnh; embedding và retrieval `COMPLETED`. Macro precision@5 = **0,064**, recall@5 = **0,1433333333**, F1@5 = **0,0857647908**, F2@5 = **0,1115579034**; hit rate@5 = **26%**. Hàm `eval_retrieval` chính thức trả F2 = `0.11155790337054611`, khớp local trong sai số epsilon. Có 6 caption bị cắt ở 384 token, ảnh hưởng 6 query, giữ `INCOMPLETE`; 3 gold citation thiếu trong index (`36/2024/QH15` / `11.0`, `QCVN 41:2024/BGTVT` / `D.11`, `QCVN 41:2024/BGTVT` / `F5`) vẫn tính vào recall. Không tự sửa nhãn hoặc tinh chỉnh cấu hình theo kết quả test.

Mong đợi 100 prediction, mỗi query có 5 citation khác nhau trong corpus; 100 dòng metric và manifest retrieval `COMPLETED`. Đọc kết quả:

```powershell
Get-Content .\artifacts\public-test\text_based_rag\query_retrieval_corpus\metrics.json
Get-Content .\artifacts\public-test\text_based_rag\query_retrieval_corpus\retrieval_preview.txt -TotalCount 35
```

Caption resume ảnh đã hoàn tất; caption bị cắt giữ `INCOMPLETE` và được embedding chấp nhận qua cờ trên, có thể giảm chất lượng. Embedding từ chối thư mục đã tồn tại; nếu chỉ chạy lại retrieval, bỏ hai lệnh tạo đầu vào và thêm `--overwrite` vào lệnh retrieval. Cờ này thay thế các file trong query_retrieval_corpus/, giữ caption/vector và kết quả cũ ở thư mục gốc. Khi OOM, đóng ứng dụng GPU khác hoặc giảm embedding bằng `--batch-size 1`; hash/split/encoder không khớp cần kiểm tra nguồn, không bỏ validation. Trạng thái bàn giao giữ `AWAITING_USER_TEST` đến khi người dùng xác nhận.

## Tái tạo đầu vào khi cần

Caption/embedding cần GPU CUDA, model local Qwen3-VL-8B NF4/BGE-M3, ảnh trong `data/processed/images/train`. Chỉ chạy các bước sau khi cần tạo lại đầu vào; embedding từ chối output đã tồn tại, cần tự lưu bản cũ trước.

```powershell
$env:PYTHONPATH="src"
$env:PYTHONUTF8="1"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -u -m text_based_rag.query_image_captioning --split dev
.\.venv\Scripts\python.exe -u -m text_based_rag.query_embedding --split dev --allow-incomplete-captions
```

Caption resume ảnh đã hoàn tất. `--allow-incomplete-captions` cho phép dùng description không rỗng bị chạm giới hạn token, giữ trạng thái `INCOMPLETE`, không đánh dấu người dùng đã duyệt; bỏ cờ để yêu cầu mọi caption hoàn tất. Caption/embedding mặc định vẫn train; phải ghi `--split dev` như trên.

Đã đổi thư mục output sang `artifacts/experiments/`; caption/vector train nằm trong `query_image_captioning_train/` và `query_embedding_train/`. Đường dẫn trong manifest cũ là thông tin lịch sử. Khi cần tái tạo corpus, đặt biến môi trường như trên và chạy lần lượt qua `.\.venv\Scripts\python.exe -m`: `text_based_rag.corpus_normalization`, `text_based_rag.corpus_image_captioning`, `text_based_rag.corpus_chunking`, `text_based_rag.corpus_embedding index`. Các bước này không mặc định ghi đè output cũ; kiểm tra manifest và preview trước bước tiếp theo.

## Dữ liệu và source

Split: 418 train / 112 dev (seed 2025, nhóm theo ID ảnh); 100 public test có nhãn; 146 private test không có gold. Không dùng public/private để chọn cấu hình; sample submission không phải ground truth.

- `src/multimodal_legal_rag/data.py`: loader, validator và split.
- `src/text_based_rag/`: chuẩn hóa, caption, chunk, embedding.
- `src/text_based_rag/query_retrieval.py`: corpus retrieval dev/public test, example retrieval dev từ train và evaluator.
- `data/processed/`: ảnh/split; `models/`: model local; `artifacts/`: output chạy.

## Lỗi và giới hạn

- Import lỗi: dùng Python `.venv`, đặt `PYTHONPATH=src` và chạy từ root.
- Output đã tồn tại: dùng `--overwrite` nếu muốn thay thế hoặc tự lưu thư mục cũ trước.
- Hash/encoder/mapping không khớp: kiểm tra nguồn vector và split dev; không bỏ validation.
- Gold citation thiếu trong index: vẫn tính trong gold recall; không tự sửa nhãn.
- Thiếu model khi tạo đầu vào: tải BGE-M3 bằng `corpus_embedding download`, Qwen bằng `temp/download_vlm.py`, qua Python `.venv` trước inference.
- OOM khi caption/embedding: đóng ứng dụng GPU; embedding có thể giảm `--batch-size 1`.
- Caption `INCOMPLETE` có thể làm giảm chất lượng. BGE-M3 nhận text/caption, không nhận trực tiếp ảnh. Chọn cấu hình nhiều lần trên dev có thể quá khớp.

Không commit/phân phối lại dataset, weights, index, cache hoặc secrets; tôn trọng data agreement và giấy phép. Git chỉ rollback được file đã theo dõi. Nguồn: [tài liệu tham khảo](multimodal_legal_qa_vlsp2025_sources.md). Chờ người dùng kiểm tra trước khi chuyển trạng thái `COMPLETED`.
