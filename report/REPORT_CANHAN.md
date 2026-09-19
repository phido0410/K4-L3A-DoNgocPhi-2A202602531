# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đỗ Ngọc Phi — MSSV 2A202602531
**Nhóm:** 4ae (vai trò: trưởng nhóm, R1 · Data; chiến lược: `SentenceChunker`)
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding gần như **cùng hướng** (góc giữa chúng nhỏ, cosine gần 1), tức là mô hình cho rằng hai đoạn văn bản nói về cùng một ý hoặc cùng một chủ đề, bất kể chúng dùng từ ngữ gì hay dài ngắn ra sao.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên được nộp đơn phúc khảo trong vòng 7 ngày."
- Câu B: "Người học có thể yêu cầu chấm lại bài thi trong một tuần."
- Tại sao tương đồng: gần như không trùng từ nào ("phúc khảo" và "chấm lại", "7 ngày" và "một tuần") nhưng cùng nghĩa. Embedding tốt đặt chúng gần nhau vì nó mã hoá **ý nghĩa**, không so khớp từ.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Hạn nộp học phí là ngày 15 hằng tháng."
- Câu B: "Con mèo đang ngủ trên ghế sofa."
- Tại sao khác: hai câu thuộc hai chủ đề không liên quan (tài chính học vụ và đời sống thường ngày), nên vector gần như vuông góc và cosine xấp xỉ 0.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ đo **hướng** (tức ý nghĩa) và bỏ qua **độ lớn** của vector. Độ lớn thường phụ thuộc vào độ dài hay tần suất từ chứ không phải nội dung, nên một đoạn dài và một câu ngắn cùng ý vẫn được cosine xem là giống nhau, trong khi khoảng cách Euclid có thể coi chúng là xa. Khi vector đã chuẩn hoá (‖v‖ = 1) thì ‖a − b‖² = 2 − 2·cos(a, b), nên hai cách cho cùng thứ hạng. Vì vậy store dùng tích vô hướng (dot product) là đủ.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* bước nhảy = 500 − 50 = 450; số chunk = ⌈(10000 − 50) / 450⌉ = ⌈9950 / 450⌉ = ⌈22.11⌉
> *Đáp án:* **23 chunks**. Đã kiểm lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` và nhận đúng 23.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước nhảy giảm xuống 400, nên số chunk = ⌈9900 / 400⌉ = ⌈24.75⌉ = **25** (tăng 2 chunk, `FixedSizeChunker` cũng cho đúng 25). Overlap lớn giúp một câu hay một con số nằm ngay ranh giới vẫn xuất hiện **trọn vẹn** trong ít nhất một chunk và giữ ngữ cảnh nối giữa hai chunk. Mỗi thông tin cũng có thêm cơ hội lọt top-k. Cái giá phải trả là nhiều chunk hơn, tốn embedding hơn, và top-k dễ bị chiếm bởi các chunk gần trùng nhau.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách câu bằng regex `(?<=[.!?])\s+`, tức cắt tại khoảng trắng đứng **sau** dấu `.`, `!`, `?`. Nhờ lookbehind nên dấu câu được giữ lại trong câu, và regex bao được cả `". "`, `"! "`, `"? "`, `".\n"`. Sau đó strip từng câu, bỏ câu rỗng, rồi gom mỗi `max_sentences_per_chunk` câu thành một chunk nối bằng dấu cách.
> Edge case đã xử lý: text rỗng hoặc chỉ có khoảng trắng trả `[]`; text không có dấu câu trả về một chunk duy nhất; số như `3.5` hay `50.000` không bị cắt vì sau dấu chấm không có khoảng trắng.
> **Chưa xử lý được:** chữ viết tắt (`TS. Nguyễn`, `v.v. `) và số thứ tự đầu mục (`Điều 2. `, `1. `) vẫn bị cắt nhầm thành câu riêng. Với văn bản quy định có nhiều "Điều N." thì đây là điểm yếu thật.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thử separator theo thứ tự ưu tiên `\n\n` → `\n` → `". "` → `" "` → `""`, tức cắt ở ranh giới "to" (đoạn văn) trước và chỉ hạ xuống ranh giới nhỏ hơn khi mảnh vẫn quá dài. Thuật toán chạy theo **hai chiều**: tách bằng separator hiện tại (giữ separator dính vào mảnh để không mất dấu câu hay xuống dòng), **gom** các mảnh nhỏ liền kề cho tới sát `chunk_size`, còn mảnh nào vẫn dài hơn `chunk_size` thì **đệ quy** với danh sách separator còn lại.
> Có 3 base case: (1) text ≤ `chunk_size` thì trả nguyên; (2) hết separator (`separators=[]`) hoặc gặp `""` thì cắt cứng theo `chunk_size` ký tự; (3) separator không xuất hiện trong text thì chuyển sang separator kế tiếp. `chunk()` strip từng mảnh và bỏ mảnh rỗng.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi `Document` được chuẩn hoá thành một record `{id, content, metadata, embedding}` qua `_make_record` (1 Document = 1 record, store không tự chunk). Metadata được **copy** chứ không dùng chung object với người gọi, và luôn có `doc_id` (mặc định bằng `doc.id`). Record lưu trong list in-memory. Theo hướng dẫn của lab, mình bỏ nhánh ChromaDB vì không test nào cần.
> `search` nhúng câu hỏi rồi tính **tích vô hướng** với từng embedding. Vì embedding đã chuẩn hoá nên giá trị này đúng bằng cosine. Kết quả sắp giảm dần và lấy `top_k`, không trả `embedding` ra ngoài. Cả `search` và `search_with_filter` đều đi qua chung helper `_search_records`, nên hai hàm không thể lệch kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc **trước** rồi mới search: chỉ giữ các record có metadata khớp **tất cả** cặp key/value trong `metadata_filter`, sau đó chạy similarity trên tập ứng viên đó. Nếu search trước rồi mới lọc, top-k có thể bị tài liệu sai đối tượng chiếm hết và sau khi lọc chỉ còn lại 0 kết quả, dù store vẫn có chunk hợp lệ. Không truyền filter thì hàm hoạt động giống `search`.
> `delete_document` dựng lại list, bỏ mọi record có `metadata['doc_id'] == doc_id`, và trả `True` nếu kích thước giảm. Khi benchmark, mỗi chunk có `id="file#i"` còn `doc_id` trỏ về **tên file gốc**, nên một lần xoá sẽ gỡ toàn bộ chunk của file đó.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Có ba bước: truy xuất top-k (qua `search_with_filter`, có tham số tuỳ chọn `metadata_filter` để câu hỏi cần lọc `audience` cũng dùng được agent), dựng prompt, rồi gọi `llm_fn`. Prompt gồm chỉ dẫn "chỉ dùng ngữ cảnh bên dưới, không có thì nói không biết, trích dẫn [n], trả lời cùng ngôn ngữ với câu hỏi", tiếp theo là từng chunk được **đánh số** `[1] [2] [3]` kèm `doc_id` và score, cuối cùng là câu hỏi. Nhờ vậy câu trả lời **truy vết được** về đúng chunk và đúng file (tiêu chí Source Traceability).
> Nếu không truy xuất được chunk nào, agent trả câu thông báo cố định và **không gọi LLM**, để tránh LLM tự bịa câu trả lời khi không có căn cứ.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.11.4, pytest-9.1.1, pluggy-1.6.0 -- /Users/dophi/Desktop/K4-L3A-Data-Foundations/.venv/bin/python
rootdir: /Users/dophi/Desktop/K4-L3A-Data-Foundations
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.02s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

> Embedder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local), tính bằng `compute_similarity()`. Quy ước "cao" nếu cosine ≥ 0.5. Dự đoán được ghi **trước khi chạy**.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên được nộp đơn phúc khảo trong vòng 7 ngày. | Người học có thể yêu cầu chấm lại bài thi trong một tuần. | cao | 0.685 | Đúng |
| 2 | Hạn nộp học phí là ngày 15 hằng tháng. | Con mèo đang ngủ trên ghế sofa. | thấp | -0.004 | Đúng |
| 3 | Giảng viên phải nộp điểm trong 10 ngày sau khi thi. | Sinh viên phải nộp đơn phúc khảo trong 10 ngày sau khi có điểm. | cao | 0.924 | Đúng |
| 4 | Sinh viên được phép mang tài liệu vào phòng thi. | Sinh viên không được phép mang tài liệu vào phòng thi. | cao | 0.525 | Đúng |
| 5 | Thư viện mở cửa lúc 7 giờ sáng. | The library opens at 7 a.m. | cao | 0.948 | Đúng |

Đối chứng với `MockEmbedder`: 5 cặp lần lượt cho +0.003, +0.017, −0.034, +0.068, −0.093, đều xấp xỉ 0 bất kể nghĩa. Mock không mã hoá ngữ nghĩa nên không dùng được để benchmark.

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Dù cả 5 dự đoán cao/thấp đều đúng, **thứ hạng** thì bất ngờ. Cặp 3 (giảng viên nộp điểm và sinh viên nộp đơn phúc khảo) khác nghĩa, khác đối tượng và khác hành động, nhưng đạt **0.924**, cao hơn hẳn cặp 1 là cặp diễn đạt lại cùng nghĩa thật (0.685). Điều này cho thấy embedding bị chi phối mạnh bởi từ vựng và cấu trúc chung ("phải nộp… trong 10 ngày… điểm") hơn là bởi *ai* làm *gì*. Nó đo độ giống **chủ đề**, không đo "cùng đáp án". Đây chính là lý do corpus quy định cần `metadata_filter={"audience": "student"}`: không lọc thì chunk của giảng viên có thể thắng chunk của sinh viên.
> Hai điểm đáng chú ý khác: câu phủ định (cặp 4) vẫn đạt 0.525, tức mô hình chỉ "thấy" chữ "không" ở mức yếu; còn câu Việt–Anh cùng nghĩa (cặp 5) đạt 0.948, cho thấy mô hình đa ngữ đặt hai ngôn ngữ vào cùng một không gian nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

> Cấu hình: `SentenceChunker(max_sentences_per_chunk=3)` (116 chunk), embedder `paraphrase-multilingual-MiniLM-L12-v2`, `top_k=3`, corpus `data/khao-thi-phuc-khao/` (10 tài liệu). Agent trả lời theo kiểu trích xuất (in chunk top-1) vì không dùng LLM trả phí. Output đầy đủ: `ket_qua_benchmark.txt`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi muốn phúc khảo bài thi tự luận thì phải làm gì và trong thời hạn bao lâu? *(filter `audience=student`)* | `phuc-khao-nguoi-hoc#3`: "Đối với môn thi tự luận — c) đơn vị chủ quản thông báo kết quả phúc khảo trong vòng 07 ngày làm việc…" | 0.575 | Một phần: đúng tài liệu nhưng là thời hạn *nhận kết quả*. Đoạn "làm đơn… trong vòng 14 ngày làm việc" đứng hạng 2 | Chưa đúng: nói về thời hạn thông báo kết quả 07 ngày, thiếu hạn nộp đơn 14 ngày làm việc. (Bỏ filter thì top-3 không còn đoạn nào về phúc khảo của người học) |
| 2 | Thời lượng tối đa của một bài thi tự luận là bao nhiêu phút? | `hinh-thuc-thoi-luong-thi#6`: "c) thi tại phòng máy 50–150 phút. d) thi tự luận tối thiểu 50 phút, tối đa là 120 phút…" | 0.877 | Có | Đúng: tối đa 120 phút |
| 3 | Đến phòng thi muộn bao lâu thì không được dự thi? | `nguoi-hoc-du-thi#3`: "Trường hợp người học dự thi đến muộn quá 15 phút sau khi đã phát đề thi sẽ không được dự thi…" | 0.809 | Có | Đúng: đến muộn quá 15 phút sau khi đã phát đề |
| 4 | Hai giảng viên chấm tiểu luận lệch nhau từ 2 điểm trở lên thì xử lý thế nào? | `cham-thi#3`: Điều 21 b) "Cho hai cán bộ chấm thi thống nhất điểm… không thống nhất thì mời cán bộ chấm thi thứ ba…" | 0.647 | Không: đây là Điều 21 (chấm **tự luận**), còn đáp án nằm ở Điều 24 (chấm **tiểu luận**), không có trong top-3 | Sai: mô tả quy trình chấm tự luận (mời người chấm thứ ba) thay vì "thảo luận, không thống nhất thì báo CNBM" |
| 5 | Những lỗi vi phạm nào khiến người học bị đình chỉ thi? | `xu-ly-vi-pham-nguoi-hoc#8`: "c) Hành vi vi phạm và hình thức xử lý… phải được ghi chú trên danh sách người học dự thi…" (thủ tục) | 0.772 | Một phần: đúng Điều 29 nhưng là thủ tục. Danh sách lỗi đình chỉ thi ở hạng 2–3 và **đủ cả 3/3 ý**, riêng Sentence làm được điều này | Thiếu: top-1 không liệt kê lỗi. Ngữ cảnh top-3 đưa cho LLM thì đủ để trả lời |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 (Q4 không có). Điểm theo rubric `SCORING.md`: **6/10**. Nếu chỉ chấm theo `doc_id` thì là 5/5, nhưng con số đó thổi phồng kết quả.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Từ Đỗ Đức Đại: chunker theo Điều **gắn tiêu đề Điều vào từng mảnh**, nên phân biệt được Điều 21 (tự luận) và Điều 24 (tiểu luận), đúng chỗ chiến lược Sentence của tôi thất bại ở Q4. Từ Phạm Cường Quốc: chấm theo **đủ các ý chính** thay vì chỉ một chuỗi, vì một chuỗi như "14 ngày làm việc" có thể khớp nhầm vào đoạn nói về nộp muộn. Bài học từ vai trò Data của tôi: chất lượng corpus quyết định mọi thứ phía sau. Tách quy chế theo `audience` mới giúp filter có việc thật để làm, còn một dòng ghi nguồn để nhầm trong nội dung đã đủ làm hỏng kết quả của chiến lược Heading. Khi điều phối nhóm, tôi thấy số liệu tổng hợp chỉ đáng tin khi mọi người chạy cùng cấu hình và chỉ khác đúng một dòng `CHUNKER`.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | / 5 |
| Hướng tiếp cận của tôi (My Approach) | / 10 |
| Hoàn thiện code (Core Implementation — tests) | / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | / 5 |
| Kết quả truy xuất của tôi (Competition Results) | / 10 |
| **Tổng phần cá nhân** | **/ 60** |
