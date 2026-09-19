# Tổng hợp kiến thức — Lab 07: Embedding, Chunking, Vector Store & RAG

Tài liệu ôn tập tóm lại những gì cần hiểu để làm lab, viết báo cáo và trả lời giảng viên khi demo. Mỗi phần đều kèm vị trí trong code của repo này.

---

## 0. Bức tranh một trang

```
file .md ──tách frontmatter──▶ metadata (audience, source_url, ...)  +  phần thân (body)
                                                                           │
                                                         Chunker.chunk(body)
                                                                           ▼
            Document(id="file#i", content=chunk, metadata={...frontmatter, "doc_id": "file"})
                                                                           │
                                                   EmbeddingStore.add_documents()
                                                   (mỗi chunk → 1 vector → 1 record)
                                                                           │
câu hỏi ──embed──▶ vector ──▶ LỌC metadata TRƯỚC ──▶ dot/cosine ──▶ top-k chunk
                                                                           │
                              prompt: [1] chunk, [2] chunk, [3] chunk + câu hỏi
                                                                           ▼
                                                 LLM ──▶ câu trả lời có trích dẫn [n]
```

**RAG (Retrieval-Augmented Generation)** gồm ba bước: tìm các đoạn liên quan trong kho tài liệu, nhét chúng vào prompt làm ngữ cảnh, rồi để LLM trả lời **dựa trên ngữ cảnh đó** thay vì dựa vào trí nhớ của nó. Chất lượng câu trả lời bị chặn trên bởi chất lượng truy xuất: nếu chunk đúng không lọt top-k thì LLM giỏi đến đâu cũng không trả lời đúng được.

---

## 1. Embedding

**Embedding** là hàm biến một đoạn text thành một vector số có số chiều cố định. Mô hình được huấn luyện sao cho các câu **gần nghĩa** cho ra các vector **gần hướng** nhau.

| Backend | Số chiều | Có ngữ nghĩa? | Khi nào dùng |
|---|---|---|---|
| `MockEmbedder` / `_mock_embed` | 64 | **Không**: băm MD5 rồi sinh số giả ngẫu nhiên | Chạy pytest, chạy offline |
| `LocalEmbedder` (`paraphrase-multilingual-MiniLM-L12-v2`) | 384 | Có, đa ngữ, hợp tiếng Việt | Benchmark miễn phí, không cần key |
| `OpenAIEmbedder` (`text-embedding-3-small`) | 1536 | Có | Có OpenAI key |
| `GeminiEmbedder` (`gemini-embedding-001`) | — | Có | Có Gemini key (free tier) |

- Chọn backend bằng `EMBEDDING_PROVIDER=mock|local|openai|gemini` trong `.env`. Chọn backend thật mà thiếu thư viện hoặc key thì code **tự quay về mock** mà không báo lỗi, nên luôn kiểm dòng `Embedding backend: ...` được in ra.
- **Mock chỉ đủ cho test cấu trúc.** Nếu benchmark bằng mock thì mọi con số đều là nhiễu: chunk đúng có thể bị điểm âm, còn chunk sai chủ đề lại lên top-1. Trong trường hợp đó phải ghi rõ trong báo cáo và phân tích bằng `count`, `avg_length` và độ mạch lạc của chunk thay vì điểm số.
- Nếu dùng OpenAI để benchmark, nên cache embedding theo hash nội dung để khi chạy lại không tốn thêm tiền.

---

## 2. Độ tương tự cosine

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)        giá trị trong [-1, 1]
```

- **≈ 1**: cùng hướng, tức cùng ý. **≈ 0**: không liên quan. **≈ -1**: ngược hướng (hiếm gặp với text embedding).
- Nếu một trong hai vector có độ dài 0 thì trả `0.0` để tránh chia cho 0 (`compute_similarity` trong `src/chunking.py`).
- **Cosine hay Euclid?** Cosine chỉ đo hướng, tức ý nghĩa, và bỏ qua độ lớn vector, vốn thường phụ thuộc vào độ dài text. Khi vector đã **chuẩn hoá** (‖v‖ = 1), ta có `‖a − b‖² = 2 − 2·cos(a, b)`, nên hai cách cho cùng thứ hạng và **dot product = cosine**. Đó là lý do `EmbeddingStore` xếp hạng bằng `_dot` cho gọn.

### Kết quả thí nghiệm của mình (bài 3.3)

Chạy bằng `paraphrase-multilingual-MiniLM-L12-v2`, so với mock:

| # | Cặp câu | Local | Mock | Bài học |
|---|---|---|---|---|
| 1 | "Sinh viên được nộp đơn phúc khảo trong vòng 7 ngày" và "Người học có thể yêu cầu chấm lại bài thi trong một tuần" | 0.685 | +0.003 | Khác từ, cùng nghĩa: embedding hiểu được diễn đạt lại |
| 2 | "Hạn nộp học phí là ngày 15 hằng tháng" và "Con mèo đang ngủ trên ghế sofa" | -0.004 | +0.017 | Không liên quan thì xấp xỉ 0 |
| 3 | "**Giảng viên** phải nộp **điểm** trong 10 ngày" và "**Sinh viên** phải nộp **đơn phúc khảo** trong 10 ngày" | **0.924** | -0.034 | **Khác nghĩa và khác đối tượng nhưng điểm cao nhất nhóm**, nên cần filter `audience` |
| 4 | "được phép mang tài liệu" và "**không** được phép mang tài liệu" | 0.525 | +0.068 | Phủ định chỉ kéo điểm xuống vừa phải |
| 5 | "Thư viện mở cửa lúc 7 giờ sáng" và "The library opens at 7 a.m." | 0.948 | -0.093 | Mô hình đa ngữ khớp được Việt–Anh |

**Kết luận:**
- Embedding đo **độ giống chủ đề và từ vựng**, không đo "cùng đáp án". Câu cùng khuôn chữ nhưng khác đối tượng có thể đạt điểm cao hơn cả câu diễn đạt lại cùng nghĩa (cặp 3 so với cặp 1).
- Mock cho mọi cặp xấp xỉ 0, nên không dùng mock để benchmark.

---

## 3. Chunking: chia nhỏ tài liệu

**Tại sao phải chunk?**
- Một vector cho cả file dài thì bị "loãng": nó đại diện cho mọi chủ đề trong file nên không khớp tốt với câu hỏi cụ thể nào.
- Ngữ cảnh của LLM có giới hạn, và nhét cả file vào prompt vừa tốn vừa gây nhiễu.
- Nhưng chunk **quá nhỏ** thì mất ngữ cảnh. Ví dụ chunk "trong 7 ngày" mà không có "phúc khảo" thì không trả lời được gì.

### 4 chiến lược

| Chiến lược | Cắt thế nào | Ưu | Nhược | Hợp với |
|---|---|---|---|---|
| `FixedSizeChunker` | Cửa sổ trượt `chunk_size` ký tự, lùi lại `overlap` | Đơn giản, kích thước đều, overlap giữ ngữ cảnh ranh giới | Cắt giữa câu, giữa từ, giữa con số | Text không có cấu trúc |
| `SentenceChunker` | Tách câu bằng regex `(?<=[.!?])\s+`, gom N câu | Không cắt giữa câu | Độ dài chunk không đều; cắt nhầm ở `TS.`, `v.v.`, `Điều 2.` | Văn xuôi, FAQ ngắn |
| `RecursiveChunker` | Thử `\n\n` → `\n` → `". "` → `" "` → `""`, gom mảnh nhỏ tới sát `chunk_size` | Tôn trọng đoạn văn và câu, kích thước gần đều | Không biết đâu là "mục" nếu mục dài | Mặc định an toàn cho Markdown |
| `HeadingChunker` (trong `bench.py`) | Mỗi heading (`#`, `## Điều 4`, `Điều 4.`) là một chunk; mục dài thì hạ xuống recursive | Mỗi chunk là một đơn vị ý trọn vẹn do người soạn chia sẵn | Mục quá ngắn hoặc quá dài; các mục cùng file có điểm gần bằng nhau | Quy định, sổ tay, văn bản theo Điều/Mục |

**Hai chi tiết quan trọng của `RecursiveChunker`:**
1. Thuật toán có **hai chiều**. Chiều xuống sâu: mảnh nào vẫn dài hơn `chunk_size` thì đệ quy với separator nhỏ hơn. Chiều gom lên: nối các mảnh nhỏ liền kề tới sát `chunk_size`. Thiếu bước gom thì một file nhiều dòng ngắn sẽ sinh ra hàng trăm chunk vụn 5–10 ký tự.
2. Có **3 base case**: text đã vừa `chunk_size`; hết separator (hoặc gặp `""`) thì cắt cứng; separator không có trong text thì thử separator kế tiếp.

**Chi tiết quan trọng của `HeadingChunker`:** khi phải cắt nhỏ một mục dài, cần **gắn lại tiêu đề vào từng mảnh con**. Nếu không, từ mảnh thứ hai trở đi sẽ mất ngữ cảnh "mục này nói về gì".

### Công thức số chunk (FixedSize)

```
số_chunk = ⌈(độ_dài − overlap) / (chunk_size − overlap)⌉
```

- Với 10.000 ký tự, size 500, overlap 50: ⌈9950/450⌉ = **23**.
- Tăng overlap lên 100: ⌈9900/400⌉ = **25**.
- Đã kiểm bằng code thật: `FixedSizeChunker(500, 50).chunk('a'*10000)` cho đúng 23.

**Đánh đổi của overlap:** overlap lớn giúp thông tin nằm ở ranh giới vẫn trọn vẹn trong ít nhất một chunk và có thêm cơ hội lọt top-k. Cái giá là nhiều chunk hơn, tốn embedding hơn, và top-k dễ bị chiếm bởi các chunk gần trùng nhau.

### So sánh chiến lược (`ChunkingStrategyComparator`)

`compare(text, chunk_size)` chạy cả 3 chiến lược có sẵn trên cùng text, cùng `chunk_size`, **không overlap** (để chỉ quy tắc cắt khác nhau). Kết quả trả về dạng `{fixed_size | by_sentences | recursive: {count, avg_length, min_length, max_length, chunks}}`. Nhớ **bỏ frontmatter** trước khi so sánh, nếu không bạn đang đo cả khối YAML.

---

## 4. Vector store (`src/store.py`)

**Record:** `{id, content, metadata, embedding}`, với quy ước **1 Document = 1 record**.
- Store **không tự chunk**. Việc chunk xảy ra bên ngoài (trong `bench.py`), mỗi chunk thành một `Document` riêng.
- `_make_record` **copy** metadata và luôn đặt `doc_id` (mặc định bằng `doc.id`).
- Hai loại id khác nhau:
  - `Document.id = "file#3"` là id của **chunk**.
  - `metadata["doc_id"] = "file"` trỏ về **file gốc**, nên `delete_document("file")` xoá mọi chunk của file.

**`search`:** nhúng câu hỏi, tính dot với mọi record, sắp giảm dần, lấy `top_k`, và không trả `embedding` ra ngoài.

**`search_with_filter`: lọc TRƯỚC rồi mới search.**

```
Store: 3 chunk faculty (điểm 0.9, 0.8, 0.7) và 2 chunk student (0.6, 0.5); top_k = 3; lọc audience=student

Lọc SAU  : top-3 = [faculty, faculty, faculty] → bỏ faculty → còn 0 kết quả  ✗
Lọc TRƯỚC: ứng viên = [student 0.6, student 0.5] → top-3 = 2 chunk student   ✓
```

- Cả `search` và `search_with_filter` đi chung helper `_search_records`, chỉ khác **tập ứng viên đầu vào**. Nhờ vậy hai hàm không thể lệch kết quả.
- Filter khớp **chính xác** mọi cặp key/value. Không truyền filter thì hoạt động giống `search`.

**ChromaDB:** repo có sẵn khung cho ChromaDB, nhưng lab hướng dẫn **bỏ hẳn** và chỉ dùng in-memory. Không test nào cần ChromaDB, và nếu máy chấm tình cờ có cài `chromadb` thì nhánh chưa làm xong sẽ làm sập 14 test.

---

## 5. Metadata: chìa khoá để lọc

**Trường bắt buộc (K4-L3A):** `doc_id`, `title`, `source_url`, `retrieved_at`, `document_version`, `audience`, cộng ít nhất một trường lọc khác (`department`, `category`, `language`...).

| Nhóm trường | Trường | Dùng để |
|---|---|---|
| Truy vết nguồn (provenance) | `source_url`, `retrieved_at`, `document_version` | Kiểm tra độ mới, truy câu trả lời về nguồn gốc. Nguồn không nêu phiên bản thì ghi `not-stated`, **không bịa** |
| Lọc | `audience` (`student`/`faculty`/`staff`/`all`), `category`, `department` | `search_with_filter` |

**Metadata phải khớp với chiều mình định lọc.** Nếu một trang gộp cả quy định cho sinh viên lẫn giảng viên mà lưu thành một file `audience: all`, thì filter `{"audience": "student"}` **không lọc được gì**, vì hai đáp án nằm chung một chunk hoặc một file. Cách sửa là **tách thành hai file**, mỗi file một `audience`.

Metadata phải được **trải vào mọi chunk**. Nếu chỉ gắn vào chunk đầu tiên thì filter sẽ bỏ sót các chunk còn lại.

**Đánh đổi của filter:** filter tăng precision (loại tài liệu sai đối tượng) nhưng có thể giảm recall. Ví dụ quy định chung ghi `audience: all` sẽ bị filter `student` loại mất, dù nó chứa thông tin cần thiết.

---

## 6. Agent RAG (`src/agent.py`)

```
answer(question, top_k=3, metadata_filter=None)
  1. chunks = store.search_with_filter(question, top_k, metadata_filter)
  2. không có chunk nào → trả câu thông báo, KHÔNG gọi LLM
  3. prompt = chỉ dẫn + ngữ cảnh đánh số + câu hỏi
  4. return llm_fn(prompt)
```

Cấu trúc prompt:

```
You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say that you do not know.
Cite the chunks you used, e.g. [1]. Answer in the same language as the question.

Context:
[1] (doc_id=quy-trinh-phuc-khao, score=0.812)
...nội dung chunk...

[2] (doc_id=..., score=...)
...

Question: ...
Answer:
```

- **Grounding (chống bịa):** yêu cầu "chỉ dùng ngữ cảnh" và "không có thì nói không biết".
- **Truy vết nguồn (traceability):** đánh số `[n]` kèm `doc_id`, để câu trả lời chỉ ra được chunk và file đã dùng.

---

## 7. Thu thập dữ liệu có trách nhiệm

1. **Trang mở công khai cho người đọc không đồng nghĩa với việc được phép truy cập tự động.** Phải kiểm `robots.txt`. Nếu bị cấm thì **đổi nguồn**, không tìm cách vượt qua.
2. Không đăng nhập, không vượt CAPTCHA, không gọi API riêng tư. Không đưa dữ liệu cá nhân hay tài liệu nội bộ vào repo.
3. Crawl chậm (≥ 1 giây giữa các request), đặt `User-Agent`, chỉ lấy 5–10 trang.
4. **Làm sạch trước khi chunk:** bỏ menu, footer, "Chuyển đến nội dung" và danh sách tin tức. Output thô đầy nhiễu, và nhiễu sẽ chiếm hết top-k.
5. Đọc lại từng file: công cụ fetch có thể **tự dịch sang tiếng Anh**. Tuyệt đối không tự thêm thông tin không có trong nguồn.
6. Crawler `scripts/fetch_public_pages.py` không xử lý PDF hay trang render bằng JS. Nó còn crash với `LookupError` khi server trả charset lạ; khi đó bỏ URL đó ra khỏi CSV và xử lý riêng.

---

## 8. Đánh giá chất lượng truy xuất

### 5 góc nhìn (`docs/EVALUATION.md`)

| Góc nhìn | Câu hỏi cần trả lời |
|---|---|
| Retrieval precision | Top-3 có chunk thật sự liên quan không? Điểm số có phân biệt được kết quả tốt và nhiễu không? |
| Chunk coherence | Chunk có giữ trọn một ý không, hay bị cắt giữa chừng? |
| Metadata utility | Filter có giúp tăng precision không, hay lọc quá tay làm mất kết quả tốt? |
| Grounding quality | Câu trả lời có dựa trên ngữ cảnh không, có chỉ ra được chunk nguồn không? |
| Data strategy impact | Bộ tài liệu và cách chunk có hợp với chủ đề và câu hỏi không? |

### Chấm hai mức: phát hiện đáng giá nhất của buổi lab

- **Mức doc_id (ngây thơ):** file gold có trong top-3 không. Cách này **thổi phồng kết quả**.
- **Mức nội dung:** mỗi câu hỏi khai báo một chuỗi đặc trưng của đáp án (ví dụ `"7 ngày"`), rồi kiểm chuỗi đó có thật trong ngữ cảnh truy xuất được không.
- Hai mức có thể lệch nhau. Chunker theo heading có thể lấy trọn 3 slot từ đúng file gold, vì các mục cùng file có điểm gần nhau, mà **không mục nào chứa đáp án**.

**Thang điểm mỗi câu hỏi:**

| Điểm | Điều kiện |
|---|---|
| 2 | Gold ở top-1 và ngữ cảnh chứa đáp án (và agent trả lời đúng) |
| 1 | Gold ở top-2/3, hoặc có chunk liên quan nhưng câu trả lời thiếu |
| 0 | Không có trong top-3, hoặc ngữ cảnh không trả lời được |

`bench.py` in cả hai mức và tổng điểm.

### A/B bắt buộc cho câu hỏi cần filter

Chạy câu hỏi đó hai lần, **có** và **không có** `metadata_filter`, trên mọi chiến lược. Nếu kết quả hai lần **giống hệt nhau** thì câu hỏi chưa thật sự cần filter. Khi đó phải sửa lại câu hỏi, hoặc tách lại file theo `audience`.

**Cách viết câu hỏi cần filter tốt:** chọn một câu hỏi **không nêu người hỏi là ai**, trong khi corpus có hai tài liệu cùng chủ đề, cùng từ vựng nhưng khác đối tượng và khác đáp án.

### Các kiểu lỗi thường gặp (failure analysis)

| Hiện tượng | Nguyên nhân | Hướng sửa |
|---|---|---|
| Chunk đúng chủ đề nhưng không có con số thắng chunk có đáp án | Cosine đo **độ giống chủ đề**, không đo mật độ thông tin trả lời được | Chunk nhỏ hơn hoặc theo mục; tăng `top_k`; rerank |
| Đúng file nhưng sai mục | Không overlap, nên mỗi thông tin chỉ có một cơ hội lọt top-k | Thêm overlap; gắn heading vào chunk |
| Trả lời nhầm đối tượng (giảng viên thay vì sinh viên) | Hai tài liệu cùng từ vựng, không lọc | `metadata_filter={"audience": ...}` |
| Filter loại mất thông tin cần | Tài liệu chung ghi `audience: all` | Filter dạng "student hoặc all", hoặc gắn nhiều giá trị |
| Top-k toàn menu và footer | Chưa làm sạch dữ liệu crawl | Làm sạch trước khi chunk |
| Điểm âm, kết quả vô lý | Đang dùng mock embedder | Bật embedder thật |

Mỗi failure case trong báo cáo cần đủ **3 phần**: câu hỏi nào hỏng, vì sao, và đề xuất sửa.

---

## 9. Bản đồ code

| File | Nội dung | Điểm then chốt |
|---|---|---|
| `src/models.py` | `Document(id, content, metadata)` | Có sẵn |
| `src/chunking.py` | 3 chunker, `compute_similarity`, comparator | Regex lookbehind giữ dấu câu; recursive hai chiều với 3 base case; chặn vector có độ dài 0 |
| `src/store.py` | `EmbeddingStore` in-memory | `doc_id` mặc định; lọc trước; chung helper `_search_records` |
| `src/agent.py` | `KnowledgeBaseAgent.answer` | Ngữ cảnh đánh số và trích dẫn; không có chunk thì không gọi LLM; hỗ trợ `metadata_filter` |
| `bench.py` | Benchmark của nhóm | Chỉ đổi dòng `CHUNKER`; điền `DATA_DIR`, `QUERIES`; tự chạy A/B và chấm 2 mức; ghi `ket_qua_benchmark.txt` |
| `main.py` | Demo thủ công | Nạp cả file làm 1 Document (không chunk), nên chỉ dùng để demo |

**Chạy:**

```bash
source .venv/bin/activate          # venv Python 3.11
pytest tests/ -v                   # 42 passed
python main.py "Chunking là gì?"   # demo từ đầu đến cuối
python bench.py                    # benchmark → ket_qua_benchmark.txt
```

**Dùng `bench.py` cho nhóm** (đã cấu hình sẵn cho corpus `data/khao-thi-phuc-khao/`):
1. `DATA_DIR` và `QUERIES` dùng chung cả nhóm. Mỗi câu hỏi có `gold_doc_id`, `must_contain` (**danh sách các ý chính** của gold answer) và `metadata_filter`.
2. Mỗi người chỉ đổi dòng `CHUNKER` sang chiến lược của mình (đã ghi chú TV1, TV2, TV3).
3. Bật embedder thật: `EMBEDDING_PROVIDER=local python bench.py`, hoặc thêm dòng đó vào `.env`.
4. Muốn agent trả lời bằng LLM thật thì đặt `LLM_PROVIDER=openai|gemini` và `LLM_MODEL=<tên model>`. Không đặt thì agent trả về chunk top-1.

---

## 10. Câu hỏi hay gặp khi demo và gợi ý trả lời

**"Chuyển sang chủ đề khác thì chiến lược nào còn dùng được?"**
Recursive dùng được ở hầu hết mọi nơi vì chỉ dựa vào đoạn văn và câu. Heading chỉ tốt khi văn bản có cấu trúc mục rõ ràng (quy định, sổ tay, tài liệu kỹ thuật). FixedSize là phương án an toàn cuối cùng cho text không có cấu trúc.

**"Metadata filter giúp ở đâu, làm mất kết quả ở đâu?"**
Filter giúp khi hai tài liệu cùng chủ đề nhưng khác đối tượng: nó loại tài liệu sai đối tượng nên tăng precision. Filter làm mất kết quả khi thông tin cần nằm trong tài liệu `audience: all` hoặc bị gán nhãn sai: recall giảm.

**"Vì sao lọc trước chứ không lọc sau?"**
Lọc sau có thể còn lại 0 kết quả, vì top-k đã bị tài liệu sai đối tượng chiếm hết (xem ví dụ ở mục 4).

**"Vì sao chấm theo doc_id là chưa đủ?"**
Đúng file không có nghĩa là đúng đoạn. Cần kiểm chuỗi đáp án có trong ngữ cảnh truy xuất được không.

**"Cosine cao có nghĩa là trả lời đúng không?"**
Không. Cosine đo độ giống chủ đề. Câu phủ định ("được phép" và "không được phép") hay câu cùng từ vựng nhưng khác đối tượng vẫn có thể cho điểm rất cao.

---

## 10b. Bài học thực tế từ phần nhóm (Khảo thí & phúc khảo, IUH)

Chi tiết và số liệu nằm trong `report/REPORT_NHOM.md` và `report/benchmark/*.txt`.

**Khi thu thập dữ liệu:**
- **`robots.txt` bị chặn theo User-Agent.** Tường lửa trả 403 cho UA mặc định `Python-urllib`, và `RobotFileParser.read()` hiểu 403 là "cấm tất cả". Phải đọc `robots.txt` bằng **đúng UA của crawler** thì mới biết thật sự được phép hay không. Không giả làm trình duyệt.
- **`CERTIFICATE_VERIFY_FAILED`** trên macOS có hai nguyên nhân:
  - Python chưa có bộ CA. Sửa bằng `SSL_CERT_FILE=$(python -m certifi)`.
  - Server không gửi chứng chỉ trung gian. Sửa bằng cách tải chứng chỉ trung gian từ địa chỉ "CA Issuers" ghi trong chứng chỉ rồi ghép vào bộ CA. **Không tắt** việc xác thực SSL.
- **PDF scan** không trích được chữ: phải OCR hoặc chép tay, và ghi `source_type` để người sau biết.
- **Bản tóm tắt trên web có thể lệch văn bản gốc.** Web ghi "14 ngày", PDF ghi "14 ngày **làm việc**". Luôn chọn văn bản gốc làm nguồn chính thức.
- Frontmatter do crawler sinh ra có ngoặc kép `doc_id: "..."` làm script CP2 báo lệch. Cần chuẩn hoá.

**Khi đánh giá (số liệu thật của nhóm, `top_k = 3`):**

| Cách chấm | Fixed | Recursive | Heading v1 | Heading v2 | Sentence |
|---|---|---|---|---|---|
| Theo `doc_id` | 10 | 10 | 10 | 10 | 10 |
| Đủ ý chính + đoạn liên quan ở top-1 | 5 | 6 | 5 | 7 | 6 |

- **Sentence** là chiến lược duy nhất giữ trọn danh sách lỗi đình chỉ thi (Q5, 3/3 ý), vì văn bản pháp quy viết cả danh sách thành một câu dài ngăn bằng `;`. Nhưng nó tách số thứ tự khoản ("2.") khỏi nội dung.

- **Đúng file nhưng sai Điều:** `cham-thi` có Điều 21 (tự luận) và Điều 24 (tiểu luận) với từ vựng gần như giống nhau. Chỉ chunker theo heading, nhờ gắn tên Điều vào chunk, mới đưa được Điều 24 vào top-3.
- **Metadata đặt nhầm vào nội dung làm hỏng embedding:** dòng ghi nguồn ~400 ký tự gộp vào chunk đầu làm Heading mất Q1. Bỏ dòng đó ra thì điểm từ 5 lên 7/10.
- **Filter:** Q1 từ 0/2 lên 2/2 khi có `audience=student`. Nhưng quy định "ra về sau 2/3 thời gian" nằm trong tài liệu cán bộ coi thi, nên bị filter loại hẳn, tăng `top_k` cũng không cứu được.
- **Danh sách dài** (điểm c, 841 ký tự) bị cắt qua 2 chunk, nửa sau mất nhãn "Đình chỉ thi". Hướng sửa: chunk theo khoản/điểm và gắn nhãn, hoặc dùng parent-document retrieval.
- **Mock embedder: 0/10** trên cùng bộ câu hỏi.

---

## 11. Checklist trước khi nộp (CP7)

- [ ] `pytest tests/ -v` cho 42 passed, không còn `raise NotImplementedError`.
- [ ] `data/<chu-de>/` có 5–10 file `.md` đủ metadata; `sources.csv` khớp 1-1; `audience` có ít nhất 2 giá trị.
- [ ] Có ít nhất 1 câu hỏi dùng `metadata_filter={"audience": "student"}`, và A/B cho thấy kết quả khác nhau.
- [ ] Ít nhất 1 thành viên chunk theo heading hoặc mục.
- [ ] `bench.py` và `ket_qua_benchmark.txt` đã commit.
- [ ] `REPORT_CANHAN.md` và `REPORT_NHOM.md` điền đủ; output pytest là thật.
- [ ] Repo tên `K4-DAY07-HoVaTen-MSSV`, không chứa `.venv/` hay `.env`, đã nộp link lên vlearn.
