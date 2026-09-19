"""
Benchmark retrieval trên corpus của nhóm (Lab 07 — CP5/CP6).

Chạy:
    python bench.py                         # ghi kết quả ra ket_qua_benchmark.txt
    python bench.py --out ket_qua_khac.txt

Mỗi thành viên chỉ đổi DÒNG CHỌN CHUNKER (biến CHUNKER) — mọi thứ khác giữ nguyên
để so sánh giữa các thành viên là công bằng. DATA_DIR và QUERIES dùng chung cả nhóm.

Embedding backend đọc từ .env (EMBEDDING_PROVIDER=mock|local|openai|gemini).
Agent dùng LLM thật nếu đặt LLM_PROVIDER=openai|gemini và LLM_MODEL=<tên model>;
nếu không, agent trả lời trích xuất (in chunk top-1) để vẫn chạy được offline.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import GeminiEmbedder, LocalEmbedder, OpenAIEmbedder, _mock_embed


class HeadingChunker:
    """
    Chunk by document structure: one chunk per section ("## ..." heading or "Điều N" line).

    Sections longer than chunk_size are split further with RecursiveChunker, and the
    section heading is re-attached to every sub-chunk so each piece keeps its context.
    The preamble before the first section (document title, source note) is merged into
    the first chunk instead of becoming a chunk of its own with no answerable content.

    drop_notes=True (v2) removes "> ..." source-note lines from that preamble: provenance
    already lives in metadata, and a long note merged into the first chunk diluted its
    embedding enough to push the answer out of top-3 (failure case Q1 with v1).
    """

    HEADING = re.compile(r"^(#{2,6}\s+\S.*|Điều\s+\d+.*)$", re.MULTILINE)

    def __init__(self, chunk_size: int = 500, drop_notes: bool = True) -> None:
        self.chunk_size = chunk_size
        self.drop_notes = drop_notes

    def chunk(self, text: str) -> list[str]:
        starts = [m.start() for m in self.HEADING.finditer(text)]
        if not starts:
            return RecursiveChunker(chunk_size=self.chunk_size).chunk(text)

        preamble = text[: starts[0]].strip()
        if self.drop_notes:
            preamble = "\n".join(line for line in preamble.splitlines() if not line.startswith(">")).strip()
        chunks: list[str] = []
        for begin, end in zip(starts, starts[1:] + [len(text)]):
            section = text[begin:end].strip()
            if section:
                chunks.extend(self._split_section(section))
        if preamble:
            chunks[0] = f"{preamble}\n\n{chunks[0]}"
        return chunks

    def _split_section(self, section: str) -> list[str]:
        if len(section) <= self.chunk_size:
            return [section]
        heading, _, body = section.partition("\n")
        inner_size = max(self.chunk_size - len(heading) - 1, 100)
        return [f"{heading}\n{piece}" for piece in RecursiveChunker(chunk_size=inner_size).chunk(body)]


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Shared by the whole group: folder with the cleaned .md files (+ sources.csv).
DATA_DIR = Path("data/khao-thi-phuc-khao")

# The ONE line each member changes to their own strategy.
CHUNKER = SentenceChunker(max_sentences_per_chunk=3)  # TV4 Đỗ Ngọc Phi
# CHUNKER = FixedSizeChunker(chunk_size=500, overlap=100)  # TV1 Nguyễn Trường Bảo
# CHUNKER = RecursiveChunker(chunk_size=500)  # TV2 Phạm Cường Quốc
# CHUNKER = HeadingChunker(chunk_size=500)  # TV3 Đỗ Đức Đại (v2; drop_notes=False gives v1)

TOP_K = 3

# Shared by the whole group: exactly 5 queries from REPORT_NHOM section 3.
#   gold_doc_id     : doc_id (file stem) that holds the answer
#   must_contain    : the key facts of the gold answer, each a string that must appear
#                     in the retrieved context (content-level check — stricter than doc_id)
#   metadata_filter : e.g. {"audience": "student"}; queries with a filter are also
#                     run WITHOUT it (A/B) to show whether the filter helps.
QUERIES = [
    {
        # Needs the filter: the faculty document has the same wording ("phúc khảo",
        # "môn thi tự luận", "ngày làm việc") but a different deadline (05 days to grade).
        "query": "Tôi muốn phúc khảo bài thi tự luận thì phải làm gì và trong thời hạn bao lâu?",
        "gold_doc_id": "phuc-khao-nguoi-hoc",
        "must_contain": ["làm đơn phúc khảo", "giáo vụ Khoa/Viện", "trong vòng 14 ngày làm việc"],
        "metadata_filter": {"audience": "student"},
    },
    {
        "query": "Thời lượng tối đa của một bài thi tự luận là bao nhiêu phút?",
        "gold_doc_id": "hinh-thuc-thoi-luong-thi",
        "must_contain": ["tối đa là 120 phút"],
        "metadata_filter": None,
    },
    {
        "query": "Đến phòng thi muộn bao lâu thì không được dự thi?",
        "gold_doc_id": "nguoi-hoc-du-thi",
        "must_contain": ["đến muộn quá 15 phút sau khi đã phát đề thi"],
        "metadata_filter": None,
    },
    {
        "query": "Hai giảng viên chấm tiểu luận lệch nhau từ 2 điểm trở lên thì xử lý thế nào?",
        "gold_doc_id": "cham-thi",
        "must_contain": ["thảo luận để thống nhất kết quả", "báo CNBM xem xét quyết định"],
        "metadata_filter": None,
    },
    {
        "query": "Những lỗi vi phạm nào khiến người học bị đình chỉ thi?",
        "gold_doc_id": "xu-ly-vi-pham-nguoi-hoc",
        "must_contain": [
            "mang theo tài liệu hoặc phương tiện bị cấm",
            "đưa đề thi ra ngoài khu vực thi",
            "không chấp hành các yêu cầu của CBCT",
        ],
        "metadata_filter": None,
    },
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_markdown(text: str) -> tuple[dict, str]:
    """Split a .md file into (frontmatter metadata, body). Handles simple `key: value` YAML."""
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text

    metadata = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or key.startswith((" ", "#")):
            continue
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # drop inline comments
        metadata[key.strip()] = value.strip("\"'")
    return metadata, text[match.end() :]


def load_chunks(data_dir: Path, chunker) -> list[Document]:
    """Chunk every .md file outside the store; each chunk becomes its own Document."""
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        frontmatter, body = parse_markdown(path.read_text(encoding="utf-8"))
        for i, chunk in enumerate(chunker.chunk(body)):
            docs.append(
                Document(
                    id=f"{path.stem}#{i}",
                    content=chunk,
                    # Frontmatter goes on EVERY chunk so search_with_filter can use it;
                    # doc_id points at the source file, not the chunk.
                    metadata={**frontmatter, "doc_id": path.stem, "chunk_index": i},
                )
            )
    return docs


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def make_embedder():
    provider = os.getenv("EMBEDDING_PROVIDER", "mock").strip().lower()
    factories = {"local": LocalEmbedder, "openai": OpenAIEmbedder, "gemini": GeminiEmbedder}
    if provider in factories:
        try:
            return factories[provider]()
        except Exception as exc:
            print(f"[warn] {provider} embedder unavailable ({exc}); falling back to mock")
    return _mock_embed


def extractive_llm(prompt: str) -> str:
    """Offline stand-in for an LLM: return the first retrieved chunk from the prompt."""
    match = re.search(r"\[1\] \(.*?\)\n(.*?)(?:\n\n\[2\]|\n\nQuestion:)", prompt, re.DOTALL)
    return "[extractive] " + (match.group(1).strip() if match else "")


def make_llm():
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip()
    if not provider or not model:
        return extractive_llm, "extractive (no LLM_PROVIDER/LLM_MODEL set)"

    try:
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI()

            def llm(prompt: str) -> str:
                response = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content or ""

            return llm, f"openai:{model}"

        if provider == "gemini":
            from google import genai

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

            def llm(prompt: str) -> str:
                return client.models.generate_content(model=model, contents=prompt).text or ""

            return llm, f"gemini:{model}"
    except Exception as exc:
        print(f"[warn] LLM {provider} unavailable ({exc}); using extractive answers")
    return extractive_llm, "extractive (fallback)"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_query(results: list[dict], gold_doc_id: str, must_contain: list[str] | str) -> dict:
    """
    Two-level scoring (docs/SCORING.md).
      doc_level : gold doc_id appears in top-k (the naive check — tends to over-score).
      content   : ALL key facts of the gold answer appear in the retrieved context.
      points    : 2 = all facts present and the top-1 CHUNK itself holds a key fact;
                  1 = some facts present (answer incomplete, or the relevant chunk is
                      not top-1); 0 = none.
    """
    facts = [must_contain] if isinstance(must_contain, str) else list(must_contain)
    doc_ids = [r["metadata"].get("doc_id") for r in results]
    normalize = lambda text: text.replace("\n", " ").lower()  # noqa: E731
    context = normalize(" ".join(r["content"] for r in results))
    found = sum(fact.lower() in context for fact in facts)
    top1_relevant = bool(results) and any(fact.lower() in normalize(results[0]["content"]) for fact in facts)

    if found == len(facts) and top1_relevant:
        points = 2
    elif found:
        points = 1
    else:
        points = 0
    return {
        "doc_level": gold_doc_id in doc_ids,
        "content": found == len(facts),
        "facts": f"{found}/{len(facts)}",
        "points": points,
    }


def format_results(results: list[dict]) -> list[str]:
    lines = []
    for rank, r in enumerate(results, start=1):
        preview = r["content"][:110].replace("\n", " ")
        lines.append(f"    {rank}. score={r['score']:.3f}  id={r['id']}  | {preview}")
    return lines or ["    (no results)"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(out_path: Path) -> int:
    load_dotenv(override=False)
    embedder = make_embedder()
    llm, llm_name = make_llm()

    docs = load_chunks(DATA_DIR, CHUNKER)
    if not docs:
        print(f"No .md files found in {DATA_DIR}")
        return 1

    store = EmbeddingStore(collection_name="bench", embedding_fn=embedder)
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)

    lengths = [len(d.content) for d in docs]
    files = sorted({d.metadata["doc_id"] for d in docs})
    lines = [
        "=== Benchmark config ===",
        f"data_dir   : {DATA_DIR}  ({len(files)} files)",
        f"chunker    : {CHUNKER.__class__.__name__} {vars(CHUNKER)}",
        f"embedder   : {getattr(embedder, '_backend_name', 'mock embeddings fallback')}",
        f"llm        : {llm_name}",
        f"chunks     : {store.get_collection_size()}  (avg {sum(lengths) / len(lengths):.0f} chars, "
        f"min {min(lengths)}, max {max(lengths)})",
        "",
    ]

    total_points = doc_hits = content_hits = 0
    for number, q in enumerate(QUERIES, start=1):
        results = store.search_with_filter(q["query"], top_k=TOP_K, metadata_filter=q["metadata_filter"])
        verdict = score_query(results, q["gold_doc_id"], q["must_contain"])
        total_points += verdict["points"]
        doc_hits += verdict["doc_level"]
        content_hits += verdict["content"]

        lines.append(f"--- Q{number}: {q['query']}")
        lines.append(f"    gold={q['gold_doc_id']}  must_contain={q['must_contain']!r}  filter={q['metadata_filter']}")
        lines.extend(format_results(results))
        lines.append(
            f"    -> doc-level hit: {verdict['doc_level']} | key facts: {verdict['facts']} "
            f"| points: {verdict['points']}/2"
        )

        if q["metadata_filter"]:
            unfiltered = store.search(q["query"], top_k=TOP_K)
            ab = score_query(unfiltered, q["gold_doc_id"], q["must_contain"])
            lines.append("    [A/B] same query WITHOUT metadata_filter:")
            lines.extend(format_results(unfiltered))
            lines.append(f"    -> points without filter: {ab['points']}/2")

        answer = agent.answer(q["query"], top_k=TOP_K, metadata_filter=q["metadata_filter"])
        lines.append(f"    agent: {answer[:300].replace(chr(10), ' ')}")
        lines.append("")

    n = len(QUERIES)
    lines += [
        "=== Summary ===",
        f"doc-level hits (naive) : {doc_hits}/{n}",
        f"all key facts found    : {content_hits}/{n}",
        f"retrieval points       : {total_points}/{2 * n}",
    ]

    report = "\n".join(lines)
    print(report)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("ket_qua_benchmark.txt"))
    raise SystemExit(run(parser.parse_args().out))
