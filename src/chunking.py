from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    # Split after ".", "!" or "?" when followed by whitespace (covers ". " and ".\n").
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s.strip() for s in self._SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
        size = self.max_sentences_per_chunk
        return [" ".join(sentences[i : i + size]) for i in range(0, len(sentences), size)]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: the piece already fits.
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Out of separators (or reached ""): hard-cut by characters.
        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator, next_separators = remaining_separators[0], remaining_separators[1:]
        if separator not in current_text:
            return self._split(current_text, next_separators)

        # Keep the separator attached so punctuation/newlines are not lost.
        parts = current_text.split(separator)
        pieces = [part + separator for part in parts[:-1]] + [parts[-1]]

        # Greedily merge small pieces; recurse into pieces that are still too long.
        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            if len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
                continue
            if buffer:
                chunks.append(buffer)
                buffer = ""
            if len(piece) <= self.chunk_size:
                buffer = piece
            else:
                chunks.extend(self._split(piece, next_separators))
        if buffer:
            chunks.append(buffer)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Same chunk_size for every strategy and no overlap, so only the split rule differs.
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            lengths = [len(c) for c in chunks]
            result[name] = {
                "count": len(chunks),
                "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
                "min_length": min(lengths, default=0),
                "max_length": max(lengths, default=0),
                "chunks": chunks,
            }
        return result
