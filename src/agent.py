from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    NO_CONTEXT_ANSWER = "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        chunks = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        # Nothing retrieved: say so instead of letting the LLM answer without grounding.
        if not chunks:
            return self.NO_CONTEXT_ANSWER
        prompt = self._build_prompt(question, chunks)
        return self.llm_fn(prompt)

    @staticmethod
    def _build_prompt(question: str, chunks: list[dict]) -> str:
        # Number each chunk and show its doc_id so the answer can cite sources.
        context = "\n\n".join(
            f"[{i}] (doc_id={chunk['metadata'].get('doc_id', chunk['id'])}, "
            f"score={chunk['score']:.3f})\n{chunk['content']}"
            for i, chunk in enumerate(chunks, start=1)
        )

        return (
            "You are a helpful assistant. Answer the question using ONLY the context below.\n"
            "If the context does not contain the answer, say that you do not know.\n"
            "Cite the chunks you used, e.g. [1]. Answer in the same language as the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
