import math
import re
from collections import Counter

from backend.app.retrievers.sparse.base import (
    SparseDocument,
    SparseSearchQuery,
    SparseSearchResult,
)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: list[SparseDocument] = []
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.total_docs = 0

    def add_documents(self, documents: list[SparseDocument]) -> None:
        for document in documents:
            tokens = self._tokenize(document.text)
            term_freq = dict(Counter(tokens))

            self.documents.append(document)
            self.term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))

            for token in term_freq:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.total_docs = len(self.documents)
        total_length = sum(self.doc_lengths)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs else 0.0

    def search(self, query: SparseSearchQuery) -> list[SparseSearchResult]:
        query_tokens = self._tokenize(query.query)
        if not query_tokens or self.total_docs == 0:
            return []

        results: list[SparseSearchResult] = []
        for index, document in enumerate(self.documents):
            if not self._matches_filter(document, query):
                continue

            score = self._score_document(query_tokens, index)
            if score <= 0:
                continue

            results.append(
                SparseSearchResult(
                    id=document.id,
                    score=score,
                    text=document.text,
                    document_id=document.document_id,
                    knowledge_base_id=document.knowledge_base_id,
                    chunk_index=document.chunk_index,
                    metadata=document.metadata,
                )
            )

        return sorted(results, key=lambda result: result.score, reverse=True)[: query.top_k]

    def clear(self) -> None:
        self.documents.clear()
        self.doc_freq.clear()
        self.term_freqs.clear()
        self.doc_lengths.clear()
        self.avg_doc_length = 0.0
        self.total_docs = 0

    def _score_document(self, query_tokens: list[str], document_index: int) -> float:
        score = 0.0
        term_freq = self.term_freqs[document_index]
        doc_length = self.doc_lengths[document_index]

        for token in query_tokens:
            frequency = term_freq.get(token, 0)
            if frequency == 0:
                continue

            doc_frequency = self.doc_freq.get(token, 0)
            idf = math.log(1 + (self.total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator

        return score

    def _matches_filter(self, document: SparseDocument, query: SparseSearchQuery) -> bool:
        if query.knowledge_base_id is not None:
            if document.knowledge_base_id != query.knowledge_base_id:
                return False

        for key, value in (query.metadata_filter or {}).items():
            if document.metadata.get(key) != value:
                return False

        return True

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        current_word: list[str] = []

        for char in text.lower():
            if self._is_cjk(char):
                if current_word:
                    tokens.append("".join(current_word))
                    current_word.clear()
                tokens.append(char)
            elif re.match(r"[a-z0-9]", char):
                current_word.append(char)
            else:
                if current_word:
                    tokens.append("".join(current_word))
                    current_word.clear()

        if current_word:
            tokens.append("".join(current_word))

        return [token for token in tokens if token]

    def _is_cjk(self, char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"
