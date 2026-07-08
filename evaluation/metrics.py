from typing import Any


def document_hit(expected_documents: list[str], sources: list[Any]) -> bool:
    if not expected_documents:
        return True

    source_names = {_source_name(source) for source in sources}
    source_names = {source for source in source_names if source}
    for expected_document in expected_documents:
        if any(
            expected_document in source or source in expected_document
            for source in source_names
        ):
            return True
    return False


def chunk_recall(expected_chunks: list[int], sources: list[Any]) -> float:
    if not expected_chunks:
        return 1.0

    actual_chunks = {
        chunk_index
        for chunk_index in (_source_chunk_index(source) for source in sources)
        if chunk_index is not None
    }
    matched_count = len(set(expected_chunks) & actual_chunks)
    return matched_count / len(set(expected_chunks))


def keyword_coverage(expected_keywords: list[str], answer: str) -> float:
    if not expected_keywords:
        return 1.0

    matched_count = sum(1 for keyword in expected_keywords if keyword in answer)
    return matched_count / len(expected_keywords)


def _source_name(source: Any) -> str | None:
    value = _get_value(source, "source")
    if value:
        return str(value)
    metadata = _get_value(source, "metadata")
    if isinstance(metadata, dict):
        metadata_source = metadata.get("source")
        return str(metadata_source) if metadata_source else None
    return None


def _source_chunk_index(source: Any) -> int | None:
    value = _get_value(source, "chunk_index")
    if isinstance(value, int):
        return value
    metadata = _get_value(source, "metadata")
    if isinstance(metadata, dict):
        metadata_value = metadata.get("chunk_index")
        return metadata_value if isinstance(metadata_value, int) else None
    return None


def _get_value(source: Any, key: str) -> Any | None:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)
