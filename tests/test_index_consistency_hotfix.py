from dataclasses import dataclass
from pathlib import Path

import pytest
from backend.app.exceptions import BusinessException
from backend.app.indexing.consistency import IndexConsistencyService
from backend.app.indexing.index_version import IndexVersionManager
from backend.app.memory.manager import MemoryManager
from backend.app.memory.providers.memory_provider import InMemoryMemoryProvider
from backend.app.retrievers.base import RetrievedChunk
from backend.app.retrievers.metadata_filter import AutoMetadataFilterResult
from backend.app.retrievers.pipeline.context import RetrieverPipelineContext
from backend.app.retrievers.pipeline.steps.dense_retrieve_step import DenseRetrieveStep
from backend.app.retrievers.pipeline.steps.fusion_step import FusionStep
from backend.app.retrievers.pipeline.steps.soft_boost_step import SoftBoostStep
from backend.app.retrievers.pipeline.steps.sparse_retrieve_step import SparseRetrieveStep
from backend.app.retrievers.sparse import BM25IndexManager, SparseDocument
from backend.app.services.document import DocumentService


@dataclass
class FakeDocument:
    id: int = 9
    knowledge_base_id: int = 4


class FakeDocumentRepository:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.document = FakeDocument()
        self.deleted = False

    def get(self, document_id: int) -> FakeDocument | None:
        return self.document if document_id == self.document.id else None

    def delete(self, document: FakeDocument) -> None:
        self.order.append("postgres")
        self.deleted = True


class FakeQdrantStore:
    def __init__(self, order: list[str], fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.deleted_document_ids: list[int] = []

    def delete_by_document_id(self, document_id: int) -> dict:
        self.order.append("qdrant")
        if self.fail:
            raise BusinessException(50004, "Qdrant文档向量删除失败")
        self.deleted_document_ids.append(document_id)
        return {"matched_points": 2, "document_id": document_id}


class FakeBM25Manager:
    def __init__(self, order: list[str], fail: bool = False) -> None:
        self.order = order
        self.fail = fail
        self.removed_document_ids: list[int] = []

    def remove_document(self, document_id: int, save: bool = True) -> None:
        self.order.append("bm25")
        if self.fail:
            raise RuntimeError("bm25 failed")
        self.removed_document_ids.append(document_id)
        assert save is True


def _patch_delete_dependencies(
    monkeypatch,
    *,
    order: list[str],
    qdrant_fail: bool = False,
    bm25_fail: bool = False,
) -> tuple[FakeQdrantStore, FakeBM25Manager]:
    qdrant = FakeQdrantStore(order, qdrant_fail)
    bm25 = FakeBM25Manager(order, bm25_fail)
    monkeypatch.setattr(
        "backend.app.services.document.QdrantVectorStore",
        lambda: qdrant,
    )
    monkeypatch.setattr(
        "backend.app.services.document.get_bm25_index_manager",
        lambda: bm25,
    )
    return qdrant, bm25


def test_delete_document_removes_qdrant_vectors(monkeypatch) -> None:
    order: list[str] = []
    qdrant, _ = _patch_delete_dependencies(monkeypatch, order=order)
    repository = FakeDocumentRepository(order)

    DocumentService(repository).delete(9)

    assert qdrant.deleted_document_ids == [9]


def test_delete_document_removes_bm25_chunks(monkeypatch) -> None:
    order: list[str] = []
    _, bm25 = _patch_delete_dependencies(monkeypatch, order=order)
    repository = FakeDocumentRepository(order)

    DocumentService(repository).delete(9)

    assert bm25.removed_document_ids == [9]


def test_delete_index_cleanup_happens_before_postgres_delete(monkeypatch) -> None:
    order: list[str] = []
    _patch_delete_dependencies(monkeypatch, order=order)
    repository = FakeDocumentRepository(order)

    DocumentService(repository).delete(9)

    assert order[:3] == ["qdrant", "bm25", "postgres"]


def test_qdrant_delete_failure_keeps_postgres_document(monkeypatch) -> None:
    order: list[str] = []
    _patch_delete_dependencies(monkeypatch, order=order, qdrant_fail=True)
    repository = FakeDocumentRepository(order)

    with pytest.raises(BusinessException):
        DocumentService(repository).delete(9)

    assert repository.deleted is False
    assert order == ["qdrant"]


def test_bm25_delete_failure_keeps_postgres_document(monkeypatch) -> None:
    order: list[str] = []
    _patch_delete_dependencies(monkeypatch, order=order, bm25_fail=True)
    repository = FakeDocumentRepository(order)

    with pytest.raises(BusinessException):
        DocumentService(repository).delete(9)

    assert repository.deleted is False
    assert order == ["qdrant", "bm25"]


def test_successful_delete_removes_postgres_document(monkeypatch) -> None:
    order: list[str] = []
    _patch_delete_dependencies(monkeypatch, order=order)
    repository = FakeDocumentRepository(order)

    DocumentService(repository).delete(9)

    assert repository.deleted is True


def test_delete_bumps_knowledge_base_index_version(monkeypatch) -> None:
    order: list[str] = []
    _patch_delete_dependencies(monkeypatch, order=order)
    repository = FakeDocumentRepository(order)
    before = IndexVersionManager().get_version(4)

    DocumentService(repository).delete(9)

    assert IndexVersionManager().get_version(4) > before


def test_bm25_persistent_index_does_not_restore_deleted_document(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    manager = BM25IndexManager(index_path=index_path)
    manager.add_documents(
        [
            SparseDocument(id="9_0", text="old", document_id=9, knowledge_base_id=4),
            SparseDocument(id="12_0", text="new", document_id=12, knowledge_base_id=4),
        ],
        save=True,
    )

    manager.remove_document(9, save=True)
    reloaded = BM25IndexManager(index_path=index_path)
    reloaded.load()

    assert reloaded.list_document_ids() == [12]


def test_knowledge_search_cache_key_contains_kb_version() -> None:
    manager = MemoryManager(InMemoryMemoryProvider())
    arguments = {"query": "劳动法第二章", "knowledge_base_id": 401}

    key = manager.build_tool_cache_key("knowledge_search", arguments)

    IndexVersionManager().bump_version(401)
    assert manager.build_tool_cache_key("knowledge_search", arguments) != key


def test_old_cache_is_not_hit_after_version_bump() -> None:
    manager = MemoryManager(InMemoryMemoryProvider())
    arguments = {"query": "劳动法第二章", "knowledge_base_id": 402}

    manager.set_tool_cache("knowledge_search", arguments, {"answer": "old"})
    IndexVersionManager().bump_version(402)
    cached, _ = manager.get_tool_cache("knowledge_search", arguments)

    assert cached is None


class FakeVectorStoreForOrphans:
    def __init__(self) -> None:
        self.document_ids = [9, 12]
        self.deleted: list[int] = []

    def list_document_ids(self) -> list[int]:
        return list(self.document_ids)

    def delete_by_document_id(self, document_id: int) -> dict:
        self.deleted.append(document_id)
        self.document_ids = [item for item in self.document_ids if item != document_id]
        return {"document_id": document_id}


def test_orphan_detector_finds_qdrant_orphan(monkeypatch) -> None:
    service = IndexConsistencyService(vector_store=FakeVectorStoreForOrphans())
    monkeypatch.setattr(service, "_postgres_active_document_ids", lambda: [12])
    monkeypatch.setattr(
        "backend.app.indexing.consistency.get_bm25_index_manager",
        lambda: type("BM25", (), {"list_document_ids": lambda self: [12]})(),
    )

    result = service.detect_orphans()

    assert result["qdrant_orphan_document_ids"] == [9]


def test_orphan_detector_finds_bm25_orphan(monkeypatch) -> None:
    service = IndexConsistencyService(vector_store=FakeVectorStoreForOrphans())
    monkeypatch.setattr(service, "_postgres_active_document_ids", lambda: [9, 12])
    monkeypatch.setattr(
        "backend.app.indexing.consistency.get_bm25_index_manager",
        lambda: type("BM25", (), {"list_document_ids": lambda self: [7, 12]})(),
    )

    result = service.detect_orphans()

    assert result["bm25_orphan_document_ids"] == [7]


def test_orphan_cleanup_removes_qdrant_and_bm25(monkeypatch) -> None:
    vector_store = FakeVectorStoreForOrphans()
    service = IndexConsistencyService(vector_store=vector_store)
    bm25_removed: list[int] = []
    monkeypatch.setattr(service, "_postgres_active_document_ids", lambda: [12])
    monkeypatch.setattr(
        "backend.app.indexing.consistency.get_bm25_index_manager",
        lambda: type(
            "BM25",
            (),
            {
                "list_document_ids": lambda self: [9, 12],
                "remove_document": lambda self, document_id, save=True: bm25_removed.append(
                    document_id
                ),
            },
        )(),
    )

    result = service.cleanup_orphan(9)

    assert result["document_id"] == 9
    assert vector_store.deleted == [9]
    assert bm25_removed == [9]


def _chunk(document_id: int, score: float = 1.0, **metadata) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}_{metadata.get('chunk_index', 0)}",
        score=score,
        text="chunk",
        document_id=document_id,
        knowledge_base_id=4,
        chunk_index=metadata.get("chunk_index", 0),
        metadata=metadata,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query) -> list[RetrievedChunk]:
        return self.chunks


def _scoped_context() -> RetrieverPipelineContext:
    context = RetrieverPipelineContext(query="劳动法第二章", knowledge_base_id=4)
    context.auto_filter_result = AutoMetadataFilterResult(candidate_document_ids=[12])
    context.metadata["retrieval_scope"] = {
        "candidate_document_ids": [12],
        "dense_scope_applied": False,
        "sparse_scope_applied": False,
        "fusion_scope_guard_applied": False,
        "dense_rejected_count": 0,
        "sparse_rejected_count": 0,
        "fusion_rejected_count": 0,
    }
    return context


def test_dense_retrieval_respects_candidate_document_ids() -> None:
    context = _scoped_context()
    step = DenseRetrieveStep(dense_retriever=FakeRetriever([_chunk(9), _chunk(12)]))

    result = step.run(context)

    assert [chunk.document_id for chunk in result.dense_chunks] == [12]
    assert result.metadata["retrieval_scope"]["dense_rejected_count"] == 1


def test_sparse_retrieval_respects_candidate_document_ids() -> None:
    context = _scoped_context()
    step = SparseRetrieveStep(sparse_retriever=FakeRetriever([_chunk(9), _chunk(12)]))

    result = step.run(context)

    assert [chunk.document_id for chunk in result.sparse_chunks] == [12]
    assert result.metadata["retrieval_scope"]["sparse_rejected_count"] == 1


def test_fusion_scope_guard_rejects_disallowed_document() -> None:
    context = _scoped_context()
    context.dense_chunks = [_chunk(9), _chunk(12)]
    context.sparse_chunks = []

    result = FusionStep().run(context)

    assert [chunk.document_id for chunk in result.fused_chunks] == [12]
    assert result.metadata["retrieval_scope"]["fusion_rejected_count"] == 1


def test_no_candidate_document_ids_keeps_old_behavior() -> None:
    context = RetrieverPipelineContext(query="普通问题", knowledge_base_id=4)
    context.dense_chunks = [_chunk(9), _chunk(12)]
    context.sparse_chunks = []

    result = FusionStep().run(context)

    assert [chunk.document_id for chunk in result.fused_chunks] == [9, 12]


def test_structure_soft_boost_reports_matched_chunk_count() -> None:
    context = RetrieverPipelineContext(query="劳动法第二章说什么", knowledge_base_id=4)
    context.fused_chunks = [
        _chunk(12, score=1.0, chapter_number=2),
        _chunk(12, score=0.9, chapter_number=3, chunk_index=1),
    ]

    result = SoftBoostStep().run(context)

    metadata = result.metadata["structure_query_hint"]
    assert metadata["requested_chapter_number"] == 2
    assert metadata["matched_chunk_count"] == 1
    assert metadata["matched_chunk_ids"] == ["12_0"]
    assert metadata["applied"] is True
