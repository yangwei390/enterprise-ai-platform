# Architecture Summary

## DocumentPipeline

Current document parsing flow:

```text
Parser
-> Cleaner
-> Chunker
-> Embedding
-> VectorStore
-> BM25IndexStep
```

`VectorStore` currently writes to Qdrant.

`BM25IndexStep` runs after VectorStore succeeds.

## Indexing

`DocumentIndexSynchronizer` manages document index synchronization.

Current responsibility:

- Remove old BM25 chunks for the same document
- Add current chunks to BM25IndexManager
- Return BM25 sync metadata
- Do not fail the main parse flow if BM25 sync fails

Future responsibility:

- Qdrant delete / rebuild
- Document-level index rebuild
- Incremental index update

## Retrieval

Current retrieval modules:

- QdrantRetriever
- DenseRetriever
- BM25Retriever
- HybridRetriever
- RRF Fusion
- Metadata Filtering

Current hybrid status:

- Dense retrieval uses QdrantRetriever.
- Sparse retrieval framework exists.
- BM25Retriever is the target replacement for DummySparseRetriever.
- RRF Fusion is already separated for reuse.

## RAG Chat

Current RAG chat flow:

```text
ChatService
-> Retriever
-> Reranker
-> ContextBuilder
-> Guardrail
-> PromptBuilder
-> LLMFactory
-> Provider
-> Client
-> DashScope/OpenAI
```

Guardrail rule:

- If context is empty, do not call LLM.
- Return fixed answer based on knowledge-base unavailability.

## LLM

Current LLM architecture:

```text
LLMConfig
-> LLMFactory
-> Provider
-> Client
```

Rules:

- ChatService does not directly depend on OpenAI / DashScope SDK.
- Provider handles model-provider semantics.
- Client handles SDK / HTTP details.
- Configuration comes from `settings` and `LLMConfig`.
