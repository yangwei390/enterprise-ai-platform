# Project State

## Current Goal

Enterprise AI Platform is an enterprise-grade RAG / AI platform.

Core direction:

- Enterprise knowledge base
- RAG retrieval and chat
- Multi-model LLM integration
- Future Agent / Memory / Workflow / MCP capabilities

## Completed Modules

- Parser
- Cleaner
- Chunker
- Embedding
- VectorStore(Qdrant)
- Retriever
- Reranker
- ContextBuilder
- PromptBuilder
- ChatService
- LLM Config
- LLM Factory
- LLM Provider
- LLM Client
- DashScope
- Guardrail
- Metadata Filtering
- Hybrid Search framework
- BM25Index
- BM25IndexManager
- DocumentIndexSynchronizer

## In Progress

- Replace DummySparseRetriever with BM25Retriever

## Next Steps

- Enable BM25 in HybridRetriever
- Switch `/chat` to HybridRetriever later

## Current Restrictions

- Do not modify unrelated root files.
- Do not modify ORM / Alembic unless explicitly requested.
- Do not modify ChatService / LLM / Prompt unless the task explicitly requires it.
- Do not change API paths.
- Do not hardcode API keys, passwords, or tokens.
- Do not commit `.env`.
- Do not overwrite an existing `.env`.

## Required Before Development

- Read `.ai` project documents before every development task.
- Follow `AGENTS.md` local collaboration rules.
- Keep changes scoped to the current request.
