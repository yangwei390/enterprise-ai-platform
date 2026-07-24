# RAG Provider 全生命周期可插拔架构

## 1. 当前架构事实

当前 `enterprise-ai-platform` 已经具备一套本地 RAG 全生命周期链路。

本地知识库模型：

- `backend/app/models/knowledge_base.py`：`KnowledgeBase` 当前包含 `name`、`description`、`embedding_model`、`vector_store`，没有 `rag_provider` 字段。
- `backend/app/models/document.py`：`Document` 当前包含 `knowledge_base_id`、文件信息、`parse_status`、`parse_message`、`document_metadata`、`chunk_count`，没有外部 Provider 映射字段。

本地文档处理链路：

- `backend/app/api/document.py` 暴露上传、解析、查询、删除接口。
- `backend/app/services/document.py` 负责文档上传、解析状态更新、删除时清理 Qdrant 和 BM25。
- `backend/app/pipeline/document_pipeline.py` 串联 `ParserStep -> CleanerStep -> DocumentIdentityStep -> ChunkStep -> EmbeddingStep -> VectorStoreStep -> BM25IndexStep`。
- `backend/app/services/knowledge_base_reindex.py` 对一个知识库下的本地文档重新运行 `DocumentPipeline`。

本地检索和 Chat 链路：

- `backend/app/retrievers/pipeline/pipeline.py` 当前默认执行 `QueryUnderstandingStep -> QueryRewriteStep -> MetadataFilterStep -> DocumentRoutingStep -> RetrievalPlanningStep -> StrategySelectionStep -> DenseRetrieveStep -> SparseRetrieveStep -> FusionStep -> SoftBoostStep -> RerankStep -> MMRStep -> NeighborExpansionStep -> ContextBuildStep -> ContextCompressionStep`。
- `backend/app/rag/chat_pipeline.py` 当前直接调用 `RetrieverPipeline`，然后执行 `PromptBuilderFactory` 和 `LLMFactory`。
- `backend/app/chat/service.py` 普通非工具 Chat 路径调用 `RagChatPipeline`；工具路径单独调用 `HybridRetriever`、`RerankerFactory`、`MMRStep`、`NeighborExpansionStep`、`BasicContextBuilder`、Context Compression、Prompt 和 LLM；流式 Chat 路径直接调用 `RetrieverPipeline`。
- `evaluation/v2/targets/rag.py` 当前直接调用 `RetrieverPipeline` 或 `RagChatPipeline`，尚无 provider 参数。
- 当前 Local `RetrieverPipeline` 已包含 `ContextBuildStep` 和 `ContextCompressionStep`，这是现有实现事实，不代表目标 `RetrievalProvider` 契约。

当前没有实现：

- `KnowledgeLifecycleProvider` 抽象。
- `RetrievalProvider` 抽象。
- `LocalKnowledgeProvider` / `RagflowKnowledgeProvider` 聚合实现。
- RAGFlow client、adapter、provider。
- 知识库级 provider 绑定。
- 本地文档与 RAGFlow dataset/document 的映射。
- RAGFlow 状态同步、失败重试、删除同步。

因此，本文档是后续接入的架构规范，不代表当前代码已经实现这些能力。

## 2. 目标与非目标

### 2.1 目标

建立 Local RAG 与 RAGFlow 的全生命周期可插拔架构：

```text
KnowledgeProvider
├── LocalKnowledgeProvider
│   ├── lifecycle: LocalKnowledgeLifecycleProvider
│   └── retrieval: LocalRetrievalProvider
└── RagflowKnowledgeProvider
    ├── lifecycle: RagflowKnowledgeLifecycleProvider
    └── retrieval: RagflowRetrievalProvider
```

目标能力：

- 默认使用 Local Provider。
- 现有知识库默认视为 `local`。
- Local 与 RAGFlow 是两套独立知识库后端。
- 上层知识库、文档和 Chat API 保持统一。
- `LocalKnowledgeProvider` 和 `RagflowKnowledgeProvider` 是按后端聚合的门面。
- 每个聚合 Provider 同时组合自己的 lifecycle 和 retrieval 实现。
- Provider 绑定发生在聚合 `KnowledgeProvider` 层。
- `KnowledgeProviderFactory` 根据知识库绑定的 provider 返回完整聚合 Provider。
- Chat 检索通过已选择的聚合 Provider 访问其 `retrieval` 能力。
- 不允许 Local lifecycle 与 RAGFlow retrieval 被错误组合。
- Lifecycle Provider 负责自己的知识库、文档、Chunk、索引、状态、删除生命周期。
- Retrieval Provider 负责自己的检索生命周期，并只返回统一检索结果。
- 平台公共层负责 Conversation、Memory、Prompt、LLM、Sources、Citations、ChatResponse 等统一能力。
- RAGFlow API 细节在下一阶段根据实际部署版本确认。

### 2.2 非目标

本阶段不做：

- 不迁移现有 Local 知识。
- 不自动同步 Local 与 RAGFlow。
- 不双写 Local 和 RAGFlow。
- 不共享底层索引。
- 不直接访问 RAGFlow 数据库、Elasticsearch、Infinity 或数据卷。
- 不修改数据库 schema。
- 不修改业务代码。
- 不部署 RAGFlow。
- 不确认具体 RAGFlow API 路径和字段。

## 3. 责任矩阵

| 能力 | Local Lifecycle Provider | RAGFlow Lifecycle Provider | Local Retrieval Provider | RAGFlow Retrieval Provider | 平台公共层 |
|---|---|---|---|---|---|
| 本地业务知识库记录 | 间接使用 | 间接使用 | 不负责 | 不负责 | 负责 |
| Provider 绑定 | 不负责 | 不负责 | 不负责 | 不负责 | 负责 |
| 本地文档保存 | 负责 | 不负责 | 不负责 | 不负责 | 编排 |
| RAGFlow Dataset 生命周期 | 不负责 | 负责 | 不负责 | 不负责 | 编排和保存映射 |
| RAGFlow 文档上传 | 不负责 | 负责 | 不负责 | 不负责 | 编排和保存映射 |
| 文档解析 | 负责 | 负责 | 不负责 | 不负责 | 不直接处理 |
| 文档清洗 | 负责 | RAGFlow 内部负责 | 不负责 | 不负责 | 不直接处理 |
| 文档切片 | 负责 | 负责 | 不负责 | 不负责 | 不直接处理 |
| Embedding | 负责 | 负责 | 不负责 | 不负责 | 不直接处理 |
| Qdrant 向量索引 | 负责 | 不直接访问 | 读取 | 不负责 | 不直接访问 |
| BM25 关键词索引 | 负责 | 不直接访问 | 读取 | 不负责 | 不直接访问 |
| RAGFlow 向量索引 | 不负责 | 负责 | 不负责 | RAGFlow 内部读取 | 不直接访问 |
| RAGFlow 全文/关键词索引 | 不负责 | 负责 | 不负责 | RAGFlow 内部读取 | 不直接访问 |
| Dense Retrieve | 不负责 | 不负责 | 负责 | RAGFlow 内部负责 | 不直接处理 |
| Sparse Retrieve | 不负责 | 不负责 | 负责 | RAGFlow 内部负责 | 不直接处理 |
| Fusion | 不负责 | 不负责 | 负责 | RAGFlow 内部负责 | 不直接处理 |
| Reranker | 不负责 | 不负责 | 负责 | 可选由 RAGFlow 内部负责 | 不直接处理 |
| MMR | 不负责 | 不负责 | 负责 | 不再执行本地 MMR | 不直接处理 |
| Neighbor Expansion | 不负责 | 不负责 | 负责 | 不再执行本地邻居扩展 | 不直接处理 |
| Context 构建或压缩 | 不负责 | 不负责 | 不负责最终 Context | 不负责最终 Context | 唯一负责最终统一处理 |
| PromptBuilder | 不负责 | 不负责 | 不负责 | 不负责 | 负责 |
| Conversation | 不负责 | 不负责 | 不负责 | 不负责 | 负责 |
| Memory | 不负责 | 不负责 | 不负责 | 不负责 | 负责 |
| LLMFactory | 不负责 | 不负责 | 不负责 | 不负责 | 负责 |
| Sources / Citations | 不负责 | 不负责 | 输出 chunks 原始材料 | 输出 chunks 原始材料 | 统一封装 |
| ChatResponse | 不负责 | 不负责 | 不负责 | 不负责 | 负责 |
| 删除和重新索引 | 负责本地资源 | 负责 RAGFlow 资源 | 不负责 | 不负责 | 编排，不跨 Provider 静默代偿 |

## 4. Local 完整文档处理和检索流程

### 4.1 文档上传

当前 Local 上传流程：

```text
POST /documents/upload
  -> DocumentService.upload_document()
  -> LocalStorageService.save_upload_file()
  -> documents 记录 status=uploaded, parse_status=pending
```

Local Provider 负责保存本地文件，并由本地 `documents.storage_path` 指向上传文件位置。

### 4.2 文档解析和索引

当前 Local 解析流程：

```text
POST /documents/{id}/parse
  -> DocumentService.parse_document()
  -> parse_status=processing
  -> DocumentPipeline
      -> ParserStep
      -> CleanerStep
      -> DocumentIdentityStep
      -> ChunkStep
      -> EmbeddingStep
      -> VectorStoreStep
      -> BM25IndexStep
  -> persist document_identity
  -> parse_status=success / failed
  -> IndexVersionManager.bump_version()
```

Local Provider 拥有：

- Parser / Cleaner / DocumentIdentity。
- ChunkStrategyRouter / ChunkerFactory / 具体 Chunker。
- EmbeddingFactory。
- QdrantVectorStore，当前 collection 为 `enterprise_ai_chunks`。
- BM25IndexManager，当前持久化路径为 `data/bm25/index.pkl`。
- IndexVersionManager。

### 4.3 本地检索

Local 检索由 `RetrieverPipeline` 负责：

```text
QueryUnderstanding
  -> QueryRewrite
  -> MetadataFilter
  -> DocumentRouting
  -> RetrievalPlanning
  -> StrategySelection
  -> DenseRetrieve
  -> SparseRetrieve
  -> Fusion
  -> SoftBoost
  -> Rerank
  -> MMR
  -> NeighborExpansion
  -> ContextBuild
  -> ContextCompression
```

Local Retrieval Provider 返回给平台公共层的结果必须保持统一检索结构：

- `provider`
- `query`
- `chunks`
- `metadata`
- `latency_ms`

目标 Provider 契约中，Local Retrieval Provider 不返回最终 `context_text`。当前 `RetrieverPipeline` 已包含 Context Build/Compression，是现有实现事实；后续 Local Provider 落地时，需要在适配层取得检索完成后的 chunks，或明确采用过渡适配，但不能把过渡实现写成最终架构。

### 4.4 删除和重新索引

当前 Local 删除流程：

```text
DocumentService.delete()
  -> QdrantVectorStore.delete_by_document_id()
  -> BM25IndexManager.remove_document()
  -> IndexVersionManager.bump_version()
  -> soft delete document
```

当前 Local 重新索引流程：

```text
KnowledgeBaseReindexService.reindex()
  -> 对知识库下 active documents 逐个运行 DocumentPipeline
  -> 更新 parse_status
  -> 成功后 bump index version
```

RAGFlow Provider 不得复用这些本地删除和重建索引实现；它必须通过 RAGFlow HTTP API 管理自己的远端资源。

## 5. RAGFlow 完整文档处理和混合检索流程

RAGFlow Provider 的目标职责是代理 RAGFlow 的全生命周期能力。

### 5.1 Dataset 生命周期

RAGFlow Provider 负责：

- 创建 Dataset。
- 查询 Dataset。
- 更新 Dataset 配置。
- 删除 Dataset。
- 保存本地 `knowledge_base_id` 与远端 `dataset_id` 的映射。

平台只保存映射和状态，不直接读写 RAGFlow 内部数据库。

### 5.2 文档上传和解析

RAGFlow Provider 负责：

- 上传文档到指定 RAGFlow Dataset。
- 触发或等待 RAGFlow 文档解析。
- 查询 RAGFlow 文档解析状态。
- 记录远端 `document_id`、解析状态、同步错误。
- 提供失败重试入口。

RAGFlow Provider 负责的解析链路包括：

```text
RAGFlow document upload
  -> RAGFlow parser
  -> RAGFlow chunking
  -> RAGFlow embedding
  -> RAGFlow vector index
  -> RAGFlow full-text / keyword index
```

### 5.3 RAGFlow 检索

RAGFlow Provider 正式主链路必须使用 RAGFlow Retrieval API 返回 chunks。

Enterprise AI Platform 使用自己的 Context、Prompt、Memory 和 LLM 生成最终答案。

RAGFlow Chat API 不进入本期正式主链路。如需使用 RAGFlow Chat API，只能作为未来独立实验能力，并且启用前必须新增架构决策。

RAGFlow 内部负责：

- 向量检索。
- 全文/关键词检索。
- 混合检索和融合。
- 可选 Reranker。
- Chunk 管理和引用原始结果。

### 5.4 RAGFlow 返回 chunks 后的本地禁用项

当请求已经选择 RAGFlow Provider，并且 RAGFlow 已经返回 chunks 或引用材料后，平台不得再执行本地：

- `DenseRetrieveStep`
- `SparseRetrieveStep`
- `FusionStep`
- `RerankStep`
- `MMRStep`
- `NeighborExpansionStep`

原因：

- RAGFlow 与 Local 使用不同索引、不同 chunk schema、不同 scoring 语义。
- 本地检索步骤依赖 Qdrant/BM25 和本地 metadata。
- 二次执行本地检索会混淆 Provider 责任边界。

平台可以继续执行：

- Context 构建或压缩。
- PromptBuilder。
- Conversation。
- Memory。
- LLMFactory。
- Sources / Citations 标准化。
- ChatResponse 封装。

## 6. Knowledge Provider 协议草案

后续代码层建议建立分层抽象；本阶段只作为规范。

`KnowledgeProvider` 是按知识库后端聚合的门面，不能画成 `KnowledgeLifecycleProvider` 和 `RetrievalProvider` 的父类树，也不能返回包含生成层字段的万能结果。它只能组合同一后端的生命周期和检索能力，并为每项操作使用独立请求/结果模型。

聚合结构固定为：

```text
KnowledgeProvider
├── LocalKnowledgeProvider
│   ├── lifecycle: LocalKnowledgeLifecycleProvider
│   └── retrieval: LocalRetrievalProvider
└── RagflowKnowledgeProvider
    ├── lifecycle: RagflowKnowledgeLifecycleProvider
    └── retrieval: RagflowRetrievalProvider
```

`KnowledgeLifecycleProvider` 与 `RetrievalProvider` 是两个独立协议，不是 `KnowledgeProvider` 的子类。`KnowledgeProviderFactory` 根据知识库绑定的 provider 返回完整聚合 Provider；Chat 检索通过已选择的聚合 Provider 访问其 `retrieval` 能力。

### 6.1 生命周期 Provider

`KnowledgeLifecycleProvider` 负责知识库生命周期、文档生命周期、Chunk 管理、状态、删除和重建类操作。

```python
class KnowledgeLifecycleProvider:
    name: str

    def create_knowledge_base(self, request):
        raise NotImplementedError

    def get_knowledge_base(self, request):
        raise NotImplementedError

    def update_knowledge_base(self, request):
        raise NotImplementedError

    def delete_knowledge_base(self, request):
        raise NotImplementedError

    def upload_document(self, request):
        raise NotImplementedError

    def get_document(self, request):
        raise NotImplementedError

    def list_documents(self, request):
        raise NotImplementedError

    def start_document_processing(self, request):
        raise NotImplementedError

    def get_document_status(self, request):
        raise NotImplementedError

    def retry_document_processing(self, request):
        raise NotImplementedError

    def cancel_document_processing(self, request):
        raise NotImplementedError

    def delete_document(self, request):
        raise NotImplementedError

    def list_chunks(self, request):
        raise NotImplementedError

    def create_chunk(self, request):
        raise NotImplementedError

    def update_chunk(self, request):
        raise NotImplementedError

    def delete_chunk(self, request):
        raise NotImplementedError

    def reindex(self, request):
        raise NotImplementedError

    def health_check(self):
        raise NotImplementedError

    def get_capabilities(self):
        raise NotImplementedError
```

`cancel_document_processing`、Chunk 编辑、`reindex` 的 RAGFlow 支持情况待下一步骤按实际 RAGFlow 版本确认。RAGFlow 版本不支持的操作必须通过 capability 或明确的 `unsupported` 错误表达，不能静默忽略。

### 6.2 检索 Provider

`RetrievalProvider` 只负责检索。

```python
class RetrievalProvider:
    name: str

    def retrieve(self, request):
        raise NotImplementedError

    def health_check(self):
        raise NotImplementedError

    def get_capabilities(self):
        raise NotImplementedError
```

统一检索结果：

```text
RetrievalResult
  provider
  query
  chunks
  metadata
  latency_ms
```

说明：

- Retrieval Provider 不返回最终 `context_text`。
- 平台公共层根据统一 chunks 执行最终 Context Build/Compression。
- `answer` 不属于 Retrieval Provider 结果。
- `prompt_text` 不属于 Retrieval Provider 结果。
- `llm_model` 不属于 Retrieval Provider 结果。
- 最终 `sources` / `citations` 不属于 Retrieval Provider 结果，而由平台公共生成层从统一 chunks 中封装。

### 6.3 聚合 Provider

聚合 `KnowledgeProvider` 的职责只能是组合同一后端的 lifecycle 与 retrieval：

```text
KnowledgeProvider
├── LocalKnowledgeProvider
│   ├── lifecycle: LocalKnowledgeLifecycleProvider
│   └── retrieval: LocalRetrievalProvider
└── RagflowKnowledgeProvider
    ├── lifecycle: RagflowKnowledgeLifecycleProvider
    └── retrieval: RagflowRetrievalProvider
```

聚合 Provider 不定义万能结果模型。Provider 绑定发生在聚合 `KnowledgeProvider` 层，不允许把 Local lifecycle 与 RAGFlow retrieval 组合到同一个运行实例。

### 6.4 Capability 原则

Provider 必须声明能力：

```text
get_capabilities()
  supports_knowledge_base_create
  supports_knowledge_base_get
  supports_knowledge_base_update
  supports_knowledge_base_delete
  supports_document_upload
  supports_document_list
  supports_document_processing_start
  supports_document_processing_cancel
  supports_document_processing_retry
  supports_document_delete
  supports_chunk_list
  supports_chunk_create
  supports_chunk_update
  supports_chunk_delete
  supports_reindex
  supports_retrieval
  supports_health_check
```

Provider 输入输出必须满足：

- 上层不直接感知 Local 或 RAGFlow 原始响应。
- Provider 失败必须返回明确错误，不得吞异常。
- Provider 结果必须携带 `provider`、远端资源 ID、耗时和错误 metadata。
- Provider 绑定后不得直接原地修改。
- 不支持的操作必须返回明确 `unsupported`，不得静默成功。
- 公共接口和 capability 使用 `knowledge_base` 术语，RAGFlow 内部再将 `knowledge_base` 映射为 Dataset。
- RAGFlow 专属能力可以放入 provider-specific capability metadata。
- `cancel_document_processing`、Chunk 编辑和 `reindex` 的实际支持情况待下一步骤按 RAGFlow 版本确认。

## 7. Provider 绑定与远端资源映射原则

### 7.1 默认规则

- 现有知识库默认视为 `local`。
- 新建知识库默认仍为 `local`，除非用户显式选择 RAGFlow。
- Provider 绑定后不得直接原地修改。
- 如需从 Local 改为 RAGFlow，必须创建新知识库或走显式迁移流程；本阶段非目标。
- 如需从 RAGFlow 改为 Local，也必须创建新知识库或走显式迁移流程；本阶段非目标。

### 7.2 映射原则

未来需要保存：

```text
local knowledge_base_id -> provider -> external dataset_id
local document_id -> provider -> external document_id
```

映射只表示关系，不表示平台接管 RAGFlow 内部数据。

禁止：

- 直接写 RAGFlow 内部表。
- 直接改 RAGFlow Elasticsearch / Infinity / vector store。
- 直接读写 RAGFlow 数据卷。
- 假设 RAGFlow 内部表结构稳定。

### 7.3 数据不同步的用户语义

本架构明确 Local 与 RAGFlow 不自动同步。

因此：

- `local` Provider 查询 Local 知识库。
- `ragflow` Provider 查询 RAGFlow Dataset。
- 两边知识内容不保证一致。
- 上层 API、返回结构、前端展示可以统一。
- 知识内容本身不承诺无感。

## 8. 状态和错误统一原则

### 8.1 平台标准状态

本节定义目标架构的逻辑标准状态，只是架构契约，不代表当前已经完成数据库 schema 变更。

知识库供应状态：

```text
KnowledgeBaseProvisionStatus
- pending
- provisioning
- ready
- failed
- deleting
- deleted
```

文档 Provider 状态：

```text
DocumentProviderStatus
- creating
- uploaded
- processing
- ready
- failed
- deleting
- deleted
- remote_missing
```

状态原则：

- Local 和 RAGFlow 的原始状态必须统一映射到平台标准状态。
- RAGFlow 原始状态枚举及映射表待下一步骤按实际版本确认。
- `ready` 才表示资源可以进入正常检索。
- `failed` 必须保留失败阶段和脱敏错误信息。
- `remote_missing` 表示本地映射存在，但远端资源不存在。
- Provider 故障不得改变知识库绑定的 provider。
- RAGFlow 创建失败保持 `provider=ragflow`、`provision_status=failed`、`external_dataset_id=null`。

具体失败阶段放入独立字段或 metadata：

```text
failed_stage
- dataset_create
- document_upload
- processing_start
- processing
- delete
- reconciliation
```

补偿流程中只能使用上述状态枚举；额外失败原因必须放入 `failed_stage` 或 metadata，不能临时发明新的状态名。

### 8.2 Local 状态

当前 Local 文档状态来自 `documents.parse_status`：

- `pending`
- `processing`
- `success`
- `failed`

该状态属于 Local 文档处理链路，不应直接代表 RAGFlow 文档解析状态。

### 8.3 RAGFlow 状态

RAGFlow Provider 后续应维护独立状态：

- Dataset 供应状态。
- 文档上传状态。
- 文档解析状态。
- 文档索引状态。
- 最近错误信息。
- 最近重试时间。

RAGFlow 原始状态值待下一阶段根据实际 RAGFlow API 确认；进入平台后必须映射到 `KnowledgeBaseProvisionStatus` 或 `DocumentProviderStatus`。

### 8.4 错误原则

Provider 故障时不得静默切换到另一套知识库。

例如：

- 用户选择 `ragflow`，RAGFlow 超时，不得静默 fallback 到 `local`。
- 用户选择 `local`，Qdrant/BM25 失败，不得静默 fallback 到 `ragflow`。

原因：

- 两套知识库数据不同步。
- 静默切换会让用户得到来自另一套知识库的答案。
- 审计和评估无法解释。

允许的行为：

- 返回明确错误。
- metadata 中记录 `provider`、`failed_stage`、`error`。
- 管理员手动切换 Provider 或重试。

## 9. 跨本地数据库与 RAGFlow 的补偿原则

RAGFlow 接入后会出现跨系统操作：

```text
PostgreSQL
Local file storage
RAGFlow HTTP API
RAGFlow dataset/document/index
```

由于这些系统之间没有单一事务，必须按补偿原则设计。

### 9.1 幂等和对账原则

每次远端创建或上传操作必须先具备本地操作标识：

```text
operation_id
provider
resource_type
local_resource_id
operation_status
external_resource_id
attempt_count
last_error
```

架构原则：

- 调用 RAGFlow 前先创建或记录本地 `operation_id`。
- 如果实际 RAGFlow 版本支持幂等键，调用时使用该能力。
- 如果不支持幂等键，必须通过本地 operation、资源名称、hash 或其他稳定关联信息进行对账。
- 远端请求超时不能直接判定为创建失败。
- 出现超时或连接中断时，必须先查询或对账远端资源，再决定是否重试创建。
- 重试不得无条件重复创建 Dataset 或重复上传文档。
- 保存 `external_dataset_id` 或 `external_document_id` 后，后续重试必须复用远端 ID。
- 无法自动对账时，标准状态进入 `failed`，并记录 `failed_stage=reconciliation`，交由人工处理。
- 不允许通过文件名和上传时间直接自动认领远端文档；这些信息只能作为人工对账线索。
- 实际 RAGFlow API 是否支持幂等键，留到下一步骤确认。

### 9.2 创建 Dataset

建议流程：

```text
1. 本地创建知识库控制面记录，provider=ragflow, provision_status=provisioning
2. 创建或记录 operation_id
3. 调用 RAGFlow 创建 Dataset
4. 保存 external_dataset_id，并更新 provision_status=ready
```

如果第 3 步确认创建失败：

- 本地控制面记录保持 `provider=ragflow`。
- 本地控制面记录保持 `provision_status=failed`。
- `external_dataset_id=null`。
- metadata 记录 `failed_stage=dataset_create` 和脱敏错误。
- 不得假装 RAGFlow 已可用。
- 不得自动改成 `local`。
- 不得静默使用 Local Provider。
- 不得让用户以为 RAGFlow 已创建成功。
- 允许重试创建 Dataset。
- 允许删除失败的本地控制面记录。
- 允许人工修复映射。

如果第 3 步超时或连接中断，不能直接判定为创建失败；应先按 `operation_id`、资源名称、hash 或 RAGFlow 查询结果进行对账。确认远端没有创建成功后，才允许重试创建；确认远端已创建成功后，必须保存并复用 `external_dataset_id`。

如果第 4 步失败：

- RAGFlow 可能已创建 Dataset。
- 本地标准状态进入 `failed`，并记录 `failed_stage=reconciliation`。
- 后续必须先按 `operation_id`、资源名称、hash 或 RAGFlow 查询结果进行对账，再决定重试绑定或人工清理。

### 9.3 上传文档

建议流程：

```text
1. 创建本地 document 控制面记录，provider_status=creating
2. 上传文档到 RAGFlow
3. 上传成功后立即保存 external_document_id
4. 启动 RAGFlow 解析
5. 异步查询或刷新解析状态
6. 更新标准状态、chunk_count 和错误信息
```

不得等解析完成后才保存 `external_document_id`。远端文档 ID 是后续查询状态、重试、取消、删除和人工恢复的关键索引。

如果上传到 RAGFlow 成功但本地保存映射失败：

- 会出现 RAGFlow 有文档、本地不知道远端 ID 的不一致。
- 本地标准状态进入 `failed`，并记录 `failed_stage=reconciliation`。
- 后续必须支持按 Dataset 扫描远端文档或人工补录映射。
- 不允许仅凭文件名和上传时间自动认领远端文档；文件名和上传时间只能作为人工对账线索。

如果 `external_document_id` 已保存但启动解析失败：

- 本地记录保留远端 ID。
- 文档标准状态进入 `failed`，并记录 `failed_stage=processing_start`。
- 允许基于远端 ID 重试启动解析或删除远端文档。

如果解析期间服务重启：

- 本地记录必须依靠已保存的 `external_document_id` 恢复状态刷新。
- 后台任务应能重新查询 RAGFlow 文档状态，而不是重新上传同一文件。

如果本地记录存在但远端文档不存在：

- 文档标准状态进入 `remote_missing`。
- 不得把该文档视为可检索。
- 允许重新上传或删除本地控制面记录。

如果远端文档存在但本地映射丢失：

- 本地标准状态进入 `failed`，并记录 `failed_stage=reconciliation`。
- 应通过 Dataset 远端文档列表、文件名、hash、上传时间等信息人工或工具化修复映射。
- 修复前不得自动把远端文档归属到任意本地文档。

如果本地保存成功但 RAGFlow 解析失败：

- 文档标准状态进入 `failed`，并记录 `failed_stage=processing`。
- 查询时不得把该文档视为可检索。

### 9.4 删除文档

Provider 必须只删除自己拥有的资源：

- Local 删除只清理 Qdrant/BM25/本地 document。
- RAGFlow 删除只通过 RAGFlow API 删除远端 document，并更新本地映射状态。

如果 RAGFlow 删除失败：

- 不得先删除本地映射后丢失远端资源 ID。
- 本地标准状态进入 `failed`，并记录 `failed_stage=delete`，支持重试。

### 9.5 重新索引

Local reindex 使用当前 `KnowledgeBaseReindexService`。

RAGFlow reindex 必须走 RAGFlow Provider 自己的能力；不得调用本地 `DocumentPipeline`，也不得写本地 Qdrant/BM25。

不要预设 RAGFlow 一定存在独立 `reindex` API。具体支持情况待下一步骤按实际 RAGFlow 版本确认；如果版本不支持，应通过 capability 或明确 `unsupported` 错误表达。

## 10. 普通、工具和流式 Chat 接入边界

### 10.1 普通 Chat

当前非工具 Chat：

```text
ChatService.chat(enable_tools=False)
  -> RagChatPipeline.run()
  -> RetrieverPipeline
  -> PromptBuilder
  -> LLMFactory
```

后续 Provider 接入建议：

```text
ChatService / RagChatPipeline
  -> KnowledgeProviderFactory
  -> LocalKnowledgeProvider 或 RagflowKnowledgeProvider
  -> selected_provider.retrieval.retrieve()
  -> RetrievalResult
  -> 平台 Context Build/Compression
  -> PromptBuilder
  -> LLMFactory
  -> ChatResponse
```

### 10.2 工具 Chat

当前工具 Chat 路径没有复用 `RagChatPipeline`，而是在 `ChatService.chat(enable_tools=True)` 中直接执行：

```text
HybridRetriever
  -> Reranker
  -> MMR
  -> NeighborExpansion
  -> BasicContextBuilder
  -> ContextCompression
  -> PromptBuilder
  -> LLM with tools
```

RAGFlow Provider 接入后，如果 `provider=ragflow`：

- 不得继续执行本地 `HybridRetriever`。
- 不得继续执行本地 `Reranker`、`MMR`、`NeighborExpansion`。
- 应由 RAGFlow Retrieval Provider 返回最终 chunks，再进入平台公共层的 Context、Prompt、LLM with tools。

这一路径需要单独改造，不能只改 `RagChatPipeline`。

### 10.3 流式 Chat

当前流式 Chat：

```text
ChatService.stream_chat_events()
  -> RetrieverPipeline
  -> PromptBuilder
  -> LLM stream
```

RAGFlow Provider 接入后：

- 流式入口必须先完成 Provider 检索。
- RAGFlow 返回 chunks 后进入统一 Prompt 和 LLM stream。
- RAGFlow 检索失败时返回明确 SSE error，不得静默切回 Local。

### 10.4 Context Build/Compression

目标架构中，Context Build/Compression 的最终所有者只有平台公共层。

Local 当前 `RetrieverPipeline` 已包含 Context Build/Compression，这是现有实现事实，不代表目标 Provider 契约。后续 Local Provider 落地时，需要在适配层取得检索完成后的 chunks，或明确采用过渡适配，但不得把过渡实现写成最终架构。

RAGFlow 已完成召回、融合和可选重排。

平台 Context Build/Compression 只处理 RAGFlow Retrieval Provider 返回的最终 chunks：

- 不重新查询本地 Qdrant。
- 不重新查询本地 BM25。
- 不重新执行本地 Fusion。
- 不重新执行本地 Rerank。
- 不重新执行本地 MMR。
- 不重新执行本地 Neighbor Expansion。

是否对 Local 与 RAGFlow 两套 Provider 使用完全相同的 Compression 参数，留到后续实现和人工验证时确认。

### 10.5 Evaluation

当前 `evaluation/v2/targets/rag.py` 直接调用 `RetrieverPipeline` 或 `RagChatPipeline`。

Provider 接入后应支持：

```text
case.input.provider = local / ragflow
```

评估目标必须记录 provider metadata，避免混淆评估结果。

## 11. 同机部署拓扑

目标部署环境为同一台老 Mac。

逻辑拓扑：

```text
Old Mac
├── enterprise-ai-platform backend
│   ├── FastAPI process/container
│   ├── PostgreSQL access
│   ├── Redis access
│   ├── Qdrant access
│   └── Local upload/BM25 data directory
│
└── RAGFlow service
    ├── RAGFlow web/api service
    ├── RAGFlow database
    ├── RAGFlow Elasticsearch or Infinity
    ├── RAGFlow object/file storage
    └── RAGFlow model provider config
```

部署原则：

- 两套系统是独立服务。
- 两套系统不共用进程。
- 两套系统不共用容器。
- 两套系统只通过 RAGFlow HTTP API 通信。
- 平台不直接访问 RAGFlow 数据库、Elasticsearch、Infinity 或数据卷。
- Local 与 RAGFlow 使用独立数据目录和数据卷。
- RAGFlow 故障不能影响 Local Provider 启动和运行。

## 12. 网络、端口、数据卷和资源要求

### 12.1 网络

同机部署时推荐：

```text
enterprise-ai-platform backend
  -> http://127.0.0.1:<ragflow_port>
  -> RAGFlow HTTP API
```

如果两者都在 Docker 中：

- 使用 Docker network 内部服务名访问。
- 不要求 RAGFlow 暴露公网。
- 不把 API Key 写入代码、日志、前端或普通请求体。

### 12.2 端口

需要人工确认：

- 当前 backend 端口。
- 当前 frontend 端口。
- 当前 PostgreSQL 端口。
- 当前 Redis 端口。
- 当前 Qdrant 端口。
- RAGFlow Web/API 端口。
- RAGFlow Elasticsearch 或 Infinity 端口。
- 是否存在端口冲突。

无法访问目标服务器时，以上均标记为“待服务器手动确认”。

### 12.3 数据卷

Local 数据目录和 RAGFlow 数据目录必须分开。

Local 当前涉及：

- `UPLOAD_DIR`
- `data/bm25/index.pkl`
- Qdrant collection `enterprise_ai_chunks`

RAGFlow 后续涉及：

- RAGFlow 文件存储目录。
- RAGFlow 数据库数据目录。
- RAGFlow Elasticsearch 或 Infinity 数据目录。
- RAGFlow 配置目录。

具体路径待服务器手动确认。

### 12.4 资源

需要在老 Mac 手动确认：

- CPU 架构。
- CPU 核数。
- 内存容量。
- 可用磁盘空间。
- Docker 版本。
- Docker Compose 版本。
- 当前运行服务和端口。
- RAGFlow 选择 Elasticsearch 还是 Infinity。
- 数据卷位置。
- 是否使用外接 SSD。
- Backend 访问 RAGFlow 的地址。

不得使用当前开发机器信息代替目标老 Mac 信息。

## 13. 安全与故障隔离

### 13.1 安全边界

- RAGFlow API Key 只能由后端从进程环境变量、容器 Secret 或专用 Secret 管理机制读取。
- 具体使用环境变量、Docker Secret 还是其他 Secret 管理机制，待部署阶段确认。
- RAGFlow API Key 必须支持密钥轮换。
- RAGFlow API Key 不得写入代码。
- RAGFlow API Key 不得进入前端。
- RAGFlow API Key 不得进入普通请求体。
- RAGFlow API Key 不得进入数据库普通业务字段。
- RAGFlow API Key 不得写入日志。
- RAGFlow API Key 不得进入错误响应。
- RAGFlow API Key 不得进入测试 Fixture。
- RAGFlow API Key 不得进入 Git。
- 日志、metadata 和错误响应只能记录脱敏后的配置标识，不能记录密钥内容。
- Provider 健康检查不得回显 RAGFlow API Key。
- 本架构文档不得写入真实 RAGFlow API Key。
- 平台通过后端代理访问 RAGFlow，不建议前端直接访问 RAGFlow。
- 平台权限仍由 `enterprise-ai-platform` 控制。
- RAGFlow 不作为业务权限中心。
- RAGFlow Dataset 与平台 knowledge_base 的权限映射必须由平台显式控制。

远端错误进入 metadata 或日志前必须脱敏，不能直接保存：

- 完整 RAGFlow 原始响应。
- `Authorization`。
- URL 查询参数。
- 文件正文。
- 内部堆栈。
- 任何密钥、token、密码或签名信息。

### 13.2 故障隔离

- RAGFlow 启动失败不得影响 Local Provider。
- RAGFlow 查询超时不得静默切到 Local。
- Local Qdrant/BM25 故障不得静默切到 RAGFlow。
- Provider 健康状态应可单独检查。
- Provider 错误应进入统一 metadata 和日志。

### 13.3 备份

Local 与 RAGFlow 分开备份：

- Local PostgreSQL、Qdrant、BM25、上传文件。
- RAGFlow 数据库、索引、文件存储、配置。

两者备份和恢复不能互相覆盖。

## 14. 后续开发阶段

### 阶段 1：Provider 架构落地

- 新增 `KnowledgeLifecycleProvider` 协议。
- 新增 `RetrievalProvider` 协议。
- 新增 `LocalKnowledgeProvider` 聚合门面，组合 Local lifecycle 和 retrieval。
- 新增 `RagflowKnowledgeProvider` 聚合门面，组合 RAGFlow lifecycle 和 retrieval。
- 新增 Provider Factory。
- 默认 Provider 为 Local。

### 阶段 2：RAGFlow Client

- 根据实际部署版本确认 RAGFlow API。
- 新增 RAGFlow HTTP client。
- 实现 health check。
- 实现 Dataset 查询。
- 实现文档上传和状态查询。
- 实现 Retrieval API 调用。

### 阶段 3：查询链路接入

- 普通 Chat 接入 Provider。
- 流式 Chat 接入 Provider。
- 工具 Chat 接入 Provider。
- RAGFlow Provider 返回 chunks 后，跳过本地 Dense/Sparse/Fusion/Rerank/MMR/Neighbor。
- 本期正式主链路只使用 RAGFlow Retrieval API，不使用 RAGFlow Chat API。

### 阶段 4：文档生命周期接入

- 知识库创建绑定 RAGFlow Dataset。
- 文档上传到 RAGFlow。
- RAGFlow 解析状态映射。
- RAGFlow 删除和重试。
- RAGFlow reindex 能力待实际版本确认，不预设存在独立 API。

### 阶段 5：评估与后台

- Evaluation V2 支持 provider 参数。
- 后台展示 Provider、远端 ID、同步状态、错误信息。
- 支持管理员重试和手动故障处理。

## 15. 待确认问题

以下信息必须在老 Mac 或实际 RAGFlow 部署环境中手动确认：

- CPU 架构。
- CPU 和内存。
- 磁盘空间。
- Docker 版本。
- Docker Compose 版本。
- 当前服务和端口。
- RAGFlow Web/API 端口。
- RAGFlow 使用 Elasticsearch 还是 Infinity。
- RAGFlow 数据卷位置。
- 是否使用外接 SSD。
- Backend 访问 RAGFlow 的地址。
- RAGFlow 版本。
- RAGFlow API 鉴权方式。
- RAGFlow Dataset API 路径和字段。
- RAGFlow Document API 路径和字段。
- RAGFlow Retrieval API 路径和字段。
- RAGFlow Chat API 能力，仅作为未来独立实验候选，不进入本期正式主链路。
- RAGFlow 文档解析状态枚举。
- RAGFlow 删除和重试能力。
- RAGFlow 是否支持取消文档处理。
- RAGFlow 是否支持 Chunk 列表、创建、更新、删除。
- RAGFlow 是否支持独立 reindex 能力。

RAGFlow API 契约调查文档见：

- `.ai/11_RAGFlow_API_Contract.md`

当前开发环境无法替代目标服务器信息，未确认项不得写成已确认事实。
