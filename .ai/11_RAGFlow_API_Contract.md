# RAGFlow API 契约调查

## 1. 调查环境与日期

- 调查日期：2026-07-23
- 当前项目：`enterprise-ai-platform`
- 当前工作目录：`/Volumes/拓展SSD/Mac/PythonWrok/enterprise-ai-platform`
- 调查目标：为后续实现 RAGFlow Client 提供 API 契约依据。
- 本步骤性质：只读调查和文档整理，不修改业务代码、不修改配置、不执行远端写操作。

本次调查分成两类结论：

- 目标老 Mac 实际部署信息：当前工作区无法直接确认，标记为“待服务器手动确认”。
- RAGFlow 官方 HTTP API 契约：基于官方文档整理，后续必须按目标老 Mac 实际 RAGFlow 版本复核。

## 2. 实际 RAGFlow 版本及证据

### 2.1 当前确认结论

目标老 Mac 上实际部署的 RAGFlow 版本：待服务器手动确认。

原因：

- 当前环境执行 `docker ps --format ...` 返回 `docker: command not found`，无法查看运行容器、镜像 tag、端口和状态。
- 当前项目目录及上级三层目录未发现 RAGFlow 安装目录、RAGFlow compose 文件或 RAGFlow 版本文件。
- 当前 `enterprise-ai-platform` 源码中尚未实现 RAGFlow client、adapter、provider，也没有 RAGFlow 连接配置。

### 2.2 官方文档版本证据

本契约参考的官方文档页面：

- RAGFlow 官方 HTTP API：`https://ragflow.com.cn/docs/http_api_reference`
- 页面导航显示存在 `开发版` 和 `0.26.0` 文档入口。

注意：

- 官方文档版本不等于目标老 Mac 实际部署版本。
- 下一步开发前必须在目标服务器确认实际版本；不能用“最新版”或官方当前文档替代目标部署版本。

## 3. 部署和访问拓扑

### 3.1 当前确认结论

目标部署拓扑：待服务器手动确认。

当前无法确认：

- RAGFlow Web 地址。
- RAGFlow API base URL。
- Backend 应使用的访问地址。
- Backend 与 RAGFlow 是宿主机访问、Docker 网络访问还是其他方式。
- RAGFlow API 端口。
- 是否需要 HTTPS。
- 是否存在反向代理。
- 目标服务健康检查访问地址。

### 3.2 后续确认项

在目标老 Mac 上只读确认：

```text
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}'
```

如果 RAGFlow 使用 Docker Compose，再只读查看：

```text
docker compose ps
docker compose config --services
```

禁止输出完整容器环境变量，避免泄露密钥。

## 4. API base URL 规则

官方 HTTP API 示例使用：

```text
http://{address}/api/v1/...
```

平台后续配置建议：

```text
RAGFLOW_BASE_URL=http://<host>:<port>
```

Client 拼接规则：

```text
{RAGFLOW_BASE_URL}/api/v1/...
```

当前 `RAGFLOW_BASE_URL` 的真实值：待服务器手动确认。

## 5. 鉴权协议

官方 HTTP API 使用 Bearer Token：

```text
Authorization: Bearer <RAGFLOW_API_KEY>
```

鉴权规则：

- Header 名称：`Authorization`
- Token 格式：`Bearer <RAGFLOW_API_KEY>`
- API Key 来源：只能由后端从进程环境变量、容器 Secret 或专用 Secret 管理机制读取。
- 示例中只能使用 `<RAGFLOW_API_KEY>` 占位符。

当前未确认：

- 目标部署中 API Key 是用户级、系统级还是其他凭证类型。
- API Key 是否具备 Dataset 范围、账户范围或租户范围。
- 目标部署未授权时的真实响应体。

官方错误码表包含：

- `401 Unauthorized`
- `403 Forbidden`

但具体未授权响应结构仍需在目标环境用只读请求验证。验证时不得打印真实 Authorization Header。

## 6. Dataset API 契约

RAGFlow 中知识库对应资源称为 Dataset。

### 6.1 创建 Dataset

```text
功能：创建知识库 / Dataset
HTTP method：POST
path：/api/v1/datasets
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `name`: string，必填。
- `avatar`: string，可选。
- `description`: string，可选。
- `embedding_model`: string，可选。
- `permission`: string，可选，官方文档示例包含 `me` / `team`。
- `chunk_method`: string，可选。
- `parser_config`: object，可选。
- `parse_type`: int，可选，使用 ingestion pipeline 时需要。
- `pipeline_id`: string，可选，使用 ingestion pipeline 时需要。

response fields：

- `code`
- `data.id`
- `data.name`
- `data.chunk_count`
- `data.document_count`
- `data.embedding_model`
- `data.chunk_method`
- `data.parser_config`
- `data.permission`
- `data.similarity_threshold`
- `data.vector_similarity_weight`
- `data.status`
- `data.tenant_id`
- `data.token_num`
- `data.create_time`
- `data.update_time`

错误响应：

```json
{
  "code": 101,
  "message": "..."
}
```

证据来源：官方 HTTP API 文档 `创建知识库 / POST /api/v1/datasets`。

### 6.2 获取单个 Dataset

官方文档未看到独立 `GET /api/v1/datasets/{dataset_id}`。

可用的只读方式：

```text
GET /api/v1/datasets?id={dataset_id}
```

说明：

- 平台 `supports_knowledge_base_get` 可以通过 list + id filter 适配。
- 目标实际版本是否存在独立 get endpoint：待服务器手动确认。

### 6.3 列出 Dataset

```text
功能：列出 Dataset
HTTP method：GET
path：/api/v1/datasets
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
```

query fields：

- `page`: integer，默认 `1`。
- `page_size`: integer，默认 `30`。
- `orderby`: string，默认 `create_time`。
- `desc`: boolean，默认 `true`。
- `name`: string，可选。
- `id`: string，可选。
- `include_parsing_status`: boolean，可选。

response fields：

- `code`
- `data[]`
- `data[].id`
- `data[].name`
- `data[].chunk_count`
- `data[].document_count`
- `data[].embedding_model`
- `data[].status`
- `total_datasets`
- 当 `include_parsing_status=true` 时，返回 `unstart_count`、`running_count`、`cancel_count`、`done_count`、`fail_count`。

证据来源：官方 HTTP API 文档 `列出数据集 / GET /api/v1/datasets`。

### 6.4 更新 Dataset

```text
功能：更新 Dataset
HTTP method：PUT
path：/api/v1/datasets/{dataset_id}
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `name`
- `avatar`
- `description`
- `embedding_model`
- `permission`
- `chunk_method`
- `pagerank`
- `parser_config`

response fields：

- `code`

证据来源：官方 HTTP API 文档 `更新数据集 / PUT /api/v1/datasets/{dataset_id}`。

### 6.5 删除 Dataset

```text
功能：删除 Dataset
HTTP method：DELETE
path：/api/v1/datasets
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `ids`: list[string] 或 null。
- `delete_all`: boolean。

response fields：

- `code`

注意：

- `delete_all=true` 属于高风险写操作，平台 Client 默认不得使用。
- Step 1 未执行任何删除操作。

证据来源：官方 HTTP API 文档 `删除数据集 / DELETE /api/v1/datasets`。

## 7. Document API 契约

### 7.1 上传文档

```text
功能：上传文档
HTTP method：POST
path：/api/v1/datasets/{dataset_id}/documents
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
```

官方文档说明此 endpoint 支持三种模式：

- `type=local` 或省略：`multipart/form-data` 上传本地文件。
- `type=web`：`multipart/form-data` 提交 `name` 和 `url`。
- `type=empty`：`application/json` 提交 `name`。

本项目接入优先只使用：

```text
type=local 或省略
Content-Type: multipart/form-data
Form: file=@{FILE_PATH}
```

response fields：

- `code`
- `data[]`
- `data[].id`
- `data[].dataset_id`
- `data[].name`
- `data[].location`
- `data[].type`
- `data[].chunk_method`
- `data[].parser_config`
- `data[].run`
- `data[].size`

结论：

- 上传成功后官方响应会立即返回 document ID。
- 平台必须在上传成功后立即保存 `external_document_id`，不能等解析完成。

证据来源：官方 HTTP API 文档 `上传文档 / POST /api/v1/datasets/{dataset_id}/documents`。

### 7.2 获取单个文档

官方文档中的：

```text
GET /api/v1/datasets/{dataset_id}/documents/{document_id}
```

语义是下载文档文件，不是获取文档 metadata。

获取文档 metadata 建议使用：

```text
GET /api/v1/datasets/{dataset_id}/documents?id={document_id}
```

目标实际版本是否提供独立 metadata get endpoint：待服务器手动确认。

### 7.3 列出文档

```text
功能：列出 Dataset 内文档
HTTP method：GET
path：/api/v1/datasets/{dataset_id}/documents
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
```

query fields：

- `page`: integer，默认 `1`。
- `page_size`: integer，默认 `30`。
- `orderby`: string，默认 `create_time`。
- `desc`: boolean，默认 `true`。
- `keywords`: string，可选。
- `id`: string，可选。
- `name`: string，可选。
- `create_time_from`: integer，可选。
- `create_time_to`: integer，可选。
- `suffix`: array[string]，可选。
- `run`: array[string]，可选。
- `metadata_condition`: object，可选。

文档处理状态枚举：

```text
0 / UNSTART: 未处理
1 / RUNNING: 处理中
2 / CANCEL: 已取消
3 / DONE: 处理成功
4 / FAIL: 处理失败
```

response fields：

- `code`
- `data.docs[]`
- `data.docs[].id`
- `data.docs[].name`
- `data.docs[].dataset_id`
- `data.docs[].chunk_count`
- `data.docs[].token_count`
- `data.docs[].run`
- `data.docs[].progress`
- `data.docs[].progress_msg`
- `data.docs[].process_begin_at`
- `data.docs[].process_duration`
- `data.docs[].status`
- `data.total`

证据来源：官方 HTTP API 文档 `列出文档 / GET /api/v1/datasets/{dataset_id}/documents`。

### 7.4 更新文档

```text
功能：更新文档配置
HTTP method：PUT
path：/api/v1/datasets/{dataset_id}/documents/{document_id}
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `name`
- `meta_fields`
- `chunk_method`
- `parser_config`
- `enabled`

response fields：

- `code`
- `data.id`
- `data.name`
- `data.dataset_id`
- `data.run`
- `data.chunk_count`
- `data.token_count`
- `data.status`

证据来源：官方 HTTP API 文档 `更新文档 / PUT /api/v1/datasets/{dataset_id}/documents/{document_id}`。

### 7.5 删除文档

```text
功能：删除文档
HTTP method：DELETE
path：/api/v1/datasets/{dataset_id}/documents
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `ids`: list[string]。
- `delete_all`: boolean。

response fields：

- `code`

注意：

- `delete_all=true` 属于高风险写操作，平台 Client 默认不得使用。
- Step 1 未执行任何删除操作。

证据来源：官方 HTTP API 文档 `删除文档 / DELETE /api/v1/datasets/{dataset_id}/documents`。

### 7.6 启动文档处理 / 解析

```text
功能：解析指定 Dataset 中的文档
HTTP method：POST
path：/api/v1/datasets/{dataset_id}/chunks
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `document_ids`: list[string]，必填。

response fields：

- `code`

证据来源：官方 HTTP API 文档 `解析文档 / POST /api/v1/datasets/{dataset_id}/chunks`。

### 7.7 查询处理状态

官方文档未看到独立“查询处理状态” endpoint。

建议使用：

```text
GET /api/v1/datasets/{dataset_id}/documents?id={document_id}
```

从返回的文档字段读取：

- `run`
- `progress`
- `progress_msg`
- `process_begin_at`
- `process_duration`
- `chunk_count`
- `token_count`

目标实际版本是否存在独立状态接口：待服务器手动确认。

### 7.8 重试处理

官方文档未看到独立 retry endpoint。

候选方式：

```text
POST /api/v1/datasets/{dataset_id}/chunks
```

是否可作为失败后的 retry，必须按目标实际版本确认，不能直接假设。

### 7.9 取消处理

```text
功能：停止解析文档
HTTP method：DELETE
path：/api/v1/datasets/{dataset_id}/chunks
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `document_ids`: list[string]，必填。

response fields：

- `code`

证据来源：官方 HTTP API 文档 `停止解析文档 / DELETE /api/v1/datasets/{dataset_id}/chunks`。

## 8. Chunk API 契约

### 8.1 列出 chunks

```text
功能：列出文档 chunks
HTTP method：GET
path：/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
```

query fields：

- `keywords`: string，可选。
- `page`: integer，默认 `1`。
- `page_size`: integer，默认 `30`。
- `id`: string，可选。

response fields：

- `code`
- `data.chunks[]`
- `data.chunks[].id`
- `data.chunks[].content`
- `data.chunks[].document_id`
- `data.chunks[].docnm_kwd`
- `data.chunks[].available`
- `data.chunks[].image_id`
- `data.chunks[].important_keywords`
- `data.chunks[].tag_kwd`
- `data.chunks[].positions`
- `data.doc`

证据来源：官方 HTTP API 文档 `列出块 / GET /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks`。

### 8.2 获取单个 chunk

```text
功能：获取单个 chunk
HTTP method：GET
path：/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
```

response fields：

- `code`
- `data.id`
- `data.content_with_weight`
- `data.doc_id`
- `data.docnm_kwd`
- `data.available_int`
- `data.img_id`
- `data.important_kwd`
- `data.question_kwd`
- `data.tag_kwd`

说明：

- 官方文档说明该接口不返回向量和 token 等运行时字段。
- 单 chunk 获取接口的文本字段是 `content_with_weight`，与 list/retrieval 的 `content` 不完全一致，Client 需要适配。

### 8.3 创建 chunk

```text
功能：添加 chunk
HTTP method：POST
path：/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `content`: string，必填。
- `important_keywords`: list[string]，可选。
- `tag_kwd`: list[string]，可选。
- `questions`: list[string]，可选。
- `image_base64`: string，可选。

response fields：

- `code`
- `data.chunk.id`
- `data.chunk.content`
- `data.chunk.dataset_id`
- `data.chunk.document_id`
- `data.chunk.important_keywords`
- `data.chunk.tag_kwd`
- `data.chunk.questions`

证据来源：官方 HTTP API 文档 `添加块 / POST /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks`。

### 8.4 更新 chunk

```text
功能：更新 chunk
HTTP method：PATCH
path：/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `content`
- `important_keywords`
- `questions`
- `positions`
- `tag_kwd`
- `available`
- `image_base64`

response fields：

- `code`

注意：

- 旧的 `PUT /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}` 已弃用。
- 后续 Client 应使用 `PATCH`，除非目标实际版本不支持。

### 8.5 删除 chunk

```text
功能：删除 chunks
HTTP method：DELETE
path：/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `chunk_ids`: list[string]。
- `delete_all`: boolean。

response fields：

- `code`

注意：

- `delete_all=true` 属于高风险写操作，平台 Client 默认不得使用。
- Step 1 未执行任何删除操作。

## 9. Retrieval API 契约

正式主链路固定使用 Retrieval API，不使用 RAGFlow Chat API。

```text
功能：从指定 Dataset / Document 检索 chunks
HTTP method：POST
path：/api/v1/retrieval
鉴权：Authorization: Bearer <RAGFLOW_API_KEY>
content type：application/json
```

request fields：

- `question`: string，必填。平台统一 query 映射到该字段。
- `dataset_ids`: list[string]。未设置时必须设置 `document_ids`。
- `document_ids`: list[string]。未设置时必须设置 `dataset_ids`。
- `page`: integer，默认 `1`。
- `page_size`: integer，默认 `30`。
- `similarity_threshold`: float，默认 `0.2`。
- `vector_similarity_weight`: float，默认 `0.3`。
- `top_k`: integer，默认 `1024`。
- `rerank_id`: string / integer，官方文档不同位置表述不完全一致，目标版本需复核。
- `keyword`: boolean，是否启用关键词匹配。
- `highlight`: boolean，是否返回高亮。
- `cross_languages`: list[string]。
- `metadata_condition`: object。
- `use_kg`: boolean。
- `toc_enhance`: boolean。

metadata filter 结构：

```json
{
  "logic": "and",
  "conditions": [
    {
      "name": "author",
      "comparison_operator": "=",
      "value": "Toby"
    }
  ]
}
```

response fields：

- `code`
- `data.chunks[]`
- `data.chunks[].id`
- `data.chunks[].content`
- `data.chunks[].content_ltks`
- `data.chunks[].document_id`
- `data.chunks[].document_keyword`
- `data.chunks[].kb_id`
- `data.chunks[].highlight`
- `data.chunks[].image_id`
- `data.chunks[].important_keywords`
- `data.chunks[].tag_kwd`
- `data.chunks[].positions`
- `data.chunks[].similarity`
- `data.chunks[].term_similarity`
- `data.chunks[].vector_similarity`
- `data.doc_aggs[]`
- `data.doc_aggs[].doc_id`
- `data.doc_aggs[].doc_name`
- `data.doc_aggs[].count`
- `data.total`

score 语义：

- `similarity`: 综合相似度。
- `vector_similarity`: 向量相似度。
- `term_similarity`: 词项 / 关键词相似度。
- `vector_similarity_weight` 控制向量相似度权重，`1 - vector_similarity_weight` 是词项相似度权重。

空结果结构：

- 官方文档未给出空结果示例。
- Client 应按 `code=0`、`data.chunks=[]`、`data.total=0` 设计兼容，但目标实际响应需只读验证。

latency：

- 官方 Retrieval API 响应示例未看到 latency 字段。
- 平台 `RetrievalResult.latency_ms` 应由 Client 自行测量。

错误响应：

```json
{
  "code": 102,
  "message": "`datasets` is required."
}
```

证据来源：官方 HTTP API 文档 `检索块 / POST /api/v1/retrieval`。

## 10. 健康检查契约

官方文档存在系统健康检查：

```text
功能：检查系统健康状况
HTTP method：GET
path：/api/v1/system/healthz
鉴权：无需授权
content type：application/json
```

response fields：

- `db`: `ok` / `nok`
- `redis`: `ok` / `nok`
- `doc_engine`: `ok` / `nok`
- `storage`: `ok` / `nok`
- `status`: `ok` / `nok`
- `_meta`: object，不健康时可能包含详细错误

HTTP 状态：

- `200 OK`: 所有服务健康。
- `500 Internal Server Error`: 至少一个服务不健康。

注意：

- 旧路径 `GET /v1/system/healthz` 已弃用。
- 后续 Client 应优先使用 `GET /api/v1/system/healthz`。
- 健康检查不得回显 API Key。

证据来源：官方 HTTP API 文档 `检查系统健康状况 / GET /api/v1/system/healthz`。

## 11. HTTP 错误结构

官方文档错误码表包含：

| HTTP 状态码 | 含义 |
|---:|---|
| 400 | Bad Request，请求参数无效 |
| 401 | Unauthorized，未经授权 |
| 403 | Forbidden，访问被拒绝 |
| 404 | Not Found，资源未找到 |
| 500 | Internal Server Error，服务器内部错误 |

业务错误响应常见结构：

```json
{
  "code": 102,
  "message": "..."
}
```

Client 处理原则：

- HTTP 非 2xx 必须转成 Provider 错误。
- HTTP 2xx 但 `code != 0` 也必须转成 Provider 错误。
- 错误进入日志、metadata 或响应前必须脱敏。
- 不保存完整远端响应、Authorization、URL 查询参数、文件正文或内部堆栈。

## 12. 分页规则

官方 Dataset、Document、Chunk、Retrieval API 均出现分页字段：

- `page`: 默认 `1`。
- `page_size`: 默认 `30`。

Dataset list 返回：

- `data[]`
- `total_datasets`

Document list 返回：

- `data.docs[]`
- `data.total`

Chunk list 返回：

- `data.chunks[]`
- `data.doc`

Retrieval 返回：

- `data.chunks[]`
- `data.doc_aggs[]`
- `data.total`

注意：

- 不同 API 的 total 字段命名不一致。
- Client 需要为每类 endpoint 单独适配分页响应。

## 13. 状态映射初稿

### 13.1 KnowledgeBaseProvisionStatus

RAGFlow Dataset 官方响应中看到 `status` 字段，例如 `"1"`，但官方文档未解释 Dataset status 枚举。

初稿映射：

| RAGFlow 字段 | RAGFlow 值 | 平台状态 | 依据 |
|---|---|---|---|
| Dataset 创建成功返回 `data.id` | 任意 | `ready` | 创建成功且可获得 ID |
| Dataset 创建失败 | `code != 0` 或 HTTP 错误 | `failed` | 错误响应 |
| Dataset 删除中 | 待确认 | `deleting` | 待实际版本确认 |
| Dataset 删除成功 | `code=0` | `deleted` | 删除接口成功响应 |
| Dataset 远端不存在 | 待确认 | `remote_missing` 不适用于 KB，建议 KB 进入 `failed + failed_stage=reconciliation` | 待实际版本确认 |

### 13.2 DocumentProviderStatus

RAGFlow Document list 支持 `run` 状态：

| RAGFlow run | 平台状态 |
|---|---|
| `0` / `UNSTART` | `uploaded` |
| `1` / `RUNNING` | `processing` |
| `2` / `CANCEL` | `failed`，`failed_stage=processing`，metadata 标记 `cancelled=true` |
| `3` / `DONE` | `ready` |
| `4` / `FAIL` | `failed`，`failed_stage=processing` |
| 本地映射存在但远端文档不存在 | `remote_missing` |

注意：

- 这是基于官方文档的初稿。
- 目标实际版本返回数字、文本还是混合格式，需只读验证。

## 14. Capability 矩阵

由于目标老 Mac 实际版本尚未确认，“实际版本是否支持”不能写成已确认。下表的 API 路径来自官方文档，下一步必须按目标实际版本复核。

| Capability | 实际版本是否支持 | API 路径 | 证据 | 备注 |
|---|---:|---|---|---|
| supports_knowledge_base_create | 待确认 | `POST /api/v1/datasets` | 官方文档支持 | RAGFlow 术语为 Dataset |
| supports_knowledge_base_get | 待确认 | `GET /api/v1/datasets?id={dataset_id}` | 官方文档支持 list + id filter | 未确认独立 get endpoint |
| supports_knowledge_base_update | 待确认 | `PUT /api/v1/datasets/{dataset_id}` | 官方文档支持 | 目标版本需复核 |
| supports_knowledge_base_delete | 待确认 | `DELETE /api/v1/datasets` | 官方文档支持 | 禁止默认使用 `delete_all=true` |
| supports_document_upload | 待确认 | `POST /api/v1/datasets/{dataset_id}/documents` | 官方文档支持 | `multipart/form-data`，成功后返回 document ID |
| supports_document_list | 待确认 | `GET /api/v1/datasets/{dataset_id}/documents` | 官方文档支持 | 可用 `id` filter 查询单文档 metadata |
| supports_document_processing_start | 待确认 | `POST /api/v1/datasets/{dataset_id}/chunks` | 官方文档支持 | 语义为解析文档 |
| supports_document_processing_cancel | 待确认 | `DELETE /api/v1/datasets/{dataset_id}/chunks` | 官方文档支持 | 语义为停止解析文档 |
| supports_document_processing_retry | 待确认 | 待确认 | 未发现独立 retry endpoint | 不能假设重新 POST parse 就是 retry |
| supports_document_delete | 待确认 | `DELETE /api/v1/datasets/{dataset_id}/documents` | 官方文档支持 | 禁止默认使用 `delete_all=true` |
| supports_chunk_list | 待确认 | `GET /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks` | 官方文档支持 | 支持 `keywords/page/page_size/id` |
| supports_chunk_create | 待确认 | `POST /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks` | 官方文档支持 | 写操作，Step 1 未执行 |
| supports_chunk_update | 待确认 | `PATCH /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks/{chunk_id}` | 官方文档支持 | 旧 PUT 已弃用 |
| supports_chunk_delete | 待确认 | `DELETE /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks` | 官方文档支持 | 禁止默认使用 `delete_all=true` |
| supports_reindex | 待确认 | 待确认 | 未发现独立 reindex endpoint | 不预设存在 |
| supports_retrieval | 待确认 | `POST /api/v1/retrieval` | 官方文档支持 | 本期正式主链路 |
| supports_health_check | 待确认 | `GET /api/v1/system/healthz` | 官方文档支持 | 无需授权 |

## 15. 已确认事实

- 当前 `enterprise-ai-platform` 没有 RAGFlow client、adapter、provider 实现。
- 当前 `enterprise-ai-platform` 没有知识库级 provider 绑定字段。
- 当前 `enterprise-ai-platform` 没有本地 document 与 RAGFlow document 的映射字段。
- 当前工作区无法确认目标老 Mac 的 RAGFlow 实际版本。
- 当前工作区无法确认 RAGFlow Web 地址、API base URL、端口、HTTPS、反向代理和 Docker 网络。
- 官方 HTTP API 文档显示 RAGFlow 使用 Dataset 表示知识库。
- 官方 HTTP API 文档显示 Retrieval API 为 `POST /api/v1/retrieval`。
- 官方 HTTP API 文档显示鉴权方式为 `Authorization: Bearer <RAGFLOW_API_KEY>`。
- 官方 HTTP API 文档显示健康检查为 `GET /api/v1/system/healthz`，无需授权。
- 官方 HTTP API 文档显示 RAGFlow Chat API 存在，但本期正式主链路不使用 Chat API。

## 16. 待手动确认事项

必须在目标老 Mac 或实际 RAGFlow 部署环境中确认：

- RAGFlow 实际版本。
- RAGFlow 页面或版本接口是否显示版本。
- 运行容器名称、镜像名、镜像 tag、端口和状态。
- RAGFlow 安装目录、Compose 文件和版本文件位置。
- RAGFlow Web 地址。
- RAGFlow API base URL。
- Backend 应使用宿主机地址还是 Docker network 服务名。
- RAGFlow API 端口。
- 是否需要 HTTPS。
- 是否存在反向代理。
- API Key 类型、范围和权限。
- 未授权时的 HTTP 状态码和响应体。
- Dataset status 字段枚举。
- Document run 字段在实际版本中返回数字、文本还是混合格式。
- 是否存在独立 document metadata get endpoint。
- 是否存在独立 document retry endpoint。
- 是否存在独立 reindex endpoint。
- Retrieval API 空结果结构。
- Retrieval API 是否返回 latency。
- `rerank_id` 在实际版本中是 string 还是 integer。
- metadata filter 在目标版本中的完整比较符支持情况。

## 17. 下一步 Client 实现边界

下一步只能在确认目标实际版本后实现 RAGFlow Client。

Client 实现边界：

- 正式主链路只实现 `POST /api/v1/retrieval` 返回 chunks。
- 不接入 RAGFlow Chat API。
- 不把 RAGFlow answer、prompt、LLM、session 作为平台主链路。
- 不在 RAGFlow 返回 chunks 后执行本地 Dense/Sparse/Fusion/Rerank/MMR/Neighbor Expansion。
- 平台继续负责 Context Build/Compression、Prompt、Conversation、Memory、LLM、Sources、Citations 和 ChatResponse。
- API Key 只能由后端安全读取，不进入前端、请求体、日志、错误响应、测试 Fixture 或 Git。
- 任何写操作都必须先具备本地 `operation_id`、幂等和对账设计。
- `delete_all=true` 这类高风险参数默认禁止。
- 如果目标版本不支持某能力，Provider 必须返回明确 `unsupported`，不得静默成功。

## 18. 证据来源

- RAGFlow 官方 HTTP API 文档：`https://ragflow.com.cn/docs/http_api_reference`
- 当前项目架构规范：`.ai/10_RAG_Provider_Architecture.md`
- 当前项目状态：`.ai/PROJECT_STATE.md`

