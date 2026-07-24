# 商品目录与推荐设计

## 1. 商品数据来源

阶段一商品数据来自版本化 Fixture，并通过脚本幂等导入 PostgreSQL。

约束：

- 商品 Fixture 只保存稳定商品数据。
- 商品 Fixture 不保存 `document_id`。
- 商品价格、库存、热度都是模拟数据。
- 不抓取九阳官网。
- 不通过 PDF 文件名创建商品。
- 不通过上传说明书自动猜测商品映射。

## 2. 商品表草案

后续计划新增表：

```text
products
```

字段至少包括：

```text
id
product_code
brand
name
model
category
description
price
currency
stock_quantity
sale_status
features
use_cases
specifications
tags
popularity_score
official_product_url
source_checked_at
is_active
created_at
updated_at
deleted_at
```

本步骤只记录目标设计，不创建 ORM，不创建 migration。

## 3. 商品说明书关联表草案

后续计划新增表：

```text
product_document_links
```

字段至少包括：

```text
id
product_id
document_id
document_type
is_primary
manual_version
source_url
downloaded_at
created_at
updated_at
```

用途：

- 把结构化商品和 Local RAG 文档显式绑定。
- 支持一个商品关联多份说明书或规则文档。
- 支持主说明书标记。
- `(product_id, document_id)` 唯一，重复绑定幂等返回已有记录。
- 一个商品最多一条 `is_primary=true` 的主说明书关联。
- 数据库使用部分唯一索引或等价约束保证主说明书唯一。

## 4. 字段和约束

下一数据库 Checkpoint 的目标约束如下。本节只记录目标设计，不代表当前数据库已经实现；schema 变更仍属于项目红线，真正创建 ORM 或 Alembic migration 前必须获得用户明确确认。

`product_code`：

- 跨环境稳定。
- Fixture 必须保存。
- Seed 时用于幂等 upsert。
- 不依赖数据库自增 ID。
- `NOT NULL + UNIQUE`。

`brand`、`name`、`model`、`category`：

- `NOT NULL`。

`price`：

- 使用定点小数。
- `NOT NULL`。
- 满足 `price >= 0`。

`currency`：

- ISO 4217 三字符。
- 默认 `CNY`。

`stock_quantity`：

- 整数。
- 默认 `0`。
- 满足 `stock_quantity >= 0`。

`popularity_score`：

- 原始范围明确为 `[0, 100]`。
- 推荐计算前再归一化为 `[0, 1]`。

`sale_status`：

- 必须属于 `on_sale/off_sale/pre_sale/discontinued`。

`is_active`：

- 默认 `true`。

`document_id`：

- 当前数据库环境产生。
- 不跨环境稳定。
- 版本化 Fixture 不保存。
- 只在说明书上传解析完成后，由人工或工具显式绑定。

`sale_status` 建议枚举：

```text
on_sale
off_sale
pre_sale
discontinued
```

`features`、`use_cases`、`specifications`、`tags`：

- 建议使用 JSONB。
- 只保存结构化、可筛选信息。
- 不保存说明书全文。
- `features` 作为商品自身功能全集；查询入参中的旧 `features` 只能作为 `required_features` 的兼容别名。
- `use_cases` 作为商品自身适用场景全集；查询入参中的旧 `use_cases` 只能作为 `preferred_use_cases` 的兼容别名。

索引规划：

- 为 `brand`、`category`、`model`、`sale_status`、`is_active` 规划必要索引。
- 为常用价格查询规划必要索引。
- JSONB 字段是否建立 GIN 索引，由数据库 Checkpoint 根据实际查询方式确认，不在本文档中假装已实现。

`product_document_links` 约束：

- `product_id` 外键指向 `products.id`。
- `document_id` 外键指向 `documents.id`。
- 两个外键删除策略统一规划为 `ON DELETE CASCADE`，避免孤立关联。
- `(product_id, document_id)` 唯一。
- 主说明书部分唯一约束：

```text
UNIQUE(product_id)
WHERE is_primary = true
  AND document_type = 'manual'
```

- `document_type` 至少规划 `manual/warranty/policy/other`。
- `is_primary` 默认 `false`。
- 非 `manual` 类型不得设置为主说明书。
- 数据库约束和 Service 校验必须同时存在。

## 5. 查询参数

`search_products` 至少支持：

```text
keyword
brand
category
price_min
price_max
required_features
excluded_features
preferred_features
required_use_cases
preferred_use_cases
in_stock_only
sale_status
sort_by
sort_order
page
page_size
```

参数规则：

- `keyword` 匹配商品名称、型号、描述、标签。
- `required_features` 表示必须具备的功能。
- `excluded_features` 表示不能包含或用户明确排除的功能。
- `preferred_features` 表示用户偏好的功能，参与评分但不作为硬过滤。
- `required_use_cases` 表示必须适用的使用场景，是硬过滤。
- `preferred_use_cases` 表示偏好的使用场景，只参与评分。
- 用户明确说“必须适合宿舍”“只能用于宿舍”等，进入 `required_use_cases`。
- 用户只说“适合宿舍”“主要在宿舍使用”，默认进入 `preferred_use_cases`。
- 旧 `features` 参数只能作为 `required_features` 的兼容别名，并由 Service 统一。
- 旧 `use_cases` 参数只能作为 `preferred_use_cases` 的兼容别名，并由 Service 统一转换。
- `in_stock_only=true` 只返回 `stock_quantity > 0`。
- `page_size` 需要上限，避免一次返回过多商品。

硬条件：

- `category`
- `brand`
- `price_min/price_max`
- `required_features`
- `excluded_features`
- `required_use_cases`
- 库存。
- 上下架状态。

软偏好：

- `preferred_features`
- `preferred_use_cases`
- 容量。
- 清洁。
- 噪音。
- 便携。
- 家庭人数。

只有用户明确说“必须”“只要”“不能接受”时，对应偏好才升级为硬条件。`required_use_cases` 不得被自动放宽；软偏好参与评分，未完全命中不能导致自动放宽硬条件。

## 6. 排序白名单

允许排序字段：

```text
popularity
price
stock_quantity
created_at
updated_at
```

排序规则：

- 默认 `sort_by=popularity`。
- 默认 `sort_order=desc`。
- 不允许用户传任意数据库字段作为排序字段。
- 非白名单字段必须返回参数错误或回退到默认排序，并在 metadata 标记。

## 7. 多轮条件合并规则

多轮条件通过相同 `conversation_id` 和现有 Agent Session Memory 继承。

规则：

- 后一轮的明确条件覆盖前一轮同类条件。
- 后一轮新增条件与前一轮条件合并。
- 用户明确说“重新来”“换个类别”时清空冲突条件。
- 用户只说价格区间时，继承上一轮 `category`、`brand`、`required_features`、`excluded_features`、`preferred_features`、`required_use_cases`、`preferred_use_cases` 等上下文。
- 不把闲聊内容写入商品查询条件。

示例：

```text
第一轮：给我推荐几个豆浆机
条件：category=豆浆机, sale_status=on_sale, in_stock_only=true, sort_by=popularity, sort_order=desc, page_size=3

第二轮：挑几款200到300区间的
条件：继承 category=豆浆机，增加 price_min=200, price_max=300
```

## 8. 确定性推荐算法

`recommend_products` 必须由确定性 Service 执行硬过滤和评分。

硬过滤：

- `is_active=true`
- 默认 `sale_status=on_sale`
- 默认 `stock_quantity > 0`
- 用户指定 `category`、`brand`、`price_min/price_max`、`required_features`、`excluded_features`、`required_use_cases`、库存、上下架状态必须全部满足。
- 不自动放宽用户硬条件。
- `preferred_features`、`preferred_use_cases`、容量、清洁、噪音、便携、家庭人数等软偏好不作为硬过滤。

评分建议：

所有评分分量在计算前必须归一化到 `[0, 1]`：

- `popularity_score_norm`
- `feature_match_score`
- `use_case_match_score`
- `price_preference_score`
- `stock_score`

```text
score =
  popularity_score_norm * 0.35
  + feature_match_score * 0.25
  + use_case_match_score * 0.25
  + price_preference_score * 0.10
  + stock_score * 0.05
```

最终 `score` 也必须限定在 `[0, 1]`。

相同分数使用稳定排序：

```text
score DESC
popularity_score DESC
product_code ASC
```

输出：

- 最多 3 款。
- 每款必须返回匹配理由。
- 没有结果时返回空列表和无结果原因。
- LLM 只能解释 Service 返回结果。

## 9. 商品对比规则

`compare_products` 输入：

- `product_codes` 或已查询结果中的商品 ID。
- 可选 `fields`。

对比字段：

- 品牌。
- 型号。
- 分类。
- 模拟价格。
- 模拟库存。
- 上下架状态。
- 功能标签。
- 使用场景。
- 结构化规格。
- 说明书证据摘要。

规则：

- 缺失字段显示“当前资料未提供”。
- 不把说明书没有写的内容当成商品功能。
- 不比较无法证明的主观体验。

## 10. 商品 Seed 流程

后续脚本：

```text
seed_customer_service_products.py
```

流程：

```text
读取商品 Fixture
-> 按 product_code 查找商品
-> 存在则更新稳定字段
-> 不存在则创建
-> 不写 document_id
-> 输出新增、更新、跳过数量
```

要求：

- 幂等。
- 可重复执行。
- 不删除数据库中未出现在 Fixture 的人工数据，除非后续显式设计。
- 不保存环境相关自增 ID。

## 11. 说明书上传和绑定流程

说明书流程：

```text
1. 使用现有 /documents/upload 上传说明书
2. 使用现有 /documents/{id}/parse 解析并写入 Local RAG
3. 确认商品 product_code
4. 执行 link_product_manual.py product_code + document_id
5. 写入 product_document_links
```

约束：

- 产品说明书上传不会自动创建商品。
- 商品可以暂时没有说明书。
- 不允许通过 PDF 文件名自动创建商品。
- 不允许通过上传说明书自动猜测商品映射。
- 没有关联说明书时，Agent 不得编造产品详细功能。
- 绑定前验证商品存在且未软删除。
- 绑定前验证文档存在、未软删除、解析成功。
- 绑定前验证文档属于允许的产品说明书知识库。
- 重复绑定幂等返回已有记录。
- 更换主说明书必须在同一事务中取消旧主说明书并设置新主说明书。

后续脚本：

```text
link_product_manual.py
```

输入：

```text
product_code
document_id
document_type
is_primary
manual_version
source_url
```

## 12. 跨环境 document_id 原则

跨环境稳定标识：

- `product_code`
- Fixture 文件路径。
- 商品型号。
- 文档源 URL。
- 手册版本。

跨环境不稳定标识：

- `products.id`
- `documents.id`
- `product_document_links.id`

原则：

- 版本化 Fixture 不保存 `document_id`。
- `document_id` 只能在当前环境上传解析后产生。
- 说明书绑定必须显式执行。
- 迁移环境时，先 seed 商品，再上传说明书，再按当前环境 document_id 绑定。
