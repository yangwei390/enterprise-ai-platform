# Enterprise AI Platform 编码规范

## 一、命名规范

文件：

snake_case

类：

PascalCase

函数：

snake_case

变量：

snake_case

常量：

UPPER_CASE

---

## 二、目录规范

一个目录只负责一个职责。

禁止：

utils 中放所有代码。

---

## 三、异常处理

所有接口必须：

- 捕获异常
- 输出日志
- 返回统一错误

禁止：

try:
    ...
except:
    pass

---

## 四、日志规范

每个 Service：

必须记录：

- 开始
- 成功
- 失败

---

## 五、配置规范

所有：

数据库

Redis

Qdrant

LLM

Embedding

全部通过配置文件读取。

禁止硬编码。

---

## 六、Git规范

一个功能

↓

一个 Commit

Commit 使用英文。

例如：

feat: add pdf upload

fix: fix parser bug

---

## 七、AI 开发规范

AI 修改代码必须：

- 阅读 .ai 文档
- 不修改无关代码
- 保持代码风格一致
- 自动补充必要注释