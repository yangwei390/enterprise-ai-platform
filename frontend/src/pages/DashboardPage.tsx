const capabilities = [
  ["RAG", "知识库问答、文档解析、上下文构建与引用来源。"],
  ["Hybrid Search", "向量检索、BM25 框架与 RRF 融合预留。"],
  ["Memory", "会话历史、摘要记忆与 Prompt 注入。"],
  ["Tool Calling", "工具注册、Schema、执行器与 Function Calling 抽象。"],
  ["Workflow", "节点式执行流、artifact、logs 与防死循环 runtime。"],
  ["Agent", "Planner、Executor、Reflection 与 final answer 汇总。"],
  ["MCP", "远程 HTTP tool 适配，预留外部 MCP Server 接入边界。"]
];

export default function DashboardPage() {
  return (
    <div>
      <div className="page-title">
        <h2>Dashboard</h2>
        <p>Enterprise AI Platform capability overview.</p>
      </div>
      <div className="card-grid">
        {capabilities.map(([title, description]) => (
          <article className="card" key={title}>
            <h3>{title}</h3>
            <p>{description}</p>
          </article>
        ))}
      </div>
    </div>
  );
}
