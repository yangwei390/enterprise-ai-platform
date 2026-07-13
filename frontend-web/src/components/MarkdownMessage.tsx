import { ReactNode } from "react";

type MarkdownMessageProps = {
  content: string;
};

type TextBlock = {
  type: "text";
  value: string;
};

type CodeBlock = {
  type: "code";
  value: string;
  language: string;
};

type MarkdownBlock = TextBlock | CodeBlock;

export default function MarkdownMessage({ content }: MarkdownMessageProps) {
  const blocks = parseFencedCode(content);
  return (
    <div className="markdown-message">
      {blocks.map((block, index) =>
        block.type === "code" ? (
          <CodeBlockView block={block} key={index} />
        ) : (
          <TextBlockView value={block.value} key={index} />
        )
      )}
    </div>
  );
}

function CodeBlockView({ block }: { block: CodeBlock }) {
  return (
    <div className="code-block">
      {block.language && <span>{block.language}</span>}
      <pre><code>{block.value}</code></pre>
    </div>
  );
}

function TextBlockView({ value }: { value: string }) {
  const lines = value.split("\n");
  const nodes: ReactNode[] = [];
  let listItems: string[] = [];
  let ordered = false;

  function flushList() {
    if (listItems.length === 0) {
      return;
    }
    const list = ordered ? (
      <ol key={`list-${nodes.length}`}>
        {listItems.map((item, index) => <li key={index}>{renderInline(item)}</li>)}
      </ol>
    ) : (
      <ul key={`list-${nodes.length}`}>
        {listItems.map((item, index) => <li key={index}>{renderInline(item)}</li>)}
      </ul>
    );
    nodes.push(list);
    listItems = [];
  }

  lines.forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      return;
    }
    const orderedMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    const unorderedMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (orderedMatch) {
      if (listItems.length > 0 && !ordered) {
        flushList();
      }
      ordered = true;
      listItems.push(orderedMatch[1]);
      return;
    }
    if (unorderedMatch) {
      if (listItems.length > 0 && ordered) {
        flushList();
      }
      ordered = false;
      listItems.push(unorderedMatch[1]);
      return;
    }

    flushList();
    if (trimmed.startsWith("### ")) {
      nodes.push(<h4 key={index}>{renderInline(trimmed.slice(4))}</h4>);
    } else if (trimmed.startsWith("## ")) {
      nodes.push(<h3 key={index}>{renderInline(trimmed.slice(3))}</h3>);
    } else if (trimmed.startsWith("# ")) {
      nodes.push(<h2 key={index}>{renderInline(trimmed.slice(2))}</h2>);
    } else if (trimmed.startsWith("> ")) {
      nodes.push(<blockquote key={index}>{renderInline(trimmed.slice(2))}</blockquote>);
    } else {
      nodes.push(<p key={index}>{renderInline(trimmed)}</p>);
    }
  });
  flushList();

  return <>{nodes}</>;
}

function parseFencedCode(content: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  const regex = /```(\w+)?\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({ type: "text", value: content.slice(lastIndex, match.index) });
    }
    blocks.push({
      type: "code",
      language: match[1] || "",
      value: match[2].replace(/\n$/, "")
    });
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < content.length) {
    blocks.push({ type: "text", value: content.slice(lastIndex) });
  }
  return blocks.length > 0 ? blocks : [{ type: "text", value: content }];
}

function renderInline(value: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(value)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(value.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      nodes.push(<code key={match.index}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={match.index}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("*")) {
      nodes.push(<em key={match.index}>{token.slice(1, -1)}</em>);
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        nodes.push(
          <a href={linkMatch[2]} key={match.index} rel="noreferrer" target="_blank">
            {linkMatch[1]}
          </a>
        );
      }
    }
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < value.length) {
    nodes.push(value.slice(lastIndex));
  }
  return nodes;
}
