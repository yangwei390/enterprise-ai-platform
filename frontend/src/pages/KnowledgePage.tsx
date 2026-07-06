import { FormEvent, useState } from "react";
import {
  createKnowledgeBase,
  parseDocument,
  uploadDocument
} from "../api/knowledge";

export default function KnowledgePage() {
  const [kbName, setKbName] = useState("Android Knowledge Base");
  const [kbDescription, setKbDescription] = useState("Mobile architecture docs");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("1");
  const [documentId, setDocumentId] = useState("1");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);

  async function run(action: () => Promise<unknown>) {
    setLoading(true);
    try {
      setResult(await action());
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  function handleCreate(event: FormEvent) {
    event.preventDefault();
    void run(() =>
      createKnowledgeBase({
        name: kbName,
        description: kbDescription,
        embedding_model: ""
      })
    );
  }

  function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setResult({ error: "请选择文件" });
      return;
    }
    void run(() => uploadDocument(Number(knowledgeBaseId), file));
  }

  function handleParse(event: FormEvent) {
    event.preventDefault();
    void run(() => parseDocument(Number(documentId)));
  }

  return (
    <div>
      <div className="page-title">
        <h2>Knowledge Base</h2>
        <p>Create KB, upload document, and trigger parsing.</p>
      </div>
      <div className="two-column">
        <div className="stack">
          <form className="card form" onSubmit={handleCreate}>
            <h3>创建知识库</h3>
            <label>
              Name
              <input value={kbName} onChange={(event) => setKbName(event.target.value)} />
            </label>
            <label>
              Description
              <textarea
                value={kbDescription}
                onChange={(event) => setKbDescription(event.target.value)}
              />
            </label>
            <button type="submit" disabled={loading}>POST /kb</button>
          </form>
          <form className="card form" onSubmit={handleUpload}>
            <h3>上传文档</h3>
            <label>
              Knowledge Base ID
              <input
                value={knowledgeBaseId}
                onChange={(event) => setKnowledgeBaseId(event.target.value)}
              />
            </label>
            <label>
              File
              <input type="file" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            </label>
            <button type="submit" disabled={loading}>POST /documents/upload</button>
          </form>
          <form className="card form" onSubmit={handleParse}>
            <h3>解析文档</h3>
            <label>
              Document ID
              <input value={documentId} onChange={(event) => setDocumentId(event.target.value)} />
            </label>
            <button type="submit" disabled={loading}>POST /documents/{documentId}/parse</button>
          </form>
        </div>
        <ResultPanel result={result} />
      </div>
    </div>
  );
}

function ResultPanel({ result }: { result: unknown }) {
  return (
    <div className="card result-card">
      <h3>Result</h3>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}
