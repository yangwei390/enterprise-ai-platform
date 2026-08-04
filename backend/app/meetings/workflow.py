import json
from typing import Any, TypedDict

from backend.app.exceptions import BusinessException
from backend.app.llms import LLMFactory, LLMMessage, LLMRequest
from backend.app.models import TranscriptSegment
from langgraph.graph import END, StateGraph


class MeetingWorkflowState(TypedDict, total=False):
    segments: list[dict[str, Any]]
    chunks: list[list[dict[str, Any]]]
    partials: list[dict[str, Any]]
    result: dict[str, Any]


class MeetingMinutesWorkflow:
    """V3 LangGraph: chunk -> local summaries -> evidence-aware global minutes."""

    def __init__(self) -> None:
        graph = StateGraph(MeetingWorkflowState)
        graph.add_node("chunk_transcript", self._chunk_node)
        graph.add_node("summarize_chunks", self._summarize_node)
        graph.add_node("merge_minutes", self._merge_node)
        graph.set_entry_point("chunk_transcript")
        graph.add_edge("chunk_transcript", "summarize_chunks")
        graph.add_edge("summarize_chunks", "merge_minutes")
        graph.add_edge("merge_minutes", END)
        self.graph = graph.compile()

    def run(self, segments: list[TranscriptSegment]) -> dict[str, Any]:
        normalized = [
            {
                "id": item.id,
                "speaker_id": item.speaker_id,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "text": item.text,
            }
            for item in segments
        ]
        state = self.graph.invoke({"segments": normalized})
        return state["result"]

    def _chunk_node(self, state: MeetingWorkflowState) -> dict[str, Any]:
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        size = 0
        for segment in state.get("segments", []):
            if current and size + len(segment["text"]) > 12000:
                chunks.append(current)
                current = []
                size = 0
            current.append(segment)
            size += len(segment["text"])
        if current:
            chunks.append(current)
        return {"chunks": chunks}

    def _summarize_node(self, state: MeetingWorkflowState) -> dict[str, Any]:
        return {
            "partials": [
                self._call_json(self._partial_prompt(chunk)) for chunk in state.get("chunks", [])
            ]
        }

    def _merge_node(self, state: MeetingWorkflowState) -> dict[str, Any]:
        valid_ids = [item["id"] for item in state.get("segments", [])]
        prompt = """Merge partial meeting notes into strict JSON with summary, topics,
decisions, action_items, unresolved_questions, risks. Include only confirmed decisions. Every
decision and important action must preserve source_segment_ids and evidence_timestamps. Never infer
an owner or deadline; use exactly 待确认. Distinguish discussion/proposal/confirmed.
""" + json.dumps(
            {"valid_ids": valid_ids, "partials": state.get("partials", [])},
            ensure_ascii=False,
        )
        return {"result": self._call_json(prompt)}

    def _partial_prompt(self, chunk: list[dict[str, Any]]) -> str:
        return """Return strict JSON: summary, topics, decisions, action_items,
unresolved_questions, risks. Preserve segment IDs and timestamps. Do not guess owner/deadline.
Transcript:\n""" + "\n".join(
            f"[{x['id']}|{x['start_time']}-{x['end_time']}|{x['speaker_id']}] {x['text']}"
            for x in chunk
        )

    def _call_json(self, prompt: str) -> dict[str, Any]:
        llm = LLMFactory.get_llm()
        if llm.__class__.__name__.lower().startswith("dummy"):
            raise BusinessException(52101, "LLM_PROVIDER=dummy 不能生成真实会议纪要")
        response = llm.chat(
            LLMRequest(messages=[LLMMessage(role="user", content=prompt)], temperature=0)
        )
        raw = response.answer.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BusinessException(52102, "LLM 未返回有效结构化纪要") from exc
        if not isinstance(value, dict):
            raise BusinessException(52102, "LLM 结构化纪要格式无效")
        return value
