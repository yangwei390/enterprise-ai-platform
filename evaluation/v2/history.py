import json
from pathlib import Path
from typing import Any

from backend.app.config.settings import settings

from evaluation.v2.errors import EvaluationReportError
from evaluation.v2.report import write_report
from evaluation.v2.schemas import EvaluationReportV2


class HistoryManager:
    def __init__(self, report_dir: str | Path | None = None) -> None:
        self.report_dir = Path(report_dir or settings.EVALUATION_REPORT_DIR)
        self.history_dir = self.report_dir / "history"

    def save(self, report: EvaluationReportV2) -> Path:
        path = self.history_dir / report.suite_id / f"{report.started_at}_{report.run_id}.json"
        safe_path = _safe_path(path, self.report_dir)
        write_report(report, safe_path)
        write_report(report, self.report_dir / "report.json")
        Path("evaluation/report.json").write_text(
            (self.report_dir / "report.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return safe_path

    def list_history(self, suite_id: str | None = None) -> list[dict[str, Any]]:
        base = self.history_dir / suite_id if suite_id else self.history_dir
        if not base.exists():
            return []
        files = sorted(base.rglob("*.json"), reverse=True)
        return [{"path": str(path), "run_id": _load_run_id(path)} for path in files]

    def load_report(self, run_id: str, suite_id: str | None = None) -> EvaluationReportV2:
        base = self.history_dir / suite_id if suite_id else self.history_dir
        for path in base.rglob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("run_id") == run_id:
                return EvaluationReportV2.model_validate(data)
        raise EvaluationReportError(f"report not found: {run_id}")

    def latest(self, suite_id: str) -> EvaluationReportV2:
        items = self.list_history(suite_id)
        if not items:
            raise EvaluationReportError(f"no history reports for suite: {suite_id}")
        data = json.loads(Path(items[0]["path"]).read_text(encoding="utf-8"))
        return EvaluationReportV2.model_validate(data)


def _safe_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in [resolved, *resolved.parents]:
        raise EvaluationReportError("report path escapes report directory")
    return resolved


def _load_run_id(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("run_id")
    except Exception:
        return None
