import json
import re
from pathlib import Path

from backend.app.config.settings import settings

from evaluation.v2.errors import EvaluationBaselineError
from evaluation.v2.report import write_report
from evaluation.v2.schemas import EvaluationReportV2


class BaselineManager:
    def __init__(self, report_dir: str | Path | None = None) -> None:
        self.baseline_dir = Path(report_dir or settings.EVALUATION_REPORT_DIR) / "baselines"

    def save_baseline(self, name: str, report: EvaluationReportV2) -> Path:
        safe_name = self._safe_name(name)
        path = self.baseline_dir / f"{safe_name}.json"
        write_report(report, path)
        return path

    def load_baseline(self, name: str) -> EvaluationReportV2:
        safe_name = self._safe_name(name)
        path = self.baseline_dir / f"{safe_name}.json"
        if not path.exists():
            raise EvaluationBaselineError(f"baseline not found: {safe_name}")
        return EvaluationReportV2.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list_baselines(self) -> list[str]:
        if not self.baseline_dir.exists():
            return []
        return sorted(path.stem for path in self.baseline_dir.glob("*.json"))

    def delete_baseline(self, name: str) -> None:
        safe_name = self._safe_name(name)
        path = self.baseline_dir / f"{safe_name}.json"
        if path.exists():
            path.unlink()

    def _safe_name(self, name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
            raise EvaluationBaselineError("invalid baseline name")
        return name
