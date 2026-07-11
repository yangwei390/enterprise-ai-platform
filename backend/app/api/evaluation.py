from pathlib import Path
from typing import Any

from backend.app.config.settings import settings
from backend.app.schemas import ApiResponse, success
from evaluation.v2.baseline import BaselineManager
from evaluation.v2.history import HistoryManager
from evaluation.v2.runner import EvaluationRunnerV2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvaluationRunRequest(BaseModel):
    suite: str = "smoke"
    case_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    baseline: str | None = None
    compare: bool = False
    save_baseline: str | None = None
    fail_fast: bool | None = None
    concurrency: int | None = None


class BaselineSaveRequest(BaseModel):
    suite_id: str = "smoke"
    run_id: str | None = None


@router.post("/run", response_model=ApiResponse)
async def run_evaluation(request: EvaluationRunRequest) -> ApiResponse:
    if not settings.EVALUATION_V2_ENABLED:
        raise HTTPException(status_code=400, detail="Evaluation V2 is disabled")

    report = await EvaluationRunnerV2(
        concurrency=request.concurrency,
        fail_fast=request.fail_fast,
    ).arun_suite(
        _resolve_suite(request.suite),
        case_ids=request.case_ids,
        tags=request.tags,
        baseline=request.baseline,
        compare=request.compare,
        save_baseline=request.save_baseline,
    )
    return success(report.model_dump(mode="json"))


@router.get("/reports", response_model=ApiResponse)
def list_reports(suite_id: str = "smoke") -> ApiResponse:
    return success({"reports": HistoryManager().list_history(suite_id)})


@router.get("/reports/{run_id}", response_model=ApiResponse)
def get_report(run_id: str, suite_id: str = "smoke") -> ApiResponse:
    report = HistoryManager().load_report(run_id, suite_id)
    return success(report.model_dump(mode="json"))


@router.get("/baselines", response_model=ApiResponse)
def list_baselines() -> ApiResponse:
    return success({"baselines": BaselineManager().list_baselines()})


@router.post("/baselines/{name}", response_model=ApiResponse)
def save_baseline(name: str, request: BaselineSaveRequest | None = None) -> ApiResponse:
    manager = BaselineManager()
    history = HistoryManager()
    request = request or BaselineSaveRequest()
    report = (
        history.load_report(request.run_id, request.suite_id)
        if request.run_id
        else history.latest(request.suite_id)
    )
    manager.save_baseline(name, report)
    return success({"name": name, "suite_id": report.suite_id, "run_id": report.run_id})


@router.delete("/baselines/{name}", response_model=ApiResponse)
def delete_baseline(name: str) -> ApiResponse:
    if settings.APP_ENV == "prod":
        raise HTTPException(status_code=403, detail="baseline deletion disabled in prod")
    BaselineManager().delete_baseline(name)
    return success({"deleted": name})


def evaluation_debug_state() -> dict[str, Any]:
    from evaluation.v2.metrics import list_metrics
    from evaluation.v2.suite import list_available_suites
    from evaluation.v2.targets import list_targets

    return {
        "enabled": settings.EVALUATION_V2_ENABLED,
        "available_suites": list_available_suites(),
        "targets": list_targets(),
        "metrics": list_metrics(),
        "baselines": BaselineManager().list_baselines(),
        "latest_reports": HistoryManager().list_history("smoke")[:5],
        "llm_judge_enabled": settings.EVALUATION_LLM_JUDGE_ENABLED,
        "max_concurrency": settings.EVALUATION_MAX_CONCURRENCY,
        "report_dir": settings.EVALUATION_REPORT_DIR,
    }


def _resolve_suite(suite: str) -> str:
    if "/" in suite or suite.endswith((".yaml", ".yml", ".json")):
        return suite
    return str(Path("evaluation/v2/fixtures/suites") / f"{suite}.yaml")
