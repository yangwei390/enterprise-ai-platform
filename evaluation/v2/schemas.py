from typing import Any

from pydantic import BaseModel, Field, field_validator

SUPPORTED_TARGETS = {"rag", "generation", "agent", "tool", "mcp", "workflow"}


class EvaluationCase(BaseModel):
    id: str
    name: str | None = None
    target: str = "rag"
    query: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    group: str | None = None
    enabled: bool = True
    timeout_seconds: float | None = None
    requires: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        if value not in SUPPORTED_TARGETS:
            raise ValueError(f"unsupported evaluation target: {value}")
        return value


class EvaluationSuite(BaseModel):
    id: str
    name: str
    version: str = "1.0"
    description: str | None = None
    cases: list[EvaluationCase] = Field(default_factory=list)
    default_metrics: list[str] = Field(default_factory=list)
    default_thresholds: dict[str, Any] = Field(default_factory=dict)
    concurrency: int = 4
    timeout_seconds: float = 120
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationContext(BaseModel):
    run_id: str
    suite_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationTargetResult(BaseModel):
    target: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None
    sources: list[Any] = Field(default_factory=list)
    citations: list[Any] = Field(default_factory=list)
    chunks: list[Any] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


class MetricResult(BaseModel):
    name: str
    value: Any = None
    passed: bool = True
    threshold: Any = None
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationCaseResult(BaseModel):
    case_id: str
    target: str
    status: str
    passed: bool
    duration_ms: float
    metrics: list[MetricResult] = Field(default_factory=list)
    target_result_summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReportV2(BaseModel):
    report_version: str = "2.0"
    run_id: str
    suite_id: str
    suite_name: str
    suite_version: str
    started_at: str
    finished_at: str
    duration_ms: float
    passed: bool
    pass_rate: float
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    target_summary: dict[str, Any] = Field(default_factory=dict)
    metric_summary: dict[str, Any] = Field(default_factory=dict)
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    cost_summary: dict[str, Any] = Field(default_factory=dict)
    regression_summary: dict[str, Any] = Field(default_factory=dict)
    cases: list[EvaluationCaseResult] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
