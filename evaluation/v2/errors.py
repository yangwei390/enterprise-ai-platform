class EvaluationError(RuntimeError):
    pass


class EvaluationCaseError(EvaluationError):
    pass


class EvaluationSuiteError(EvaluationError):
    pass


class EvaluationTargetError(EvaluationError):
    pass


class EvaluationMetricError(EvaluationError):
    pass


class EvaluationThresholdError(EvaluationError):
    pass


class EvaluationBaselineError(EvaluationError):
    pass


class EvaluationRegressionError(EvaluationError):
    pass


class EvaluationReportError(EvaluationError):
    pass


class EvaluationTimeoutError(EvaluationError):
    pass
