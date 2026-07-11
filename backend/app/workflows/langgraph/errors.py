class WorkflowV2Error(RuntimeError):
    pass


class WorkflowDefinitionError(WorkflowV2Error):
    pass


class WorkflowValidationError(WorkflowV2Error):
    pass


class WorkflowNodeError(WorkflowV2Error):
    pass


class WorkflowConditionError(WorkflowV2Error):
    pass


class WorkflowMaxStepsExceeded(WorkflowV2Error):
    pass


class WorkflowCheckpointError(WorkflowV2Error):
    pass


class WorkflowResumeError(WorkflowV2Error):
    pass


class WorkflowNotInterruptedError(WorkflowV2Error):
    pass


class WorkflowAlreadyCompletedError(WorkflowV2Error):
    pass


class WorkflowApprovalError(WorkflowV2Error):
    pass


class WorkflowPermissionError(WorkflowV2Error):
    pass


class WorkflowTimeoutError(WorkflowV2Error):
    pass
