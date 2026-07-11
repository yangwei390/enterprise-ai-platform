class AgentLoopError(RuntimeError):
    pass


class AgentToolCallingError(AgentLoopError):
    pass


class AgentToolCallParseError(AgentToolCallingError):
    pass


class AgentUnknownToolError(AgentToolCallingError):
    pass


class AgentReflectionError(AgentLoopError):
    pass


class AgentBudgetExceeded(AgentLoopError):
    pass


class AgentLoopTerminationError(AgentLoopError):
    pass
