class MCPError(RuntimeError):
    pass


class MCPConfigError(MCPError):
    pass


class MCPTransportError(MCPError):
    pass


class MCPConnectionError(MCPError):
    pass


class MCPPermissionError(MCPError):
    pass


class MCPToolError(MCPError):
    pass
