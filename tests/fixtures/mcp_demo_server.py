import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("enterprise-ai-platform-demo")


@mcp.tool()
def mcp_echo(text: str) -> dict:
    return {"text": text}


@mcp.tool()
def mcp_add(a: int, b: int) -> dict:
    return {"result": a + b}


if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "streamable-http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8899
        mcp.settings.port = port
        mcp.settings.host = "127.0.0.1"
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
