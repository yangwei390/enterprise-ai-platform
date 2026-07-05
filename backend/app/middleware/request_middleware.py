import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from backend.app.logger import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        method = scope["method"]
        path = scope["path"]
        client = scope.get("client")
        client_ip = client[0] if client else ""
        start_time = time.perf_counter()

        logger.info(
            f"Request started | request_id={request_id} | method={method} | "
            f"path={path} | client_ip={client_ip}"
        )

        send_with_request_id = self._build_send(
            send=send,
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            start_time=start_time,
        )

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                f"Request failed | request_id={request_id} | method={method} | "
                f"path={path} | client_ip={client_ip} | duration_ms={duration_ms:.2f}"
            )
            raise

    def _build_send(
        self,
        send: Send,
        request_id: str,
        method: str,
        path: str,
        client_ip: str,
        start_time: float,
    ) -> Callable[[Message], Awaitable[None]]:
        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("utf-8")))
                message["headers"] = headers

                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    f"Request finished | request_id={request_id} | method={method} | "
                    f"path={path} | client_ip={client_ip} | "
                    f"status_code={message['status']} | duration_ms={duration_ms:.2f}"
                )

            await send(message)

        return send_with_request_id
