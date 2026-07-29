"""V2 极简会话状态。

砍掉 DST/FSM/slot_change_log，只保留业务必需的上下文。
存储位置：metadata["customer_service"]["v2_state"]。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProductRef:
    """候选商品引用。"""

    ref: str  # "1", "2", "3"
    name: str
    product_code: str
    category: str | None = None


@dataclass
class SessionState:
    """V2 会话状态，跨轮持久化。"""

    # 商品上下文
    active_products: list[ProductRef] = field(default_factory=list)  # 最多 5 个
    last_filters: dict[str, Any] = field(default_factory=dict)

    # 订单上下文
    active_order_ref: str | None = None

    # 确认流程（售后工单待确认）
    pending_action: dict[str, Any] | None = None

    # 元信息
    turn_count: int = 0


_V2_STATE_KEY = "v2_state"
_MAX_ACTIVE_PRODUCTS = 5


def load_session(metadata: dict[str, Any]) -> SessionState:
    """从 metadata 加载 V2 会话状态。"""
    customer_service = metadata.setdefault("customer_service", {})
    stored = customer_service.get(_V2_STATE_KEY)
    if not isinstance(stored, dict):
        return SessionState()
    products = [
        ProductRef(**item)
        for item in stored.get("active_products", [])
        if isinstance(item, dict)
    ]
    return SessionState(
        active_products=products[:_MAX_ACTIVE_PRODUCTS],
        last_filters=stored.get("last_filters", {}),
        active_order_ref=stored.get("active_order_ref"),
        pending_action=stored.get("pending_action"),
        turn_count=stored.get("turn_count", 0),
    )


def save_session(metadata: dict[str, Any], session: SessionState) -> None:
    """持久化 V2 会话状态到 metadata。"""
    session.turn_count += 1
    # 截断 active_products
    session.active_products = session.active_products[:_MAX_ACTIVE_PRODUCTS]
    customer_service = metadata.setdefault("customer_service", {})
    customer_service[_V2_STATE_KEY] = asdict(session)
