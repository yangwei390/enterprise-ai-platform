from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

AfterSalesIssueType = Literal["quality", "repair", "return", "exchange", "other"]
AfterSalesStatus = Literal["draft", "created"]
HandoffReason = Literal["customer_request", "complaint", "tool_unavailable", "other"]
LogisticsStatus = Literal[
    "pending",
    "picked_up",
    "in_transit",
    "out_for_delivery",
    "delivered",
    "exception",
    "returned",
]
OrderStatus = Literal[
    "created",
    "paid",
    "packed",
    "shipped",
    "delivered",
    "cancelled",
    "refunding",
    "refunded",
    "after_sales",
]


class CustomerOrderQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str = Field(min_length=4, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{3,63}$")
    customer_phone_last4: str = Field(pattern=r"^\d{4}$")

    @field_validator("order_no")
    @classmethod
    def validate_order_no(cls, value: str) -> str:
        return value.strip()

    @field_validator("customer_phone_last4")
    @classmethod
    def validate_phone_last4(cls, value: str) -> str:
        return value.strip()


class CustomerOrderItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    product_code: str
    product_name: str
    quantity: int = Field(ge=1)


class CustomerOrderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str
    status: OrderStatus
    placed_at: datetime
    items: list[CustomerOrderItem]
    amount: str
    currency: str = "CNY"
    receiver_name: str
    phone: str
    address: str
    mock: bool = True


class CustomerLogisticsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: LogisticsStatus
    occurred_at: datetime
    description: str
    location: str


class CustomerLogisticsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_no: str
    tracking_no: str
    status: LogisticsStatus
    carrier: str
    events: list[CustomerLogisticsEvent]
    mock: bool = True


class AfterSalesDraftRequest(CustomerOrderQuery):
    issue_type: AfterSalesIssueType = "other"
    issue_description: str = Field(min_length=5, max_length=1000)

    @field_validator("issue_description")
    @classmethod
    def validate_issue_description(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 5:
            raise ValueError("issue_description 不能少于 5 个字符")
        return normalized


class AfterSalesDraftResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    draft_id: str
    operation_id: str
    order_no: str
    issue_type: AfterSalesIssueType
    issue_description_preview: str
    summary: str
    requires_confirmation: bool = True
    mock: bool = True


class AfterSalesConfirmRequest(CustomerOrderQuery):
    draft_id: str = Field(
        min_length=35,
        max_length=35,
        pattern=r"^mock-draft-[0-9a-f]{24}$",
    )
    operation_id: str = Field(
        min_length=35,
        max_length=35,
        pattern=r"^mock-draft-[0-9a-f]{24}$",
    )
    confirmed: StrictBool

    @field_validator("draft_id")
    @classmethod
    def validate_draft_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return value.strip()


class AfterSalesTicketResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticket_id: str
    order_no: str
    status: AfterSalesStatus
    issue_type: AfterSalesIssueType
    issue_description_preview: str
    idempotency_key: str
    idempotent_replay: bool = False
    mock: bool = True


class HumanHandoffRequest(CustomerOrderQuery):
    reason: HandoffReason = "customer_request"
    message: str = Field(min_length=2, max_length=1000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 2:
            raise ValueError("message 不能少于 2 个字符")
        return normalized


class HumanHandoffResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    handoff_id: str
    order_no: str
    reason: HandoffReason
    message_preview: str
    mock: bool = True
