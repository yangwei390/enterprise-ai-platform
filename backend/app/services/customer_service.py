from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from threading import RLock

from backend.app.exceptions import BusinessException
from backend.app.logger import logger
from backend.app.schemas.customer_service import (
    AfterSalesConfirmRequest,
    AfterSalesDraftRequest,
    AfterSalesDraftResult,
    AfterSalesIssueType,
    AfterSalesTicketResult,
    CustomerLogisticsEvent,
    CustomerLogisticsResult,
    CustomerOrderItem,
    CustomerOrderQuery,
    CustomerOrderResult,
    HumanHandoffRequest,
    HumanHandoffResult,
    LogisticsStatus,
    OrderStatus,
)

CUSTOMER_SERVICE_ERROR_INVALID_STATE = 40040
CUSTOMER_SERVICE_ERROR_CONFIRMATION_REQUIRED = 40041
CUSTOMER_SERVICE_ERROR_FORBIDDEN = 40340
CUSTOMER_SERVICE_ERROR_NOT_FOUND = 40440
CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND = 40441
CUSTOMER_SERVICE_ORDER_AUTH_FAILED_MESSAGE = "订单信息校验未通过"
CUSTOMER_SERVICE_DRAFT_AUTH_FAILED_MESSAGE = "售后草稿不存在或已失效"

_MAINLAND_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_MAINLAND_ID_RE = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
_BANK_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)")
_ADDRESS_RE = re.compile(
    r"[\u4e00-\u9fa5]{2,}(?:省|市|自治区|区|县|镇|乡|街道|路|街|巷|号)"
    r"[\u4e00-\u9fa5A-Za-z0-9#\-]{2,}"
)


@dataclass(frozen=True, slots=True)
class MockOrderItem:
    product_code: str
    product_name: str
    quantity: int


@dataclass(frozen=True, slots=True)
class MockOrder:
    order_no: str
    customer_phone_last4: str
    status: OrderStatus
    placed_at: datetime
    items: tuple[MockOrderItem, ...]
    amount: Decimal
    receiver_name: str
    phone: str
    address: str
    tracking_no: str | None = None
    carrier: str | None = None


@dataclass(frozen=True, slots=True)
class MockLogisticsEvent:
    status: LogisticsStatus
    occurred_at: datetime
    description: str
    location: str


@dataclass(frozen=True, slots=True)
class MockLogistics:
    tracking_no: str
    carrier: str
    status: LogisticsStatus
    events: tuple[MockLogisticsEvent, ...]


@dataclass(frozen=True, slots=True)
class MockAfterSalesDraftRecord:
    draft_id: str
    operation_id: str
    order_no: str
    customer_identity: str
    issue_type: AfterSalesIssueType
    issue_description_normalized: str
    issue_description_preview: str
    created_at: datetime


@dataclass(slots=True)
class CustomerServiceMockStore:
    orders: dict[str, MockOrder] = field(default_factory=dict)
    logistics: dict[str, MockLogistics] = field(default_factory=dict)
    drafts: dict[str, MockAfterSalesDraftRecord] = field(default_factory=dict)
    tickets: dict[str, AfterSalesTicketResult] = field(default_factory=dict)
    handoffs: dict[str, HumanHandoffResult] = field(default_factory=dict)
    _ticket_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def save_draft(self, draft: MockAfterSalesDraftRecord) -> None:
        self.drafts[draft.draft_id] = draft

    def get_draft(self, draft_id: str) -> MockAfterSalesDraftRecord | None:
        return self.drafts.get(draft_id)

    def get_or_create_ticket(
        self,
        draft: MockAfterSalesDraftRecord,
    ) -> AfterSalesTicketResult:
        with self._ticket_lock:
            existing = self.tickets.get(draft.draft_id)
            if existing is not None:
                return existing.model_copy(update={"idempotent_replay": True})

            ticket = AfterSalesTicketResult(
                ticket_id=f"mock-ticket-{draft.draft_id[-12:]}",
                order_no=_mask_order_no(draft.order_no),
                status="created",
                issue_type=draft.issue_type,
                issue_description_preview=draft.issue_description_preview,
                idempotency_key=draft.operation_id,
                idempotent_replay=False,
                mock=True,
            )
            self.tickets[draft.draft_id] = ticket
            return ticket.model_copy()


_DEFAULT_STORE: CustomerServiceMockStore | None = None
_DEFAULT_STORE_LOCK = RLock()


def get_default_customer_service_mock_store() -> CustomerServiceMockStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = default_customer_service_mock_store()
    return _DEFAULT_STORE


class CustomerServiceMockService:
    after_sales_allowed_statuses = {"paid", "packed", "shipped", "delivered"}

    def __init__(self, store: CustomerServiceMockStore | None = None) -> None:
        self.store = store or get_default_customer_service_mock_store()

    def query_order(self, query: CustomerOrderQuery) -> CustomerOrderResult:
        logger.info("Customer mock order query started | order_no=%s", query.order_no)
        order = self._get_authorized_order(query)
        result = self._order_result(order)
        logger.info("Customer mock order query succeeded | order_no=%s", query.order_no)
        return result

    def query_logistics(self, query: CustomerOrderQuery) -> CustomerLogisticsResult:
        logger.info("Customer mock logistics query started | order_no=%s", query.order_no)
        order = self._get_authorized_order(query)
        if order.tracking_no is None:
            raise BusinessException(CUSTOMER_SERVICE_ERROR_NOT_FOUND, "该订单暂无物流信息")
        logistics = self.store.logistics.get(order.tracking_no)
        if logistics is None:
            raise BusinessException(CUSTOMER_SERVICE_ERROR_NOT_FOUND, "物流信息不存在")
        result = CustomerLogisticsResult(
            order_no=_mask_order_no(order.order_no),
            tracking_no=_mask_tracking_no(logistics.tracking_no),
            status=logistics.status,
            carrier=logistics.carrier,
            events=[
                CustomerLogisticsEvent(
                    status=event.status,
                    occurred_at=event.occurred_at,
                    description=event.description,
                    location=_mask_location(event.location),
                )
                for event in logistics.events
            ],
            mock=True,
        )
        logger.info("Customer mock logistics query succeeded | order_no=%s", query.order_no)
        return result

    def create_after_sales_draft(
        self,
        request: AfterSalesDraftRequest,
    ) -> AfterSalesDraftResult:
        logger.info("Customer mock after-sales draft started | order_no=%s", request.order_no)
        order = self._get_authorized_order(request)
        self._ensure_after_sales_allowed(order)
        draft_id = self._after_sales_draft_id(request)
        issue_description_normalized = _mask_sensitive_text(
            _normalize_free_text(request.issue_description)
        )
        issue_preview = _preview(issue_description_normalized)
        draft_record = MockAfterSalesDraftRecord(
            draft_id=draft_id,
            operation_id=draft_id,
            order_no=order.order_no,
            customer_identity=_customer_identity_fingerprint(
                order.order_no,
                order.customer_phone_last4,
            ),
            issue_type=request.issue_type,
            issue_description_normalized=issue_description_normalized,
            issue_description_preview=issue_preview,
            created_at=datetime.now(UTC),
        )
        self.store.save_draft(draft_record)
        draft = AfterSalesDraftResult(
            draft_id=draft_record.draft_id,
            operation_id=draft_record.operation_id,
            order_no=_mask_order_no(order.order_no),
            issue_type=draft_record.issue_type,
            issue_description_preview=issue_preview,
            summary=(
                f"模拟售后申请：订单 {_mask_order_no(order.order_no)}，"
                f"问题类型 {draft_record.issue_type}，问题描述 {issue_preview}"
            ),
            requires_confirmation=True,
            mock=True,
        )
        logger.info("Customer mock after-sales draft succeeded | draft_id=%s", draft_id)
        return draft

    def confirm_after_sales_ticket(
        self,
        request: AfterSalesConfirmRequest,
    ) -> AfterSalesTicketResult:
        logger.info(
            "Customer mock after-sales confirm started | draft_id=%s",
            request.draft_id,
        )
        order = self._get_authorized_order(request)
        if not request.confirmed:
            raise BusinessException(
                CUSTOMER_SERVICE_ERROR_CONFIRMATION_REQUIRED,
                "创建售后工单前需要用户明确确认",
            )
        draft = self.store.get_draft(request.draft_id)
        if draft is None or not self._is_draft_authorized(draft, request, order):
            raise BusinessException(
                CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND,
                CUSTOMER_SERVICE_DRAFT_AUTH_FAILED_MESSAGE,
            )

        ticket = self.store.get_or_create_ticket(draft)
        logger.info("Customer mock after-sales confirm succeeded | ticket_id=%s", ticket.ticket_id)
        return ticket

    def create_human_handoff(self, request: HumanHandoffRequest) -> HumanHandoffResult:
        logger.info("Customer mock handoff started | order_no=%s", request.order_no)
        order = self._get_authorized_order(request)
        handoff_id = self._handoff_id(request)
        handoff = HumanHandoffResult(
            handoff_id=handoff_id,
            order_no=_mask_order_no(order.order_no),
            reason=request.reason,
            message_preview=_preview(_mask_sensitive_text(request.message)),
            mock=True,
        )
        self.store.handoffs[handoff_id] = handoff
        logger.info("Customer mock handoff succeeded | handoff_id=%s", handoff_id)
        return handoff

    def _get_authorized_order(self, query: CustomerOrderQuery) -> MockOrder:
        order = self.store.orders.get(query.order_no)
        if order is None or order.customer_phone_last4 != query.customer_phone_last4:
            raise BusinessException(
                CUSTOMER_SERVICE_ERROR_FORBIDDEN,
                CUSTOMER_SERVICE_ORDER_AUTH_FAILED_MESSAGE,
            )
        return order

    def _is_draft_authorized(
        self,
        draft: MockAfterSalesDraftRecord,
        request: AfterSalesConfirmRequest,
        order: MockOrder,
    ) -> bool:
        if draft.operation_id != request.operation_id:
            return False
        if draft.order_no != order.order_no:
            return False
        return draft.customer_identity == _customer_identity_fingerprint(
            order.order_no,
            order.customer_phone_last4,
        )

    def _ensure_after_sales_allowed(self, order: MockOrder) -> None:
        if order.status not in self.after_sales_allowed_statuses:
            raise BusinessException(
                CUSTOMER_SERVICE_ERROR_INVALID_STATE,
                "当前订单状态不支持模拟售后",
            )

    def _order_result(self, order: MockOrder) -> CustomerOrderResult:
        return CustomerOrderResult(
            order_no=_mask_order_no(order.order_no),
            status=order.status,
            placed_at=order.placed_at,
            items=[
                CustomerOrderItem(
                    product_code=item.product_code,
                    product_name=item.product_name,
                    quantity=item.quantity,
                )
                for item in order.items
            ],
            amount=f"{order.amount:.2f}",
            currency="CNY",
            receiver_name=_mask_name(order.receiver_name),
            phone=_mask_phone(order.phone),
            address=_mask_address(order.address),
            mock=True,
        )

    def _after_sales_draft_id(self, request: AfterSalesDraftRequest) -> str:
        key = "|".join(
            [
                request.order_no,
                request.customer_phone_last4,
                request.issue_type,
                request.issue_description,
            ]
        )
        return "mock-draft-" + sha256(key.encode("utf-8")).hexdigest()[:24]

    def _handoff_id(self, request: HumanHandoffRequest) -> str:
        key = "|".join(
            [
                request.order_no,
                request.customer_phone_last4,
                request.reason,
                request.message,
            ]
        )
        return "mock-handoff-" + sha256(key.encode("utf-8")).hexdigest()[:24]


def default_customer_service_mock_store() -> CustomerServiceMockStore:
    return CustomerServiceMockStore(
        orders={
            "202607240001": MockOrder(
                order_no="202607240001",
                customer_phone_last4="5678",
                status="delivered",
                placed_at=datetime(2026, 7, 20, 10, 30, tzinfo=UTC),
                items=(
                    MockOrderItem(
                        product_code="JOYOUNG-DJ-M1",
                        product_name="模拟豆浆机 M1",
                        quantity=1,
                    ),
                ),
                amount=Decimal("299.00"),
                receiver_name="张*",
                phone="138****5678",
                address="广东省深圳市***街道",
                tracking_no="SF202607240001",
                carrier="顺丰速运",
            ),
            "202607240002": MockOrder(
                order_no="202607240002",
                customer_phone_last4="1111",
                status="cancelled",
                placed_at=datetime(2026, 7, 21, 14, 10, tzinfo=UTC),
                items=(
                    MockOrderItem(
                        product_code="JOYOUNG-DJ-M2",
                        product_name="模拟豆浆机 M2",
                        quantity=1,
                    ),
                ),
                amount=Decimal("199.00"),
                receiver_name="李*",
                phone="139****1111",
                address="上海市浦东新区***路",
            ),
        },
        logistics={
            "SF202607240001": MockLogistics(
                tracking_no="SF202607240001",
                carrier="顺丰速运",
                status="delivered",
                events=(
                    MockLogisticsEvent(
                        status="picked_up",
                        occurred_at=datetime(2026, 7, 21, 9, 0, tzinfo=UTC),
                        description="模拟包裹已揽收",
                        location="广东省深圳市***网点",
                    ),
                    MockLogisticsEvent(
                        status="delivered",
                        occurred_at=datetime(2026, 7, 22, 16, 30, tzinfo=UTC),
                        description="模拟包裹已签收",
                        location="广东省深圳市***街道",
                    ),
                ),
            )
        },
    )


def _mask_order_no(order_no: str) -> str:
    if len(order_no) <= 8:
        return order_no
    return f"{order_no[:4]}****{order_no[-4:]}"


def _mask_tracking_no(tracking_no: str) -> str:
    if len(tracking_no) <= 6:
        return tracking_no
    return f"{tracking_no[:2]}****{tracking_no[-4:]}"


def _mask_location(location: str) -> str:
    if "***" in location:
        return location
    return f"{location[:6]}***"


def _mask_name(name: str) -> str:
    if "*" in name:
        return name
    return f"{name[:1]}*"


def _mask_phone(phone: str) -> str:
    if "*" in phone:
        return phone
    if len(phone) < 7:
        return phone
    return f"{phone[:3]}****{phone[-4:]}"


def _mask_address(address: str) -> str:
    if "***" in address:
        return address
    if len(address) <= 6:
        return address
    return f"{address[:6]}***"


def _preview(text: str, limit: int = 80) -> str:
    normalized = _normalize_free_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "..."


def _normalize_free_text(text: str) -> str:
    return " ".join(text.split())


def _mask_sensitive_text(text: str) -> str:
    masked = _MAINLAND_PHONE_RE.sub(lambda match: _mask_phone(match.group(0)), text)
    masked = _MAINLAND_ID_RE.sub(
        lambda match: f"{match.group(0)[:6]}********{match.group(0)[-4:]}",
        masked,
    )
    masked = _BANK_CARD_RE.sub(_mask_bank_card_match, masked)
    return _ADDRESS_RE.sub(lambda match: _mask_address(match.group(0)), masked)


def _mask_bank_card_match(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = re.sub(r"\D", "", value)
    if len(digits) < 16:
        return value
    return f"{digits[:4]}********{digits[-4:]}"


def _customer_identity_fingerprint(order_no: str, phone_last4: str) -> str:
    return sha256(f"{order_no}|{phone_last4}".encode()).hexdigest()
