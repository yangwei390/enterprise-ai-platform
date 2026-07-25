from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from threading import Barrier

import pytest
from backend.app.exceptions import BusinessException
from backend.app.schemas.customer_service import (
    AfterSalesConfirmRequest,
    AfterSalesDraftRequest,
    CustomerOrderQuery,
    HumanHandoffRequest,
)
from backend.app.services.customer_service import (
    CUSTOMER_SERVICE_ERROR_CONFIRMATION_REQUIRED,
    CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND,
    CUSTOMER_SERVICE_ERROR_FORBIDDEN,
    CUSTOMER_SERVICE_ERROR_INVALID_STATE,
    CUSTOMER_SERVICE_ORDER_AUTH_FAILED_MESSAGE,
    CustomerServiceMockService,
    CustomerServiceMockStore,
    MockOrder,
    MockOrderItem,
    default_customer_service_mock_store,
)
from pydantic import ValidationError


def make_service(store: CustomerServiceMockStore | None = None) -> CustomerServiceMockService:
    return CustomerServiceMockService(store or default_customer_service_mock_store())


def test_query_order_returns_masked_mock_order() -> None:
    service = make_service()

    result = service.query_order(
        CustomerOrderQuery(order_no="202607240001", customer_phone_last4="5678")
    )

    assert result.mock is True
    assert result.order_no == "2026****0001"
    assert result.phone == "138****5678"
    assert result.address == "广东省深圳市***街道"
    assert result.receiver_name == "张*"
    assert result.items[0].product_code == "JOYOUNG-DJ-M1"


def test_query_order_masks_raw_mock_contact_fields() -> None:
    service = CustomerServiceMockService(
        CustomerServiceMockStore(
            orders={
                "202607240003": MockOrder(
                    order_no="202607240003",
                    customer_phone_last4="4321",
                    status="delivered",
                    placed_at=datetime(2026, 7, 23, 9, 0, tzinfo=UTC),
                    items=(
                        MockOrderItem(
                            product_code="JOYOUNG-DJ-M3",
                            product_name="模拟豆浆机 M3",
                            quantity=1,
                        ),
                    ),
                    amount=Decimal("399.00"),
                    receiver_name="王小明",
                    phone="13800004321",
                    address="北京市朝阳区建国路100号",
                )
            }
        )
    )

    result = service.query_order(
        CustomerOrderQuery(order_no="202607240003", customer_phone_last4="4321")
    )

    assert result.receiver_name == "王*"
    assert result.phone == "138****4321"
    assert result.address == "北京市朝阳区***"


def test_query_order_rejects_wrong_customer_identity_without_details() -> None:
    service = make_service()

    with pytest.raises(BusinessException) as exc_info:
        service.query_order(
            CustomerOrderQuery(order_no="202607240001", customer_phone_last4="0000")
        )

    assert exc_info.value.code == CUSTOMER_SERVICE_ERROR_FORBIDDEN
    assert exc_info.value.message == CUSTOMER_SERVICE_ORDER_AUTH_FAILED_MESSAGE
    assert "138" not in exc_info.value.message
    assert "广东" not in exc_info.value.message


def test_query_order_unknown_and_wrong_identity_return_same_error() -> None:
    service = make_service()

    errors: list[tuple[int, str]] = []
    for query in (
        CustomerOrderQuery(order_no="202607249999", customer_phone_last4="0000"),
        CustomerOrderQuery(order_no="202607240001", customer_phone_last4="0000"),
    ):
        with pytest.raises(BusinessException) as exc_info:
            service.query_order(query)
        errors.append((exc_info.value.code, exc_info.value.message))

    assert errors[0] == errors[1]
    assert errors[0] == (
        CUSTOMER_SERVICE_ERROR_FORBIDDEN,
        CUSTOMER_SERVICE_ORDER_AUTH_FAILED_MESSAGE,
    )


def test_query_logistics_requires_order_ownership_and_masks_tracking() -> None:
    service = make_service()

    result = service.query_logistics(
        CustomerOrderQuery(order_no="202607240001", customer_phone_last4="5678")
    )

    assert result.mock is True
    assert result.order_no == "2026****0001"
    assert result.tracking_no == "SF****0001"
    assert result.events[0].location == "广东省深圳市***网点"


def test_query_logistics_rejects_wrong_customer_identity() -> None:
    service = make_service()

    with pytest.raises(BusinessException) as exc_info:
        service.query_logistics(
            CustomerOrderQuery(order_no="202607240001", customer_phone_last4="0000")
        )

    assert exc_info.value.code == CUSTOMER_SERVICE_ERROR_FORBIDDEN


def test_query_logistics_unknown_and_wrong_identity_return_same_error() -> None:
    service = make_service()

    errors: list[tuple[int, str]] = []
    for query in (
        CustomerOrderQuery(order_no="202607249999", customer_phone_last4="0000"),
        CustomerOrderQuery(order_no="202607240001", customer_phone_last4="0000"),
    ):
        with pytest.raises(BusinessException) as exc_info:
            service.query_logistics(query)
        errors.append((exc_info.value.code, exc_info.value.message))

    assert errors[0] == errors[1]


def test_after_sales_draft_requires_allowed_order_status() -> None:
    service = make_service()

    with pytest.raises(BusinessException) as exc_info:
        service.create_after_sales_draft(
            AfterSalesDraftRequest(
                order_no="202607240002",
                customer_phone_last4="1111",
                issue_type="return",
                issue_description="我想申请退货处理",
            )
        )

    assert exc_info.value.code == CUSTOMER_SERVICE_ERROR_INVALID_STATE


def test_after_sales_draft_is_stable_and_does_not_create_ticket() -> None:
    service = make_service()
    request = AfterSalesDraftRequest(
        order_no="202607240001",
        customer_phone_last4="5678",
        issue_type="repair",
        issue_description="机器启动后有异响，需要售后检查",
    )

    first = service.create_after_sales_draft(request)
    second = service.create_after_sales_draft(request)

    assert first.draft_id == second.draft_id
    assert first.operation_id == first.draft_id
    assert first.requires_confirmation is True
    assert service.store.tickets == {}


def test_after_sales_changed_content_generates_new_draft() -> None:
    service = make_service()

    first = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )
    second = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器无法启动，需要售后检查",
        )
    )

    assert first.draft_id != second.draft_id


def test_after_sales_confirm_requires_explicit_confirmation() -> None:
    service = make_service()
    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )

    with pytest.raises(BusinessException) as exc_info:
        service.confirm_after_sales_ticket(
            AfterSalesConfirmRequest(
                order_no="202607240001",
                customer_phone_last4="5678",
                draft_id=draft.draft_id,
                operation_id=draft.operation_id,
                confirmed=False,
            )
        )

    assert exc_info.value.code == CUSTOMER_SERVICE_ERROR_CONFIRMATION_REQUIRED
    assert service.store.tickets == {}


def test_after_sales_confirm_is_idempotent() -> None:
    service = make_service()
    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )
    request = AfterSalesConfirmRequest(
        order_no="202607240001",
        customer_phone_last4="5678",
        draft_id=draft.draft_id,
        operation_id=draft.operation_id,
        confirmed=True,
    )

    first = service.confirm_after_sales_ticket(request)
    second = service.confirm_after_sales_ticket(request)

    assert first.ticket_id == second.ticket_id
    assert first.idempotent_replay is False
    assert second.idempotent_replay is True
    assert second.idempotency_key == draft.draft_id
    assert len(service.store.tickets) == 1


def test_after_sales_confirm_rejects_cross_order_draft() -> None:
    service = make_service()
    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )

    with pytest.raises(BusinessException) as exc_info:
        service.confirm_after_sales_ticket(
            AfterSalesConfirmRequest(
                order_no="202607240002",
                customer_phone_last4="1111",
                draft_id=draft.draft_id,
                operation_id=draft.operation_id,
                confirmed=True,
            )
        )

    assert exc_info.value.code == CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND
    assert service.store.tickets == {}


def test_after_sales_confirm_rejects_wrong_operation_id_and_missing_draft() -> None:
    service = make_service()
    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )

    with pytest.raises(BusinessException) as wrong_operation:
        service.confirm_after_sales_ticket(
            AfterSalesConfirmRequest(
                order_no="202607240001",
                customer_phone_last4="5678",
                draft_id=draft.draft_id,
                operation_id="mock-draft-aaaaaaaaaaaaaaaaaaaaaaaa",
                confirmed=True,
            )
        )
    with pytest.raises(BusinessException) as missing_draft:
        service.confirm_after_sales_ticket(
            AfterSalesConfirmRequest(
                order_no="202607240001",
                customer_phone_last4="5678",
                draft_id="mock-draft-bbbbbbbbbbbbbbbbbbbbbbbb",
                operation_id="mock-draft-bbbbbbbbbbbbbbbbbbbbbbbb",
                confirmed=True,
            )
        )

    assert wrong_operation.value.code == CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND
    assert missing_draft.value.code == CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND
    assert service.store.tickets == {}


def test_after_sales_confirm_cannot_tamper_draft_content() -> None:
    service = make_service()
    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )

    with pytest.raises(ValidationError):
        AfterSalesConfirmRequest.model_validate(
            {
                "order_no": "202607240001",
                "customer_phone_last4": "5678",
                "draft_id": draft.draft_id,
                "operation_id": draft.operation_id,
                "confirmed": True,
                "issue_type": "return",
            }
        )
    ticket = service.confirm_after_sales_ticket(
        AfterSalesConfirmRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            draft_id=draft.draft_id,
            operation_id=draft.operation_id,
            confirmed=True,
        )
    )

    assert ticket.issue_type == "repair"
    assert "异响" in ticket.issue_description_preview


def test_after_sales_concurrent_confirm_is_atomic_idempotent() -> None:
    service = make_service()
    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="机器启动后有异响，需要售后检查",
        )
    )
    request = AfterSalesConfirmRequest(
        order_no="202607240001",
        customer_phone_last4="5678",
        draft_id=draft.draft_id,
        operation_id=draft.operation_id,
        confirmed=True,
    )
    worker_count = 16
    barrier = Barrier(worker_count)

    def confirm_ticket() -> tuple[str, bool]:
        barrier.wait(timeout=5)
        result = service.confirm_after_sales_ticket(request)
        return result.ticket_id, result.idempotent_replay

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _: confirm_ticket(), range(worker_count)))

    assert {ticket_id for ticket_id, _ in results} == {f"mock-ticket-{draft.draft_id[-12:]}"}
    assert [replay for _, replay in results].count(False) == 1
    assert [replay for _, replay in results].count(True) == worker_count - 1
    assert len(service.store.tickets) == 1


def test_after_sales_and_handoff_mask_sensitive_free_text() -> None:
    service = make_service()
    sensitive_text = (
        "手机号13800004321，身份证11010119900307888X，"
        "银行卡6222021234567890123，地址北京市朝阳区建国路100号"
    )

    draft = service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description=sensitive_text,
        )
    )
    ticket = service.confirm_after_sales_ticket(
        AfterSalesConfirmRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            draft_id=draft.draft_id,
            operation_id=draft.operation_id,
            confirmed=True,
        )
    )
    handoff = service.create_human_handoff(
        HumanHandoffRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            reason="customer_request",
            message=sensitive_text,
        )
    )

    combined = " ".join(
        [
            draft.summary,
            draft.issue_description_preview,
            ticket.issue_description_preview,
            handoff.message_preview,
        ]
    )
    assert "13800004321" not in combined
    assert "11010119900307888X" not in combined
    assert "6222021234567890123" not in combined
    assert "建国路100号" not in combined
    assert "138****4321" in combined
    assert "110101********888X" in combined
    assert "6222********0123" in combined


def test_human_handoff_returns_mock_record_without_fake_agent_details() -> None:
    service = make_service()

    result = service.create_human_handoff(
        HumanHandoffRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            reason="customer_request",
            message="我要人工客服帮我继续处理",
        )
    )

    data = result.model_dump()
    assert result.mock is True
    assert result.handoff_id.startswith("mock-handoff-")
    assert "agent_name" not in data
    assert "wait_time" not in data
    assert "resolution" not in data


def test_service_has_no_database_or_tool_side_effect_methods() -> None:
    service = make_service()

    assert not hasattr(service, "commit")
    assert not hasattr(service, "flush")
    assert not hasattr(service, "get_db")


def test_default_services_share_process_store_for_mock_idempotency() -> None:
    first_service = CustomerServiceMockService()
    second_service = CustomerServiceMockService()
    draft = first_service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="exchange",
            issue_description="共享 Store 生命周期验证，需要售后检查",
        )
    )

    ticket = second_service.confirm_after_sales_ticket(
        AfterSalesConfirmRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            draft_id=draft.draft_id,
            operation_id=draft.operation_id,
            confirmed=True,
        )
    )

    assert ticket.ticket_id == f"mock-ticket-{draft.draft_id[-12:]}"


def test_explicit_stores_are_isolated_from_each_other() -> None:
    first_service = make_service()
    second_service = make_service()
    draft = first_service.create_after_sales_draft(
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="显式 Store 隔离验证，需要售后检查",
        )
    )

    with pytest.raises(BusinessException) as exc_info:
        second_service.confirm_after_sales_ticket(
            AfterSalesConfirmRequest(
                order_no="202607240001",
                customer_phone_last4="5678",
                draft_id=draft.draft_id,
                operation_id=draft.operation_id,
                confirmed=True,
            )
        )

    assert exc_info.value.code == CUSTOMER_SERVICE_ERROR_DRAFT_NOT_FOUND


def test_schema_accepts_phone_last4_leading_zero_and_rejects_invalid_values() -> None:
    query = CustomerOrderQuery(order_no="202607240001", customer_phone_last4="0001")

    assert query.customer_phone_last4 == "0001"
    with pytest.raises(ValidationError):
        CustomerOrderQuery.model_validate(
            {"order_no": "202607240001", "customer_phone_last4": 1}
        )
    for value in ("001", "00001"):
        with pytest.raises(ValidationError):
            CustomerOrderQuery(order_no="202607240001", customer_phone_last4=value)


def test_schema_rejects_extra_fields_strict_bool_and_invalid_ids() -> None:
    with pytest.raises(ValidationError):
        AfterSalesDraftRequest.model_validate(
            {
                "order_no": "202607240001",
                "customer_phone_last4": "5678",
                "issue_type": "repair",
                "issue_description": "机器启动后有异响，需要售后检查",
                "extra": "forbidden",
            }
        )
    for confirmed in (1, "yes", "true"):
        with pytest.raises(ValidationError):
            AfterSalesConfirmRequest.model_validate(
                {
                    "order_no": "202607240001",
                    "customer_phone_last4": "5678",
                    "draft_id": "mock-draft-aaaaaaaaaaaaaaaaaaaaaaaa",
                    "operation_id": "mock-draft-aaaaaaaaaaaaaaaaaaaaaaaa",
                    "confirmed": confirmed,
                }
            )
    with pytest.raises(ValidationError):
        AfterSalesConfirmRequest.model_validate(
            {
                "order_no": "202607240001",
                "customer_phone_last4": "5678",
                "draft_id": "bad",
                "operation_id": "mock-draft-aaaaaaaaaaaaaaaaaaaaaaaa",
                "confirmed": True,
            }
        )


def test_schema_rejects_blank_and_too_long_free_text() -> None:
    with pytest.raises(ValidationError):
        AfterSalesDraftRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            issue_type="repair",
            issue_description="     ",
        )
    with pytest.raises(ValidationError):
        HumanHandoffRequest(
            order_no="202607240001",
            customer_phone_last4="5678",
            reason="customer_request",
            message="x" * 1001,
        )
