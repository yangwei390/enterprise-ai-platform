from __future__ import annotations

from typing import Any


class CustomerServicePresenter:
    """不调用 LLM 的可信结果 Presenter。"""

    def present(self, state: Any) -> str:
        observations = state.get("observations", [])
        if not observations:
            return str(state.get("final_answer") or "当前没有可展示的业务结果。")
        observation = observations[-1]
        if not observation.get("success", False):
            return "本次业务操作未成功，可信状态未发生变化。"
        tool_name = str(observation.get("tool_name") or "")
        result = observation.get("raw_result")
        if tool_name == "knowledge_search":
            answer = self._knowledge(result)
            notice = self._category_switch_notice(state, knowledge_answer=True)
            return f"{notice}\n{answer}" if notice else answer
        if tool_name in {"search_products", "recommend_products", "compare_products"}:
            response_mode, question_slots = self._response_contract(state)
            if response_mode in {"product_feature_match", "product_use_case_match"}:
                return self._product_match(
                    result,
                    response_mode=response_mode,
                    question_slots=question_slots,
                )
            answer = self._products(result, self._requested_product_type(state))
            if not self._has_product_items(result):
                return answer
            notice = self._category_switch_notice(state, knowledge_answer=False)
            return f"{notice}\n{answer}" if notice else answer
        if tool_name == "query_order":
            return self._order(result)
        if tool_name == "query_logistics":
            return self._logistics(result)
        if tool_name == "create_after_sales_ticket":
            return self._after_sales(result)
        if tool_name == "create_human_handoff":
            return self._handoff(result)
        return str(result)

    @staticmethod
    def _knowledge(result: Any) -> str:
        if not isinstance(result, dict):
            return "当前没有找到可核验的说明书证据。"
        citations = result.get("citations")
        sources = result.get("sources")
        answer = result.get("answer")
        if not answer or not isinstance(citations, list) or not citations or not sources:
            return "当前没有找到可核验的说明书证据。"
        return str(answer)

    @staticmethod
    def _products(result: Any, product_type: str) -> str:
        if not isinstance(result, dict):
            return "商品工具未返回可展示的数据。"
        raw_items = result.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            return (
                f"抱歉呢亲亲，小店目前还没有上架 【{product_type}】 这类宝贝呢 。\n\n"
                "非常感谢您的咨询，我已经把您的需求记在小本本上反馈给采购部门啦！"
                "要不您看看咱们店的主打宝贝？或者告诉我您的具体用途，"
                "我帮您在现货里挑挑看有没有能替代的合适好物~"
            )
        lines: list[str] = []
        for index, raw in enumerate(raw_items, start=1):
            item = raw.get("product", raw) if isinstance(raw, dict) else {}
            name = item.get("name") or item.get("product_code") or "未命名商品"
            facts = [f"{index}. {name}"]
            if item.get("price") is not None:
                facts.append(f"价格 {item['price']} {item.get('currency', '')}".rstrip())
            if item.get("stock_quantity") is not None:
                facts.append(f"库存 {item['stock_quantity']}")
            lines.append("；".join(facts))
        return "\n".join(lines)

    @staticmethod
    def _has_product_items(result: Any) -> bool:
        return (
            isinstance(result, dict)
            and isinstance(result.get("items"), list)
            and bool(result["items"])
        )

    @staticmethod
    def _response_contract(state: Any) -> tuple[str | None, dict[str, Any]]:
        execution = state.get("customer_service_execution")
        transaction = (
            execution.get("pending_transaction") if isinstance(execution, dict) else None
        )
        response_mode = (
            transaction.get("expected_result_type")
            if isinstance(transaction, dict)
            else None
        )
        goal = execution.get("goal") if isinstance(execution, dict) else None
        frame = goal.get("semantic_frame") if isinstance(goal, dict) else None
        slots = frame.get("slots") if isinstance(frame, dict) else None
        return (
            response_mode if isinstance(response_mode, str) else None,
            slots if isinstance(slots, dict) else {},
        )

    @staticmethod
    def _product_match(
        result: Any,
        *,
        response_mode: str,
        question_slots: dict[str, Any],
    ) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("items"), list):
            return "商品工具未返回可核验的数据。"
        if len(result["items"]) != 1 or not isinstance(result["items"][0], dict):
            return "当前没有找到唯一商品，无法核验该问题。"
        item = result["items"][0].get("product", result["items"][0])
        if not isinstance(item, dict):
            return "商品工具未返回可核验的数据。"
        name = str(item.get("name") or item.get("product_code") or "该商品")
        if response_mode == "product_use_case_match":
            target = question_slots.get("use_case")
            evidence = item.get("use_cases")
            label = "适合"
        else:
            target = question_slots.get("feature")
            evidence = item.get("features")
            label = "支持"
        if not isinstance(target, str) or not target.strip():
            return f"当前商品资料无法确认{name}的相关能力。"
        values = [str(value) for value in evidence] if isinstance(evidence, list) else []
        normalized_target = target.strip().casefold()
        matched = any(
            normalized_target in value.casefold() or value.casefold() in normalized_target
            for value in values
            if value
        )
        if matched:
            return f"{name}{label}{target.strip()}。"
        return f"当前商品资料无法确认{name}是否{label}{target.strip()}。"

    @staticmethod
    def _requested_product_type(state: Any) -> str:
        execution = state.get("customer_service_execution")
        goal = execution.get("goal") if isinstance(execution, dict) else None
        frame = goal.get("semantic_frame") if isinstance(goal, dict) else None
        slots = frame.get("slots") if isinstance(frame, dict) else None
        keyword = slots.get("keyword") if isinstance(slots, dict) else None
        return keyword if isinstance(keyword, str) and keyword else "商品"

    @staticmethod
    def _category_switch_notice(state: Any, *, knowledge_answer: bool) -> str | None:
        execution = state.get("customer_service_execution")
        goal = execution.get("goal") if isinstance(execution, dict) else None
        frame = goal.get("semantic_frame") if isinstance(goal, dict) else None
        if not isinstance(frame, dict) or frame.get("intent") != "product_fact_with_selection":
            return None
        slots = frame.get("slots")
        keyword = slots.get("keyword") if isinstance(slots, dict) else None
        product_type = keyword if isinstance(keyword, str) and keyword else "商品"
        if knowledge_answer:
            return (
                f"抱歉，不确定您询问的是哪款{product_type}。"
                f"我已重新查询一款{product_type}并核对说明书："
            )
        return f"抱歉，不确定您询问的是哪款{product_type}。现在为您推荐以下{product_type}："

    @staticmethod
    def _order(result: Any) -> str:
        if not isinstance(result, dict):
            return "订单工具未返回可展示的数据。"
        if result.get("mode") == "list":
            items = result.get("items")
            if not isinstance(items, list) or not items:
                return "当前账户没有可查询的订单。"
            lines = [f"共有 {len(items)} 笔订单："]
            for index, item in enumerate(items, start=1):
                ref = item.get("order_ref") or item.get("order_no") or "未知订单"
                lines.append(f"{index}. {ref}，状态：{item.get('status', '未知')}")
            return "\n".join(lines)
        return "；".join(f"{key}：{value}" for key, value in result.items())

    @staticmethod
    def _logistics(result: Any) -> str:
        if not isinstance(result, dict):
            return "物流工具未返回可展示的数据。"
        return "；".join(f"{key}：{value}" for key, value in result.items())

    @staticmethod
    def _after_sales(result: Any) -> str:
        if not isinstance(result, dict):
            return "售后工具未返回可展示的数据。"
        if result.get("status") == "draft":
            return f"{result.get('summary', '售后草稿已生成')}。请明确回复确认后再提交。"
        return f"售后工单已创建：{result.get('ticket_id', result.get('operation_id', '已受理'))}"

    @staticmethod
    def _handoff(result: Any) -> str:
        if not isinstance(result, dict):
            return "转人工请求未返回可展示的数据。"
        return f"转人工请求已记录：{result.get('handoff_id', result.get('status', '已记录'))}"
