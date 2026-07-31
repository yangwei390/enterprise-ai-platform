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
            answer = self._products(result)
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
    def _products(result: Any) -> str:
        if not isinstance(result, dict):
            return "商品工具未返回可展示的数据。"
        raw_items = result.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            return "没有找到符合条件的商品。"
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
