"""Agent 节点函数
四类处理节点，来自 CC 版 orchestrator 的完整逻辑，
适配使用 QM-newer 的枚举常量和基础设施。
"""

from src.core.enums import CallAction


def route_by_type(state: dict) -> str:
    """条件路由：根据 type_id 选择处理节点"""
    type_id = state.get("type_id", "general")

    if type_id in ("scam", "scam_risk", "telemarketing", "game_promo"):
        return "scam"
    if type_id in ("food_delivery", "express", "taxi_arrived", "bank"):
        return "business"
    if type_id in ("family", "leader", "urgent"):
        return "urgent"
    return "normal"


def handle_scam(state: dict, llm_client, card_builder, profile_store, caller_number: str) -> dict:
    """诈骗处理节点：分析 + 卡片 + 更新画像"""
    text = state.get("call_text", "")
    confidence = state.get("confidence", 0.8)
    type_name = state.get("call_type_name", "诈骗拦截")

    # LLM 分析
    try:
        analysis = llm_client.chat(
            messages=[{"role": "user", "content": f"分析以下来电是否为诈骗，简要说明理由：\n{text}"}],
            response_type="text",
            default_response="疑似诈骗电话",
        )
    except Exception:
        analysis = "疑似诈骗电话"

    # 更新来电画像
    if caller_number:
        profile = profile_store.get_or_create(caller_number)
        profile.add_call("scam", confidence)
        profile_store.update(profile)

    # 构建卡片
    card = card_builder.build_scam_log(
        scam_type=type_name,
        reason=analysis[:100] if analysis else "疑似诈骗",
        confidence=confidence,
    )

    return {
        "agent_reply": "",
        "notification_card": card.to_dict() if hasattr(card, "to_dict") else card,
    }


def handle_business(state: dict, call_types_module, presence, card_builder, conversation_memory, profile_store, caller_number: str) -> dict:
    """业务处理节点（外卖/快递/银行等）：对话 + 卡片 + 画像"""
    text = state.get("call_text", "")
    type_id = state.get("type_id", "general")
    presence_mode = state.get("presence_mode", "free")

    conversation_memory.add_user_message(text)

    # 提取业务信息
    _extract_business_info(conversation_memory.working_memory, text, type_id)

    # 生成回复
    call_type = call_types_module.get_call_type(type_id)
    reply = call_types_module.get_reply(call_type, presence_mode)
    conversation_memory.add_assistant_message(reply)

    # 构建卡片（对话完成时）
    wm = conversation_memory.working_memory
    is_complete = wm.is_complete_for_delivery() if type_id in ("food_delivery", "express") else True

    card = None
    if is_complete:
        if type_id in ("food_delivery", "express"):
            card = card_builder.build_delivery_card(
                call_type="外卖" if type_id == "food_delivery" else "快递",
                company=wm.caller_company or "未知平台",
                item=wm.purpose_detail or "物品",
                location=wm.delivery_location or "待确认",
                notes=wm.delivery_notes or "",
            )
        else:
            card = card_builder.build_message_card(
                caller=wm.caller_identity or "来电者",
                message=text[:100],
                relationship=state.get("call_type_name", "业务"),
            )

    # 更新画像
    if caller_number:
        profile = profile_store.get_or_create(caller_number)
        profile.add_call(type_id, state.get("confidence", 0.8))
        profile_store.update(profile)

    return {
        "agent_reply": reply,
        "notification_card": card.to_dict() if card and hasattr(card, "to_dict") else card,
        "final_action": CallAction.SUMMARY_CARD.value if is_complete else CallAction.CONTINUE_CONVERSATION.value,
    }


def handle_urgent(state: dict, llm_client, call_types_module, presence, card_builder, conversation_memory, profile_store, caller_number: str) -> dict:
    """紧急来电处理节点：评估紧急程度 + 转接"""
    text = state.get("call_text", "")
    presence_mode = state.get("presence_mode", "free")

    conversation_memory.add_user_message(text)

    # LLM 评估紧急程度
    try:
        analysis = llm_client.chat(
            messages=[{"role": "user", "content": f"评估以下来电的紧急程度（high/medium/low），简要说明：\n{text}"}],
            response_type="text",
            default_response='{"should_forward": true, "urgency_level": "high", "agent_text": "好的，我马上通知机主。"}',
        )
        import json
        # Try to parse as JSON
        try:
            result = json.loads(analysis)
        except json.JSONDecodeError:
            result = {"should_forward": True, "urgency_level": "high", "agent_text": "好的，我马上通知机主。"}
    except Exception:
        result = {"should_forward": True, "urgency_level": "high", "agent_text": "好的，我马上通知机主。"}

    should_forward = result.get("should_forward", True)
    urgency = result.get("urgency_level", "high")
    reply = result.get("agent_text", "好的，我马上通知机主。")

    conversation_memory.add_assistant_message(reply)

    if should_forward:
        card = card_builder.build_urgent_alert(
            caller=conversation_memory.working_memory.caller_identity or "来电者",
            reason=text[:100],
            urgency_level=urgency,
        )
        effective_action = CallAction.FORWARD.value
    else:
        card = None
        effective_action = CallAction.CONTINUE_CONVERSATION.value
        if presence_mode == "free":
            call_type = call_types_module.get_call_type(state.get("type_id", "general"))
            reply = call_types_module.get_reply(call_type, "free")

    # 更新画像
    if caller_number:
        profile = profile_store.get_or_create(caller_number)
        profile.add_call("urgent", state.get("confidence", 0.8))
        profile_store.update(profile)

    return {
        "agent_reply": reply,
        "notification_card": card.to_dict() if card and hasattr(card, "to_dict") else card,
        "final_action": effective_action,
    }


def handle_normal(state: dict, call_types_module, presence, card_builder, conversation_memory, profile_store, caller_number: str) -> dict:
    """普通来电处理节点：询问意图 + 决定转接"""
    text = state.get("call_text", "")
    type_id = state.get("type_id", "general")
    presence_mode = state.get("presence_mode", "free")
    final_action = state.get("final_action", CallAction.FORWARD.value)

    conversation_memory.add_user_message(text)

    if final_action == CallAction.PROXY.value:
        reply = "您好，机主现在不方便接听电话。请问您有什么事，我可以帮忙转达。"
        conversation_memory.add_assistant_message(reply)
        card = card_builder.build_message_card(
            caller=conversation_memory.working_memory.caller_identity or "来电者",
            message=text[:100],
            relationship="普通来电",
        )
        effective_action = CallAction.CONTINUE_CONVERSATION.value
    else:
        call_type = call_types_module.get_call_type(type_id)
        reply = call_types_module.get_reply(call_type, presence_mode)
        conversation_memory.add_assistant_message(reply)
        card = None
        effective_action = CallAction.CONTINUE_CONVERSATION.value

    # 更新画像
    if caller_number:
        profile = profile_store.get_or_create(caller_number)
        profile.add_call(type_id, state.get("confidence", 0.5))
        profile_store.update(profile)

    return {
        "agent_reply": reply,
        "notification_card": card.to_dict() if card and hasattr(card, "to_dict") else card,
        "final_action": effective_action,
    }


def _extract_business_info(working_memory, text: str, type_id: str) -> None:
    """从对话文本中提取业务关键信息"""
    if type_id == "food_delivery" and not working_memory.caller_identity:
        working_memory.caller_identity = "外卖配送员"
    elif type_id == "express" and not working_memory.caller_identity:
        working_memory.caller_identity = "快递员"

    platforms = ["美团", "饿了么", "顺丰", "京东", "中通", "圆通", "韵达", "申通", "极兔"]
    for p in platforms:
        if p in text and not working_memory.caller_company:
            working_memory.caller_company = p
            break

    location_keywords = ["放", "放在", "送到", "在", "门口", "楼下", "驿站", "快递柜", "前台"]
    for kw in location_keywords:
        idx = text.find(kw)
        if idx >= 0 and not working_memory.delivery_location:
            working_memory.delivery_location = text[idx:idx + 20]
            break

    if type_id == "food_delivery" and not working_memory.caller_purpose:
        working_memory.caller_purpose = "送外卖"
    elif type_id == "express" and not working_memory.caller_purpose:
        working_memory.caller_purpose = "送快递"
