"""
主调度器 (Orchestrator) - 整合版
基于 QM-newer 的架构骨架 + CC 的完整功能实现。

整合模块:
  - LLM: 多后端 DeepSeek/GLM/Qwen (QM-newer)
  - 检索: BM25 + 向量 双路混合 (QM-newer)
  - 枚举常量 (QM-newer)
  - 来电类型 + 回复模板 (CC)
  - 三层对话记忆 (CC)
  - 来电画像 + 信任评分 (CC)
  - 习惯学习 (CC)
  - 通知卡片 (CC)
  - 通话记录 (CC)
  - 通话录音 (CC)
  - 机主状态 (CC)
"""

import time
import uuid
from typing import Optional

from langgraph.graph import StateGraph, END

from src.core.enums import CallAction
from src.core.llm_client import LLMClient
from src.classification.classifier import ThreeTierClassifier
from src.orchestration.state import OrchestratorState
from src.orchestration.nodes import route_by_type, handle_scam, handle_business, handle_urgent, handle_normal
from src.agents.call_types import get_call_type, get_reply, get_action, CALL_TYPES
from src.agents.conversation_memory import ConversationMemory
from src.knowledge.profile_store import CallerProfileStore, CallerProfile
from src.knowledge.habit_learner import HabitLearner
from src.knowledge.knowledge_expander import KnowledgeExpander
from src.notification.card_builder import CardBuilder, NotificationStore
from src.store.call_logger import CallLogger
from src.utils.logger import setup_logger
from src.utils.presence import get_presence
from src.orchestration.nodes import _extract_business_info

logger = setup_logger(__name__)


class CallOrchestrator:
    """基于 LangGraph 的多智能体调度器（整合版）"""

    def __init__(self, config: dict):
        self.config = config

        # --- LLM 客户端（QM-newer 多后端）---
        self.llm_client = LLMClient(config)

        # --- 分类器 ---
        self.classifier = ThreeTierClassifier(config, retriever=None, llm_client=self.llm_client)

        # --- 业务处理 ---
        self.card_builder = CardBuilder()
        self.conversation_memory = ConversationMemory(
            max_short_term=config.get("conversation_memory", {}).get("max_short_term", 50)
        )

        # --- 来电画像（CC）---
        profile_cfg = config.get("caller_profile", {})
        self.caller_profile_store = CallerProfileStore(
            persist_path=profile_cfg.get("persist_path", "./data/profiles/caller_profiles.json")
        )

        # --- 习惯学习（CC）---
        self.habit_learner = HabitLearner(config)

        # --- 知识库扩展 ---
        self.knowledge_expander = KnowledgeExpander()

        # --- 通知系统（CC）---
        notif_cfg = config.get("notification", {})
        self.notification_store = NotificationStore(
            persist_path=notif_cfg.get("persist_path", "./data/notifications")
        )

        # --- 通话记录（CC）---
        log_cfg = config.get("call_log", {})
        self.call_logger = CallLogger(
            persist_dir=log_cfg.get("persist_dir", "./data/call_logs")
        )

        # --- 通话录音（CC）---
        recorder_cfg = config.get("recorder", {})
        try:
            from src.voice.recorder import CallRecorder
            self.recorder = CallRecorder(
                persist_dir=recorder_cfg.get("persist_dir", "./data/recordings")
            )
        except Exception as e:
            logger.warning(f"录音模块初始化失败（将跳过录音功能）: {e}")
            self.recorder = None

        # --- 机主状态（CC）---
        self.presence = get_presence()

        # --- 构建 LangGraph ---
        self.graph = self._build_graph()
        logger.info("CallOrchestrator 整合版初始化完成（QM-newer 架构 + CC 功能实现）")

    def _init_enhanced_retrieval(self, config: dict) -> None:
        """初始化混合检索系统（QM-newer BM25 + 向量 双路）"""
        from src.retrieval.embedder import Embedder
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import Reranker

        self.embedder = None
        self.hybrid_retriever = None
        self.reranker = None

        try:
            embedder = Embedder(config)
            self.hybrid_retriever = HybridRetriever(config, embedder)
            hybrid_cfg = config.get("hybrid_retrieval", {})
            self.reranker = Reranker(
                embedder=embedder,
                semantic_weight=hybrid_cfg.get("reranker_semantic_weight", 0.5),
                keyword_weight=hybrid_cfg.get("reranker_keyword_weight", 0.3),
            )
            logger.info("混合检索系统初始化完成（BM25 + 向量 双路）")
        except Exception as e:
            logger.warning(f"混合检索初始化失败: {e}")

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        graph = StateGraph(OrchestratorState)

        graph.add_node("classify", self._node_classify)
        graph.add_node("lookup_profile", self._node_lookup_profile)
        graph.add_node("infer_presence", self._node_infer_presence)
        graph.add_node("route", self._node_route)
        graph.add_node("handle_scam", self._node_handle_scam)
        graph.add_node("handle_business", self._node_handle_business)
        graph.add_node("handle_urgent", self._node_handle_urgent)
        graph.add_node("handle_normal", self._node_handle_normal)
        graph.add_node("notify", self._node_notify)

        graph.set_entry_point("classify")

        graph.add_edge("classify", "lookup_profile")
        graph.add_edge("lookup_profile", "infer_presence")
        graph.add_edge("infer_presence", "route")
        graph.add_conditional_edges(
            "route",
            route_by_type,
            {
                "scam": "handle_scam",
                "business": "handle_business",
                "urgent": "handle_urgent",
                "normal": "handle_normal",
            },
        )
        graph.add_edge("handle_scam", "notify")
        graph.add_edge("handle_business", "notify")
        graph.add_edge("handle_urgent", "notify")
        graph.add_edge("handle_normal", "notify")
        graph.add_edge("notify", END)

        return graph.compile()

    # ============================================================
    # LangGraph 节点
    # ============================================================

    def _node_classify(self, state: dict) -> dict:
        """分类节点：三级分类器"""
        text = state.get("call_text", "")
        if not text:
            return {"type_id": "general", "call_type_name": "未知", "confidence": 0.0, "classify_method": "empty"}

        result = self.classifier.classify(text)
        return {
            "type_id": result.type_id,
            "call_type_name": result.call_type.display_name if hasattr(result.call_type, 'display_name') else result.type_id,
            "confidence": result.confidence,
            "classify_method": result.method,
        }

    def _node_lookup_profile(self, state: dict) -> dict:
        """画像查询节点：查看来电者历史（CC）"""
        caller_number = state.get("caller_number", "")
        if not caller_number:
            return {"caller_profile": None}

        profile = self.caller_profile_store.lookup(caller_number)
        if profile:
            logger.info(f"来电画像: {caller_number} 信任={profile.trust_score:.2f} 标签={profile.tags} 来电={profile.call_count}次")
            # 设置长期记忆
            self.conversation_memory.set_long_term(profile.to_dict())
            return {"caller_profile": profile.to_dict()}
        return {"caller_profile": None}

    def _node_infer_presence(self, state: dict) -> dict:
        """习惯推断节点：根据习惯推断机主状态（CC）"""
        mode, reason = self.habit_learner.infer_presence_mode()
        if mode != "free":
            self.presence.set(mode, reason)
        return {"presence_mode": mode, "presence_reason": reason}

    def _node_route(self, state: dict) -> dict:
        """路由节点：根据分类结果 + 机主状态 + 来电画像决定路由"""
        type_id = state.get("type_id", "general")
        presence_mode = state.get("presence_mode", "free")
        profile_dict = state.get("caller_profile")

        # 黑名单直接走诈骗
        if profile_dict and profile_dict.get("is_blacklisted"):
            logger.info("来电者在黑名单中，路由到诈骗处理")
            return {"final_action": CallAction.REJECT.value}

        # 白名单直接走转接
        if profile_dict and profile_dict.get("is_whitelisted"):
            logger.info("来电者在白名单中，优先转接")
            if type_id in ("scam",):
                return {"final_action": CallAction.REJECT.value}
            return {"final_action": CallAction.FORWARD.value}

        # 机主繁忙/免打扰时，只有紧急来电转接
        if presence_mode in ("busy", "dnd"):
            if type_id in ("scam",):
                return {"final_action": CallAction.REJECT.value}
            if type_id in ("family", "urgent"):
                return {"final_action": CallAction.FORWARD.value}
            return {"final_action": CallAction.PROXY.value}

        # 正常模式：使用 CC 的动作路由
        call_type = get_call_type(type_id)
        action = get_action(call_type, presence_mode)
        return {"final_action": action}

    def _node_handle_scam(self, state: dict) -> dict:
        """诈骗处理节点"""
        caller_number = state.get("caller_number", "")
        return handle_scam(state, self.llm_client, self.card_builder, self.caller_profile_store, caller_number)

    def _node_handle_business(self, state: dict) -> dict:
        """业务处理节点"""
        import src.agents.call_types as ct
        caller_number = state.get("caller_number", "")
        return handle_business(state, ct, self.presence, self.card_builder, self.conversation_memory, self.caller_profile_store, caller_number)

    def _node_handle_urgent(self, state: dict) -> dict:
        """紧急来电处理节点"""
        import src.agents.call_types as ct
        caller_number = state.get("caller_number", "")
        return handle_urgent(state, self.llm_client, ct, self.presence, self.card_builder, self.conversation_memory, self.caller_profile_store, caller_number)

    def _node_handle_normal(self, state: dict) -> dict:
        """普通来电处理节点"""
        import src.agents.call_types as ct
        caller_number = state.get("caller_number", "")
        return handle_normal(state, ct, self.presence, self.card_builder, self.conversation_memory, self.caller_profile_store, caller_number)

    def _node_notify(self, state: dict) -> dict:
        """通知节点：保存卡片 + 记录通话"""
        card_dict = state.get("notification_card")
        if card_dict:
            try:
                if hasattr(card_dict, 'title'):
                    from src.notification.card_builder import NotificationCard
                    card = NotificationCard(**card_dict) if isinstance(card_dict, dict) else card_dict
                    self.notification_store.save(card)
            except Exception as e:
                logger.warning(f"保存通知卡片失败: {e}")

        # 记录通话
        try:
            session_id = self.call_logger.start_session(
                caller_number=state.get("caller_number", ""),
                caller_text=state.get("call_text", ""),
            )
            self.call_logger.log_classification(
                session_id,
                state.get("type_id", "general"),
                state.get("call_type_name", "未知"),
                state.get("confidence", 0.0),
                state.get("classify_method", "unknown"),
            )
            self.call_logger.log_action(
                session_id,
                state.get("final_action", "unknown"),
                state.get("agent_reply", ""),
            )
            self.call_logger.end_session(
                session_id,
                notification_card=card_dict,
            )
            if self.recorder:
                try:
                    self.recorder.update_recording_info(
                        session_id,
                        call_type=state.get("type_id", ""),
                        final_action=state.get("final_action", ""),
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"记录通话失败: {e}")

        return {}

    # ============================================================
    # 公共接口
    # ============================================================

    def run(self, call_text: str, caller_number: str = "") -> dict:
        """处理一次来电"""
        self.conversation_memory.reset()

        call_id = f"call_{uuid.uuid4().hex[:8]}"
        if self.recorder:
            try:
                self.recorder.start(call_id, caller_number=caller_number)
            except Exception as e:
                logger.debug(f"开始录音失败: {e}")

        initial_state = {"call_text": call_text, "caller_number": caller_number}
        try:
            result = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"LangGraph 执行失败: {e}")
            result = {"final_action": CallAction.ERROR.value, "final_message": str(e), "agent_reply": ""}

        if self.recorder:
            try:
                self.recorder.stop()
            except Exception:
                pass

        return dict(result)

    def resume_conversation(self, previous_result: dict, new_text: str) -> dict:
        """继续多轮对话（CC 完整逻辑）"""
        logger.info(f"继续对话: {new_text[:50]}...")
        prev_type = previous_result.get("type_id", "general")

        classify_result = self.classifier.classify(new_text)
        new_type = classify_result.type_id

        # 诈骗检测
        if new_type in ("scam", "scam_risk", "telemarketing", "game_promo"):
            caller_number = previous_result.get("caller_number", "")
            if caller_number:
                profile = self.caller_profile_store.get_or_create(caller_number)
                profile.add_call(new_type, classify_result.confidence)
                self.caller_profile_store.update(profile)
            card = self.card_builder.build_scam_log(
                scam_type=classify_result.type_id,
                reason="多轮对话中检测到诈骗特征",
                confidence=classify_result.confidence,
            )
            return {
                "final_action": CallAction.REJECT.value,
                "type_id": new_type,
                "call_type_name": classify_result.type_id,
                "confidence": classify_result.confidence,
                "classify_method": classify_result.method,
                "agent_reply": "",
                "notification_card": card.to_dict(),
            }

        # 业务类继续对话
        import src.agents.call_types as ct
        if new_type in ("food_delivery", "express", "taxi_arrived", "bank"):
            self.conversation_memory.add_user_message(new_text)
            from src.orchestration.nodes import _extract_business_info; _extract_business_info(self.conversation_memory.working_memory, new_text, new_type)
            call_type = ct.get_call_type(new_type)
            reply = ct.get_reply(call_type, self.presence.get_mode())
            self.conversation_memory.add_assistant_message(reply)

            wm = self.conversation_memory.working_memory
            is_complete = wm.is_complete_for_delivery() if new_type in ("food_delivery", "express") else True
            card = None
            if is_complete:
                if new_type in ("food_delivery", "express"):
                    card = self.card_builder.build_delivery_card(
                        call_type="外卖" if new_type == "food_delivery" else "快递",
                        company=wm.caller_company or "未知平台",
                        item=wm.purpose_detail or "物品",
                        location=wm.delivery_location or "待确认",
                        notes=wm.delivery_notes or "",
                    )
                else:
                    card = self.card_builder.build_message_card(
                        caller=wm.caller_identity or "来电者",
                        message=new_text[:100],
                        relationship=classify_result.type_id,
                    )

            return {
                "final_action": CallAction.SUMMARY_CARD.value if is_complete else CallAction.CONTINUE_CONVERSATION.value,
                "type_id": new_type, "call_type_name": classify_result.type_id,
                "confidence": classify_result.confidence, "classify_method": classify_result.method,
                "agent_reply": reply,
                "notification_card": card.to_dict() if card else None,
            }

        # 亲友/领导/普通来电
        self.conversation_memory.add_user_message(new_text)
        call_type = ct.get_call_type(new_type)
        reply = ct.get_reply(call_type, self.presence.get_mode())
        self.conversation_memory.add_assistant_message(reply)

        turns = len(self.conversation_memory.short_term) // 2
        if turns >= 3:
            card = self.card_builder.build_message_card(
                caller=self.conversation_memory.working_memory.caller_identity or "来电者",
                message=new_text[:100],
                relationship=classify_result.type_id,
            )
            return {
                "final_action": CallAction.GENERAL_REPLY.value,
                "type_id": new_type, "call_type_name": classify_result.type_id,
                "confidence": classify_result.confidence, "classify_method": classify_result.method,
                "agent_reply": reply,
                "notification_card": card.to_dict() if card else None,
            }

        return {
            "final_action": CallAction.CONTINUE_CONVERSATION.value,
            "type_id": new_type, "call_type_name": classify_result.type_id,
            "confidence": classify_result.confidence, "classify_method": classify_result.method,
            "agent_reply": reply,
        }

    def learn_habit(self, user_input: str) -> dict:
        """机主习惯学习接口（CC）"""
        return self.habit_learner.learn_from_conversation(user_input)

    def end_activity(self, keyword: str = "") -> Optional[str]:
        """结束当前活动（CC）"""
        return self.habit_learner.end_current_activity(keyword)

    def get_habits_summary(self) -> str:
        """获取习惯摘要（CC）"""
        return self.habit_learner.get_habits_summary()

    def get_today_stats(self) -> dict:
        """获取今日通话统计（CC）"""
        return self.call_logger.get_today_stats()

    def get_caller_profile(self, phone_number: str) -> Optional[CallerProfile]:
        """查询来电画像（CC）"""
        return self.caller_profile_store.lookup(phone_number)

    def blacklist_caller(self, phone_number: str) -> None:
        """将号码加入黑名单（CC）"""
        self.caller_profile_store.set_blacklist(phone_number)

    def whitelist_caller(self, phone_number: str) -> None:
        """将号码加入白名单（CC）"""
        self.caller_profile_store.set_whitelist(phone_number)

    def get_profile_stats(self) -> dict:
        """获取画像统计（CC）"""
        return self.caller_profile_store.get_stats()

    def get_recent_recordings(self, limit: int = 10) -> list:
        """获取最近的录音列表（CC）"""
        if self.recorder:
            try:
                return self.recorder.list_recordings(limit=limit)
            except Exception:
                pass
        return []

    def get_recent_notifications(self, limit: int = 10) -> list:
        """获取最近的通知卡片（CC）"""
        return self.notification_store.load_recent(limit)

    def set_presence(self, mode: str, reason: str = "") -> None:
        """设置机主状态（QM-newer API 兼容）"""
        self.presence.set(mode, reason)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """双路混合检索（QM-newer API 兼容）"""
        if self.hybrid_retriever:
            fused = self.hybrid_retriever.retrieve(query, top_k=top_k)
            if self.reranker and fused:
                fused = self.reranker.rerank(query, fused, top_n=top_k)
            return fused[:top_k]
        return []

