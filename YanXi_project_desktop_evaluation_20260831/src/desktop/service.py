"""GUI-independent processing and a sequential, failure-isolated batch runner."""
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.desktop.audio import LocalTranscriber
from src.desktop.settings import ROOT, build_config, safe_error

ACTION_NAMES = {"reject": "拒接 / 拦截", "forward": "建议转接机主", "proxy": "AI 代接",
    "record": "记录留言", "ask": "询问信息", "general_reply": "普通回复",
    "summary_card": "生成摘要卡片", "continue_conversation": "需要继续对话", "error": "处理失败"}


@dataclass
class Job:
    source: str
    kind: str = "audio"
    text: str = ""
    caller_number: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "待处理"
    transcript: str = ""
    result: dict = field(default_factory=dict)
    error: str = ""
    elapsed: float = 0.0

    def export(self):
        return asdict(self)


def readable_result(result):
    if not result:
        return "处理完成后，这里显示来电类型、动作、回复和通知摘要。"
    card = result.get("notification_card") or {}
    confidence = result.get("confidence", 0)
    lines = [f"来电类型：{result.get('call_type_name', result.get('type_id', '未知'))}",
             f"置信度：{confidence:.0%}    判断方式：{result.get('classify_method', '未知')}",
             f"处理动作：{ACTION_NAMES.get(result.get('final_action'), result.get('final_action', '未知'))}",
             "", "AI 回复", result.get("agent_reply") or "（无口头回复）"]
    if card:
        lines += ["", "通知摘要", card.get("title", ""), card.get("body", "")]
        if card.get("data"):
            lines.append(json.dumps(card["data"], ensure_ascii=False, indent=2))
    if result.get("desktop_warning"):
        lines += ["", "提示：" + result["desktop_warning"]]
    if result.get("save_warning"):
        lines += ["", "保存提示：" + result["save_warning"]]
    lines += ["", "以上动作是系统的处理建议，不会真实接听、拒接或拨打电话。"]
    return "\n".join(lines)


def speech_text(result, full=False):
    card = result.get("notification_card") or {}
    reply = result.get("agent_reply", "").strip()
    if reply and not full:
        return reply
    parts = [f"来电类型：{result.get('call_type_name', '未知')}。",
             f"处理建议：{ACTION_NAMES.get(result.get('final_action'), '请查看结果')}。",
             reply, card.get("title", ""), card.get("body", "")]
    return "\n".join(str(part) for part in parts if part)


class DesktopService:
    def __init__(self, settings):
        self.settings = settings
        self.transcriber = LocalTranscriber(settings.stt_model, settings.stt_engine)
        self.orchestrator = None

    def test_connection(self):
        from src.core.llm_client import LLMClient
        client = LLMClient(build_config(self.settings))
        try:
            reply = client.chat([{"role": "user", "content": "只回复OK"}], max_tokens=16, temperature=0)
            if not reply or not str(reply).strip():
                raise RuntimeError(client.last_error or "未收到模型回复。请检查 API Key、服务区域、模型名称和账户额度。")
            return "连接成功，模型已返回回复。"
        finally:
            client.close()

    def process(self, job, progress, allow_download=False):
        text = job.text if job.kind == "text" else self.transcriber.transcribe(job.source, progress, allow_download)
        job.transcript = text.strip()
        if not job.transcript:
            raise ValueError("输入文字为空。")
        progress("分析来电内容")
        if self.orchestrator is None:
            from src.orchestration.orchestrator import CallOrchestrator
            self.orchestrator = CallOrchestrator(build_config(self.settings))
        self.orchestrator.llm_client.last_error = None
        result = self.orchestrator.run(job.transcript, caller_number=job.caller_number)
        if result.get("final_action") == "error":
            raise RuntimeError(result.get("final_message") or "来电处理失败")
        if getattr(self.orchestrator.llm_client, "last_error", None):
            result["desktop_warning"] = self.orchestrator.llm_client.last_error + "。本条使用规则或模板降级处理。"
        return result

    def close(self):
        if self.orchestrator is not None:
            self.orchestrator.llm_client.close()
            self.orchestrator.habit_learner.llm.close()
        self.orchestrator = None


def run_batch(jobs, service, stop, emit, allow_download=False, output_dir=None):
    """Stop means finish the current job, leave remaining jobs pending for resume."""
    try:
        for index, job in enumerate(jobs):
            if stop.is_set():
                break
            job.error, job.result, job.elapsed = "", {}, 0.0
            job.transcript = ""
            job.status = "处理中"
            emit("job", job.id)
            started = time.monotonic()
            try:
                def progress(message):
                    emit("progress", (job.id, message, index + 1, len(jobs)))
                job.result = service.process(job, progress, allow_download)
                job.status = "降级完成" if job.result.get("desktop_warning") else "完成"
            except Exception as error:
                job.status = "失败"
                job.error = safe_error(error, service.settings.api_key)
            job.elapsed = round(time.monotonic() - started, 2)
            if output_dir is not None:
                try:
                    directory = Path(output_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f"{job.id}.json"
                    temp = path.with_suffix(".tmp")
                    temp.write_text(json.dumps(job.export(), ensure_ascii=False, indent=2), encoding="utf-8")
                    temp.replace(path)
                except OSError as error:
                    job.result["save_warning"] = "自动保存失败，请手动导出。" + safe_error(error)
                    emit("notice", "结果已生成，但自动保存失败：" + safe_error(error))
            emit("job", job.id)
    finally:
        emit("batch_done", stop.is_set())
