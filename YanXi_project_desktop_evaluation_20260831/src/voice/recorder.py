"""
通话录音模块 (CallRecorder)
============================
管理来电录音的启动、停止、信息关联和查询。

使用方式:
    from src.voice.recorder import CallRecorder

    recorder = CallRecorder(persist_dir="./data/recordings")
    recorder.start("call_abc", caller_number="13800138000")
    # ... 处理来电 ...
    recorder.stop()
    recorder.update_recording_info("call_abc", call_type="food_delivery", final_action="proxy")
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RecordingInfo:
    """录音记录"""
    call_id: str = ""
    caller_number: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    filepath: str = ""
    call_type: str = ""
    final_action: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = f"rec_{uuid.uuid4().hex[:8]}"
        if not self.start_time:
            self.start_time = time.time()

    @property
    def duration_sec(self) -> float:
        if self.end_time > 0:
            return round(self.end_time - self.start_time, 1)
        return 0.0


class CallRecorder:
    """通话录音管理器（当前为元数据记录模式，实际音频采集由 STT 模块处理）。"""

    def __init__(self, persist_dir: str = "./data/recordings"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, RecordingInfo] = {}
        self._index_path = self.persist_dir / "recording_index.jsonl"

    def start(self, call_id: str, caller_number: str = "") -> None:
        """开始录音（记录元数据）。"""
        info = RecordingInfo(
            call_id=call_id,
            caller_number=caller_number,
        )
        self._active[call_id] = info
        logger.debug(f"录音开始: {call_id}")

    def stop(self) -> None:
        """停止当前录音。"""
        now = time.time()
        for info in self._active.values():
            if info.end_time == 0:
                info.end_time = now
                self._save(info)
        self._active.clear()

    def update_recording_info(
        self,
        call_id: str,
        call_type: str = "",
        final_action: str = "",
    ) -> None:
        """更新录音关联的分类和动作信息。"""
        info = self._active.get(call_id)
        if info:
            if call_type:
                info.call_type = call_type
            if final_action:
                info.final_action = final_action

    def list_recordings(self, limit: int = 10) -> list[RecordingInfo]:
        """获取最近的录音记录。"""
        results = []
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-limit * 2:]):
                try:
                    data = json.loads(line.strip())
                    results.append(RecordingInfo(**data))
                except Exception:
                    continue
        return results[:limit]

    def _save(self, info: RecordingInfo) -> None:
        """保存录音记录到索引文件。"""
        with open(self._index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(info), ensure_ascii=False) + "\n")
