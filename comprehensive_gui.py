"""
言犀 AI 电话代接助手——综合可视化版。

在一个 Tkinter 窗口中统一提供：
1. 麦克风录音并分析；
2. 选择单个语音文件并分析；
3. 扫描目录并执行批量自动测试。

核心 ASR、RAG、多 Agent 和 TTS 逻辑复用 call_agent.py。重型依赖采用
延迟导入，确保界面可以先显示，模型初始化和网络请求全部在后台线程执行。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


BASE_DIR = Path(__file__).resolve().parent
# call_agent.py 的知识库和向量库路径是相对路径。固定工作目录后，从快捷方式、
# PyCharm 或命令行启动都会读取同一份项目数据。
os.chdir(BASE_DIR)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
DEFAULT_BATCH_DIR = BASE_DIR / "data" / "test_audio_batch_200"
REPORT_DIR = BASE_DIR / "outputs" / "test_reports"
MIC_SAMPLE_RATE = 44100
MIC_MAX_SECONDS = 120


class PipelineService:
    """延迟初始化并复用项目原有的完整处理流水线。"""

    def __init__(self) -> None:
        self.asr = None
        self.rag = None
        self.scheduler = None
        self.tts = None
        self._init_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.asr is not None

    def initialize(self, progress: Callable[[str], None]) -> None:
        if self.ready:
            return
        with self._init_lock:
            if self.ready:
                return
            progress("正在加载 Whisper、知识库和多 Agent 系统，首次启动可能需要一些时间……")
            try:
                from call_agent import init_system
            except ImportError as exc:
                raise RuntimeError(
                    f"缺少运行依赖：{exc.name}。请使用项目原有虚拟环境，或按 README.md 安装依赖。"
                ) from exc
            self.asr, self.rag, self.scheduler, self.tts = init_system()
            progress("系统初始化完成，后续任务将复用已加载模型。")

    def analyze(
        self,
        audio_path: Path,
        play_reply: bool,
        progress: Callable[[str], None],
    ) -> dict:
        self.initialize(progress)
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"找不到语音文件：{audio_path}")

        started = time.time()
        progress(f"[1/4] 语音识别：{audio_path.name}")
        call_text = self.asr.transcribe(str(audio_path))
        if not call_text:
            raise RuntimeError("语音识别结果为空，请检查音频内容或录音设备。")

        progress("[2/4] 检索知识库规则……")
        rag_hint = self.rag.search(call_text)

        progress("[3/4] 多 Agent 场景、风险和业务分析……")
        context = self.scheduler.handle(call_text, rag_hint)

        reply_audio = ""
        if play_reply:
            progress("[4/4] 正在合成并播放 AI 回复……")
            reply_audio = asyncio.run(self.tts.speak(context.final_reply))
        else:
            progress("[4/4] 已跳过 TTS（批量测试默认只生成文字结果）。")

        scene_category = ""
        for entry in context.logs:
            if entry.step == "场景分析":
                match = re.search(r"类别=([^,，]+)", entry.content)
                if match:
                    scene_category = match.group(1).strip()
                    break

        return {
            "file": audio_path.name,
            "path": str(audio_path),
            "status": "OK",
            "call_text": call_text,
            "rag_hint": rag_hint,
            "scene_category": scene_category,
            "risk_level": context.risk_result.get("risk_level", "未知"),
            "risk_type": context.risk_result.get("risk_type", "未知"),
            "handle_suggestion": context.risk_result.get("handle_suggestion", "未知"),
            "business_type": context.business_result.get("business_type", ""),
            "final_reply": context.final_reply,
            "risk_result": context.risk_result,
            "business_result": context.business_result,
            "transfer_result": context.transfer_result,
            "logs": [{"step": item.step, "content": item.content} for item in context.logs],
            "reply_audio": reply_audio,
            "elapsed_seconds": round(time.time() - started, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


class YanxiComprehensiveGUI:
    COLORS = {
        "nav": "#17243A",
        "nav_active": "#2E69FF",
        "bg": "#F3F6FB",
        "card": "#FFFFFF",
        "text": "#172033",
        "muted": "#667085",
        "line": "#DCE3EF",
        "blue": "#2E69FF",
        "blue_hover": "#2457D6",
        "green": "#12A36D",
        "orange": "#F79009",
        "red": "#E5484D",
        "log": "#101827",
        "log_fg": "#D7E1F1",
    }

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("言犀 AI 管家 · 综合可视化版")
        self.root.geometry("1280x850")
        self.root.minsize(1050, 720)
        self.root.configure(bg=self.COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue = queue.Queue()
        self.service = PipelineService()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.is_busy = False
        self.current_mode = "microphone"
        self.last_result: Optional[dict] = None
        self.batch_results: list[dict] = []
        self.last_report_path: Optional[Path] = None

        self.mic_stream = None
        self.mic_chunks: list = []
        self.mic_started_at = 0.0
        self.mic_devices: dict[str, int] = {}
        self.last_recording: Optional[Path] = None

        self.file_path_var = tk.StringVar()
        self.file_tts_var = tk.BooleanVar(value=True)
        self.mic_device_var = tk.StringVar()
        self.mic_auto_var = tk.BooleanVar(value=True)
        self.mic_tts_var = tk.BooleanVar(value=True)
        self.batch_dir_var = tk.StringVar(value=str(DEFAULT_BATCH_DIR))
        self.batch_limit_var = tk.StringVar(value="0")
        self.batch_recursive_var = tk.BooleanVar(value=False)
        self.batch_judge_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")

        self._configure_styles()
        self._build_layout()
        self._show_mode("microphone")
        self.root.after(100, self._poll_events)
        self.root.after(250, self._refresh_microphones)

    # --------------------------- 界面构建 ---------------------------

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Yanxi.Horizontal.TProgressbar",
            troughcolor="#E7ECF4",
            background=self.COLORS["blue"],
            bordercolor="#E7ECF4",
            lightcolor=self.COLORS["blue"],
            darkcolor=self.COLORS["blue"],
            thickness=12,
        )
        style.configure(
            "Yanxi.Treeview",
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground=self.COLORS["text"],
            rowheight=30,
            borderwidth=0,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Yanxi.Treeview.Heading",
            background="#EDF2FA",
            foreground=self.COLORS["text"],
            relief="flat",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map("Yanxi.Treeview", background=[("selected", "#DDE8FF")])
        style.configure("TCombobox", padding=5)

    def _build_layout(self) -> None:
        outer = tk.Frame(self.root, bg=self.COLORS["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        self.nav = tk.Frame(outer, bg=self.COLORS["nav"], width=224)
        self.nav.pack(side=tk.LEFT, fill=tk.Y)
        self.nav.pack_propagate(False)

        tk.Label(
            self.nav,
            text="言犀 AI 管家",
            bg=self.COLORS["nav"],
            fg="white",
            font=("Microsoft YaHei UI", 18, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=24, pady=(32, 2))
        tk.Label(
            self.nav,
            text="综合可视化控制台",
            bg=self.COLORS["nav"],
            fg="#9EABC0",
            font=("Microsoft YaHei UI", 9),
            anchor="w",
        ).pack(fill=tk.X, padx=24, pady=(0, 28))

        self.nav_buttons = {}
        modes = [
            ("microphone", "麦克风收音", "实时录制并识别"),
            ("file", "语音文件", "选择单条音频"),
            ("batch", "批量自动测试", "目录扫描与质检"),
        ]
        for key, title, subtitle in modes:
            button = tk.Button(
                self.nav,
                text=f"{title}\n{subtitle}",
                command=lambda mode=key: self._show_mode(mode),
                bg=self.COLORS["nav"],
                fg="#E6ECF7",
                activebackground="#22334F",
                activeforeground="white",
                bd=0,
                padx=24,
                pady=12,
                anchor="w",
                justify=tk.LEFT,
                font=("Microsoft YaHei UI", 10),
                cursor="hand2",
            )
            button.pack(fill=tk.X, padx=10, pady=4)
            self.nav_buttons[key] = button

        tk.Label(
            self.nav,
            text="处理链路\nWhisper → RAG → 多 Agent → TTS",
            bg=self.COLORS["nav"],
            fg="#8390A6",
            justify=tk.LEFT,
            anchor="sw",
            font=("Microsoft YaHei UI", 8),
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=24)

        content = tk.Frame(outer, bg=self.COLORS["bg"])
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header = tk.Frame(content, bg=self.COLORS["bg"], height=84)
        header.pack(fill=tk.X, padx=24, pady=(18, 6))
        header.pack_propagate(False)
        self.page_title = tk.Label(
            header,
            text="",
            bg=self.COLORS["bg"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 20, "bold"),
            anchor="w",
        )
        self.page_title.pack(side=tk.LEFT, fill=tk.Y)
        self.status_label = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#E4F6EF",
            fg=self.COLORS["green"],
            padx=14,
            pady=7,
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        self.status_label.pack(side=tk.RIGHT, pady=22)

        self.page_host = tk.Frame(content, bg=self.COLORS["bg"])
        self.page_host.pack(fill=tk.BOTH, expand=True, padx=24)
        self.page_host.grid_rowconfigure(0, weight=1)
        self.page_host.grid_columnconfigure(0, weight=1)

        self.pages = {
            "microphone": self._build_microphone_page(),
            "file": self._build_file_page(),
            "batch": self._build_batch_page(),
        }

        log_card = self._card(content)
        log_card.pack(fill=tk.X, padx=24, pady=(10, 18))
        log_header = tk.Frame(log_card, bg=self.COLORS["card"])
        log_header.pack(fill=tk.X, padx=14, pady=(10, 5))
        tk.Label(
            log_header,
            text="运行日志",
            bg=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        tk.Button(
            log_header,
            text="清空",
            command=self._clear_log,
            bg=self.COLORS["card"],
            fg=self.COLORS["muted"],
            activebackground=self.COLORS["card"],
            bd=0,
            cursor="hand2",
        ).pack(side=tk.RIGHT)
        self.log_text = scrolledtext.ScrolledText(
            log_card,
            height=7,
            wrap=tk.WORD,
            bg=self.COLORS["log"],
            fg=self.COLORS["log_fg"],
            insertbackground="white",
            relief=tk.FLAT,
            font=("Consolas", 9),
            padx=10,
            pady=8,
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.log_text.tag_config("info", foreground=self.COLORS["log_fg"])
        self.log_text.tag_config("success", foreground="#55D6A7")
        self.log_text.tag_config("warning", foreground="#FFBE62")
        self.log_text.tag_config("error", foreground="#FF7B81")

    def _card(self, parent: tk.Misc) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=self.COLORS["card"],
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
            bd=0,
        )

    def _primary_button(self, parent, text, command, width=16) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=self.COLORS["blue"],
            fg="white",
            activebackground=self.COLORS["blue_hover"],
            activeforeground="white",
            disabledforeground="#D7DCE5",
            bd=0,
            pady=9,
            font=("Microsoft YaHei UI", 10, "bold"),
            cursor="hand2",
        )

    def _secondary_button(self, parent, text, command, width=12) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg="#E9EFF9",
            fg=self.COLORS["text"],
            activebackground="#DDE6F5",
            activeforeground=self.COLORS["text"],
            bd=0,
            pady=8,
            font=("Microsoft YaHei UI", 9),
            cursor="hand2",
        )

    def _build_microphone_page(self) -> tk.Frame:
        page = tk.Frame(self.page_host, bg=self.COLORS["bg"])
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=2, uniform="mic")
        page.grid_columnconfigure(1, weight=3, uniform="mic")
        page.grid_rowconfigure(0, weight=1)

        controls = self._card(page)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            controls,
            text="麦克风录音",
            bg=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w", padx=22, pady=(22, 4))
        tk.Label(
            controls,
            text="选择输入设备后开始录音，停止后可自动进入完整分析链路。",
            bg=self.COLORS["card"],
            fg=self.COLORS["muted"],
            wraplength=340,
            justify=tk.LEFT,
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=22, pady=(0, 22))

        tk.Label(
            controls, text="输入设备", bg=self.COLORS["card"], fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(anchor="w", padx=22)
        device_row = tk.Frame(controls, bg=self.COLORS["card"])
        device_row.pack(fill=tk.X, padx=22, pady=(6, 18))
        self.mic_combo = ttk.Combobox(
            device_row, textvariable=self.mic_device_var, state="readonly", font=("Microsoft YaHei UI", 9)
        )
        self.mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.refresh_mic_btn = self._secondary_button(device_row, "刷新", self._refresh_microphones, 7)
        self.refresh_mic_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.mic_timer_label = tk.Label(
            controls,
            text="00:00",
            bg=self.COLORS["card"],
            fg=self.COLORS["blue"],
            font=("Consolas", 34, "bold"),
        )
        self.mic_timer_label.pack(pady=(4, 2))
        self.mic_hint_label = tk.Label(
            controls,
            text="等待录音",
            bg=self.COLORS["card"],
            fg=self.COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        )
        self.mic_hint_label.pack(pady=(0, 18))

        button_row = tk.Frame(controls, bg=self.COLORS["card"])
        button_row.pack(pady=4)
        self.mic_start_btn = self._primary_button(button_row, "开始录音", self._start_recording, 13)
        self.mic_start_btn.pack(side=tk.LEFT, padx=4)
        self.mic_stop_btn = tk.Button(
            button_row,
            text="停止录音",
            command=self._stop_recording,
            width=13,
            bg=self.COLORS["red"],
            fg="white",
            activebackground="#C83C42",
            activeforeground="white",
            disabledforeground="#E6E8EC",
            bd=0,
            pady=9,
            font=("Microsoft YaHei UI", 10, "bold"),
            state=tk.DISABLED,
        )
        self.mic_stop_btn.pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            controls,
            text="停止后自动分析",
            variable=self.mic_auto_var,
            bg=self.COLORS["card"],
            activebackground=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=22, pady=(20, 2))
        tk.Checkbutton(
            controls,
            text="合成并播放 AI 回复",
            variable=self.mic_tts_var,
            bg=self.COLORS["card"],
            activebackground=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=22)

        result = self._card(page)
        result.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.mic_result_text = self._build_result_text(result)
        return page

    def _build_file_page(self) -> tk.Frame:
        page = tk.Frame(self.page_host, bg=self.COLORS["bg"])
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_columnconfigure(0, weight=2, uniform="file")
        page.grid_columnconfigure(1, weight=3, uniform="file")
        page.grid_rowconfigure(0, weight=1)

        controls = self._card(page)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(
            controls,
            text="选择语音文件",
            bg=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w", padx=22, pady=(22, 4))
        tk.Label(
            controls,
            text="支持 MP3、WAV、M4A、FLAC、OGG 和 AAC。",
            bg=self.COLORS["card"],
            fg=self.COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=22, pady=(0, 22))

        entry = tk.Entry(
            controls,
            textvariable=self.file_path_var,
            state="readonly",
            readonlybackground="#F8FAFD",
            fg=self.COLORS["text"],
            relief=tk.FLAT,
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
            font=("Microsoft YaHei UI", 9),
        )
        entry.pack(fill=tk.X, padx=22, ipady=9)
        self.file_browse_btn = self._secondary_button(controls, "浏览文件", self._choose_audio_file, 14)
        self.file_browse_btn.pack(anchor="w", padx=22, pady=(12, 20))
        tk.Checkbutton(
            controls,
            text="合成并播放 AI 回复",
            variable=self.file_tts_var,
            bg=self.COLORS["card"],
            activebackground=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        ).pack(anchor="w", padx=22, pady=(0, 18))
        self.file_run_btn = self._primary_button(controls, "开始分析", self._analyze_selected_file, 15)
        self.file_run_btn.pack(anchor="w", padx=22)

        result = self._card(page)
        result.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.file_result_text = self._build_result_text(result)
        return page

    def _build_result_text(self, parent: tk.Frame) -> scrolledtext.ScrolledText:
        tk.Label(
            parent,
            text="最近一次分析结果",
            bg=self.COLORS["card"],
            fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(anchor="w", padx=20, pady=(18, 8))
        text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            bg="#FBFCFE",
            fg=self.COLORS["text"],
            relief=tk.FLAT,
            font=("Microsoft YaHei UI", 10),
            padx=14,
            pady=12,
            state=tk.DISABLED,
        )
        text.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        text.tag_config("title", font=("Microsoft YaHei UI", 10, "bold"), foreground=self.COLORS["blue"])
        text.tag_config("reply", font=("Microsoft YaHei UI", 11, "bold"), foreground=self.COLORS["green"])
        text.tag_config("muted", foreground=self.COLORS["muted"])
        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "尚无分析结果。\n\n选择一种输入方式后开始处理。", "muted")
        text.configure(state=tk.DISABLED)
        return text

    def _build_batch_page(self) -> tk.Frame:
        page = tk.Frame(self.page_host, bg=self.COLORS["bg"])
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)

        controls = self._card(page)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top = tk.Frame(controls, bg=self.COLORS["card"])
        top.pack(fill=tk.X, padx=18, pady=14)
        tk.Label(
            top, text="测试目录", bg=self.COLORS["card"], fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.batch_dir_entry = tk.Entry(
            top,
            textvariable=self.batch_dir_var,
            bg="#F8FAFD",
            fg=self.COLORS["text"],
            relief=tk.FLAT,
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
            font=("Microsoft YaHei UI", 9),
        )
        self.batch_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=7)
        self.batch_browse_btn = self._secondary_button(top, "选择目录", self._choose_batch_dir, 10)
        self.batch_browse_btn.pack(side=tk.LEFT, padx=8)
        self.batch_start_btn = self._primary_button(top, "开始测试", self._start_batch, 11)
        self.batch_start_btn.pack(side=tk.LEFT, padx=(8, 4))
        self.batch_stop_btn = tk.Button(
            top,
            text="停止",
            command=self._stop_batch,
            width=8,
            bg=self.COLORS["red"],
            fg="white",
            activebackground="#C83C42",
            activeforeground="white",
            disabledforeground="#E6E8EC",
            bd=0,
            pady=9,
            font=("Microsoft YaHei UI", 9, "bold"),
            state=tk.DISABLED,
        )
        self.batch_stop_btn.pack(side=tk.LEFT, padx=(4, 0))

        options = tk.Frame(controls, bg=self.COLORS["card"])
        options.pack(fill=tk.X, padx=18, pady=(0, 14))
        tk.Label(
            options, text="数量上限（0=全部）", bg=self.COLORS["card"], fg=self.COLORS["muted"],
            font=("Microsoft YaHei UI", 9),
        ).pack(side=tk.LEFT)
        tk.Entry(
            options,
            textvariable=self.batch_limit_var,
            width=7,
            justify=tk.CENTER,
            relief=tk.FLAT,
            highlightbackground=self.COLORS["line"],
            highlightthickness=1,
        ).pack(side=tk.LEFT, padx=(7, 18), ipady=4)
        tk.Checkbutton(
            options, text="扫描子目录", variable=self.batch_recursive_var,
            bg=self.COLORS["card"], activebackground=self.COLORS["card"], fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        ).pack(side=tk.LEFT, padx=(0, 18))
        tk.Checkbutton(
            options, text="启用 AI 质检评分（每条会增加一次模型调用）", variable=self.batch_judge_var,
            bg=self.COLORS["card"], activebackground=self.COLORS["card"], fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 9),
        ).pack(side=tk.LEFT)

        self.batch_progress = ttk.Progressbar(
            controls, style="Yanxi.Horizontal.TProgressbar", mode="determinate"
        )
        self.batch_progress.pack(fill=tk.X, padx=18, pady=(0, 5))
        self.batch_progress_label = tk.Label(
            controls,
            text="等待开始测试",
            bg=self.COLORS["card"],
            fg=self.COLORS["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 8),
        )
        self.batch_progress_label.pack(fill=tk.X, padx=18, pady=(0, 10))

        table_card = self._card(page)
        table_card.grid(row=1, column=0, sticky="nsew")
        table_card.grid_rowconfigure(1, weight=1)
        table_card.grid_columnconfigure(0, weight=1)
        table_header = tk.Frame(table_card, bg=self.COLORS["card"])
        table_header.grid(row=0, column=0, sticky="ew", columnspan=2, padx=14, pady=10)
        tk.Label(
            table_header, text="测试结果（双击查看详情）", bg=self.COLORS["card"], fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side=tk.LEFT)
        self.batch_export_btn = self._secondary_button(table_header, "导出报告", self._export_batch_report, 10)
        self.batch_export_btn.configure(state=tk.DISABLED)
        self.batch_export_btn.pack(side=tk.RIGHT)

        columns = ("file", "text", "risk", "handle", "score", "verdict")
        self.batch_tree = ttk.Treeview(
            table_card, columns=columns, show="headings", style="Yanxi.Treeview", selectmode="browse"
        )
        headings = {
            "file": "文件名", "text": "识别内容", "risk": "风险", "handle": "处理建议",
            "score": "评分", "verdict": "质检",
        }
        widths = {"file": 200, "text": 360, "risk": 100, "handle": 100, "score": 70, "verdict": 70}
        for column in columns:
            self.batch_tree.heading(column, text=headings[column], anchor=tk.W)
            self.batch_tree.column(column, width=widths[column], minwidth=60, anchor=tk.W)
        self.batch_tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 12))
        scrollbar = ttk.Scrollbar(table_card, orient=tk.VERTICAL, command=self.batch_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", padx=(0, 12), pady=(0, 12))
        self.batch_tree.configure(yscrollcommand=scrollbar.set)
        self.batch_tree.tag_configure("PASS", background="#E4F6EF")
        self.batch_tree.tag_configure("WARN", background="#FFF3DD")
        self.batch_tree.tag_configure("FAIL", background="#FFE7E8")
        self.batch_tree.tag_configure("ERROR", background="#F1F2F4")
        self.batch_tree.bind("<Double-1>", self._show_selected_batch_detail)
        return page

    # --------------------------- 模式与状态 ---------------------------

    def _show_mode(self, mode: str) -> None:
        self.current_mode = mode
        titles = {
            "microphone": "麦克风收音",
            "file": "语音文件分析",
            "batch": "批量自动测试",
        }
        self.page_title.configure(text=titles[mode])
        self.pages[mode].tkraise()
        for key, button in self.nav_buttons.items():
            button.configure(bg=self.COLORS["nav_active"] if key == mode else self.COLORS["nav"])

    def _set_status(self, text: str, kind: str = "ready") -> None:
        self.status_var.set(text)
        palette = {
            "ready": ("#E4F6EF", self.COLORS["green"]),
            "busy": ("#E6EEFF", self.COLORS["blue"]),
            "warning": ("#FFF3DD", self.COLORS["orange"]),
            "error": ("#FFE7E8", self.COLORS["red"]),
        }
        bg, fg = palette.get(kind, palette["ready"])
        self.status_label.configure(bg=bg, fg=fg)

    def _set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        normal = tk.NORMAL if not busy else tk.DISABLED
        for widget in (
            self.file_browse_btn,
            self.file_run_btn,
            self.batch_browse_btn,
            self.batch_start_btn,
            self.refresh_mic_btn,
        ):
            widget.configure(state=normal)
        if self.mic_stream is None:
            self.mic_start_btn.configure(state=normal)
        self.batch_stop_btn.configure(state=tk.NORMAL if busy and self.current_mode == "batch" else tk.DISABLED)
        self._set_status("处理中…" if busy else "就绪", "busy" if busy else "ready")

    def _post(self, event_type: str, payload=None) -> None:
        self.events.put((event_type, payload))

    def _log(self, message: str, tag: str = "info") -> None:
        self.log_text.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # --------------------------- 单文件分析 ---------------------------

    def _choose_audio_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择来电语音文件",
            initialdir=str(DEFAULT_BATCH_DIR if DEFAULT_BATCH_DIR.exists() else BASE_DIR),
            filetypes=[
                ("语音文件", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.file_path_var.set(path)
            self._log(f"已选择语音文件：{path}")

    def _analyze_selected_file(self) -> None:
        path = Path(self.file_path_var.get()) if self.file_path_var.get() else None
        if not path or not path.is_file():
            messagebox.showwarning("请选择文件", "请先选择一个有效的语音文件。")
            return
        self._start_single_analysis(path, self.file_tts_var.get(), "file")

    def _start_single_analysis(self, path: Path, play_reply: bool, source: str) -> None:
        if self.is_busy:
            messagebox.showinfo("任务进行中", "请等待当前任务结束后再开始新的分析。")
            return
        self._set_busy(True)
        self._log(f"开始分析：{path.name}", "success")

        def run() -> None:
            try:
                result = self.service.analyze(
                    path,
                    play_reply=play_reply,
                    progress=lambda message: self._post("log", (message, "info")),
                )
                self._post("analysis_result", (source, result))
            except Exception as exc:
                self._post("error", f"分析失败：{exc}")
                self._post("log", (traceback.format_exc(), "error"))
            finally:
                self._post("job_done")

        self.worker = threading.Thread(target=run, daemon=True, name="yanxi-single-analysis")
        self.worker.start()

    def _render_result(self, text_widget: scrolledtext.ScrolledText, result: dict) -> None:
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)

        def section(title: str, content: str, tag: Optional[str] = None) -> None:
            text_widget.insert(tk.END, title + "\n", "title")
            text_widget.insert(tk.END, (content or "无") + "\n\n", tag or "")

        section("识别到的来电内容", result.get("call_text", ""))
        section("场景与风险", (
            f"场景：{result.get('scene_category') or '未知'}\n"
            f"风险等级：{result.get('risk_level', '未知')}\n"
            f"风险类型：{result.get('risk_type', '未知')}\n"
            f"处理建议：{result.get('handle_suggestion', '未知')}"
        ))
        section("AI 最终回复", result.get("final_reply", ""), "reply")
        if result.get("business_type"):
            section("业务类型", result["business_type"])
        log_lines = "\n".join(
            f"[{item['step']}] {item['content']}" for item in result.get("logs", [])
        )
        section("执行轨迹", log_lines)
        text_widget.insert(
            tk.END,
            f"总耗时：{result.get('elapsed_seconds', 0):.2f} 秒\n文件：{result.get('path', '')}",
            "muted",
        )
        text_widget.configure(state=tk.DISABLED)

    # --------------------------- 麦克风录音 ---------------------------

    def _refresh_microphones(self) -> None:
        if self.mic_stream is not None:
            return
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            found = {}
            for index, device in enumerate(devices):
                if int(device.get("max_input_channels", 0)) > 0:
                    label = f"[{index}] {device['name']}"
                    found[label] = index
            self.mic_devices = found
            values = list(found.keys())
            self.mic_combo.configure(values=values)
            if values:
                default_input = sd.default.device[0] if sd.default.device else None
                preferred = next((label for label, idx in found.items() if idx == default_input), values[0])
                self.mic_device_var.set(preferred)
                self._log(f"检测到 {len(values)} 个麦克风输入设备。", "success")
            else:
                self.mic_device_var.set("")
                self._log("未检测到可用麦克风。", "warning")
        except ImportError:
            self._log("缺少 sounddevice，麦克风模式不可用；请安装 sounddevice 和 soundfile。", "error")
            self.mic_device_var.set("缺少 sounddevice")
        except Exception as exc:
            self._log(f"读取麦克风设备失败：{exc}", "error")

    def _start_recording(self) -> None:
        if self.is_busy:
            messagebox.showinfo("任务进行中", "请等待当前分析任务结束。")
            return
        label = self.mic_device_var.get()
        if label not in self.mic_devices:
            messagebox.showwarning("麦克风不可用", "请刷新并选择一个有效的麦克风输入设备。")
            return
        try:
            import sounddevice as sd

            self.mic_chunks = []

            def callback(indata, frames, timing, status) -> None:
                del frames, timing
                if status:
                    self._post("log", (f"录音设备提示：{status}", "warning"))
                self.mic_chunks.append(indata.copy())

            self.mic_stream = sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.mic_devices[label],
                callback=callback,
            )
            self.mic_stream.start()
            self.mic_started_at = time.time()
            self.mic_start_btn.configure(state=tk.DISABLED)
            self.mic_stop_btn.configure(state=tk.NORMAL)
            self.mic_hint_label.configure(text="正在录音，请对着麦克风说话", fg=self.COLORS["red"])
            self._set_status("录音中…", "warning")
            self._log(f"开始录音，设备：{label}", "success")
            self._tick_recording()
        except ImportError as exc:
            messagebox.showerror("缺少依赖", f"麦克风录音缺少依赖：{exc.name}")
        except Exception as exc:
            self.mic_stream = None
            messagebox.showerror("无法录音", str(exc))
            self._log(f"开始录音失败：{exc}", "error")

    def _tick_recording(self) -> None:
        if self.mic_stream is None:
            return
        elapsed = time.time() - self.mic_started_at
        minutes, seconds = divmod(int(elapsed), 60)
        self.mic_timer_label.configure(text=f"{minutes:02d}:{seconds:02d}")
        if elapsed >= MIC_MAX_SECONDS:
            self._log(f"已达到最长录音时间 {MIC_MAX_SECONDS} 秒，自动停止。", "warning")
            self._stop_recording()
            return
        self.root.after(200, self._tick_recording)

    def _stop_recording(self) -> None:
        stream = self.mic_stream
        if stream is None:
            return
        self.mic_stream = None
        try:
            stream.stop()
            stream.close()
            if not self.mic_chunks:
                raise RuntimeError("没有录到音频数据。")
            import numpy as np
            import soundfile as sf

            audio = np.concatenate(self.mic_chunks, axis=0)
            duration = len(audio) / MIC_SAMPLE_RATE
            if duration < 0.5:
                raise RuntimeError(f"录音太短（{duration:.1f} 秒），请至少录制 0.5 秒。")
            target_dir = Path(tempfile.gettempdir()) / "yanxi_comprehensive_gui"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"mic_{datetime.now():%Y%m%d_%H%M%S}.wav"
            sf.write(str(target), audio, MIC_SAMPLE_RATE)
            self.last_recording = target
            self.mic_hint_label.configure(text=f"录音完成：{duration:.1f} 秒", fg=self.COLORS["green"])
            self._log(f"录音已保存：{target}（{duration:.1f} 秒）", "success")
            if self.mic_auto_var.get():
                self._start_single_analysis(target, self.mic_tts_var.get(), "microphone")
            else:
                self._set_status("录音已保存", "ready")
        except ImportError as exc:
            messagebox.showerror("缺少依赖", f"保存录音缺少依赖：{exc.name}")
            self._log(f"保存录音失败，缺少 {exc.name}", "error")
        except Exception as exc:
            messagebox.showerror("录音失败", str(exc))
            self._log(f"停止或保存录音失败：{exc}", "error")
        finally:
            self.mic_chunks = []
            self.mic_start_btn.configure(state=tk.NORMAL if not self.is_busy else tk.DISABLED)
            self.mic_stop_btn.configure(state=tk.DISABLED)

    # --------------------------- 批量测试 ---------------------------

    def _choose_batch_dir(self) -> None:
        path = filedialog.askdirectory(
            title="选择批量测试语音目录",
            initialdir=self.batch_dir_var.get() or str(BASE_DIR),
        )
        if path:
            self.batch_dir_var.set(path)
            self._log(f"批量测试目录：{path}")

    def _scan_batch_files(self) -> list[Path]:
        folder = Path(self.batch_dir_var.get().strip())
        if not folder.is_dir():
            raise ValueError("请选择一个有效的批量测试目录。")
        iterator = folder.rglob("*") if self.batch_recursive_var.get() else folder.glob("*")
        files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
        try:
            limit = int(self.batch_limit_var.get().strip() or "0")
        except ValueError as exc:
            raise ValueError("数量上限必须是整数，0 表示全部。") from exc
        if limit < 0:
            raise ValueError("数量上限不能小于 0。")
        return files[:limit] if limit else files

    def _start_batch(self) -> None:
        if self.is_busy:
            return
        try:
            files = self._scan_batch_files()
        except ValueError as exc:
            messagebox.showwarning("批量测试", str(exc))
            return
        if not files:
            messagebox.showwarning("批量测试", "所选目录中没有支持的语音文件。")
            return

        self.batch_results = []
        self.last_report_path = None
        self.stop_event.clear()
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        self.batch_progress.configure(maximum=len(files), value=0)
        self.batch_progress_label.configure(text=f"准备处理 {len(files)} 个文件")
        self.batch_export_btn.configure(state=tk.DISABLED)
        self._set_busy(True)
        self.batch_stop_btn.configure(state=tk.NORMAL)
        judge_enabled = self.batch_judge_var.get()
        self._log(
            f"开始批量测试：{len(files)} 个文件；AI 质检={'开启' if judge_enabled else '关闭'}。",
            "success",
        )

        def run() -> None:
            judge = None
            try:
                self.service.initialize(lambda msg: self._post("log", (msg, "info")))
                if judge_enabled:
                    from auto_test_gui import JudgeAgent
                    judge = JudgeAgent()

                for index, audio_path in enumerate(files, start=1):
                    if self.stop_event.is_set():
                        break
                    self._post("batch_progress", (index - 1, len(files), f"正在处理：{audio_path.name}"))
                    try:
                        result = self.service.analyze(
                            audio_path,
                            play_reply=False,
                            progress=lambda msg: self._post("log", (msg, "info")),
                        )
                        if judge is not None and not self.stop_event.is_set():
                            self._post("log", (f"AI 质检评分：{audio_path.name}", "info"))
                            result["evaluation"] = judge.evaluate(
                                call_text=result["call_text"],
                                risk_level=result["risk_level"],
                                risk_type=result["risk_type"],
                                handle_suggestion=result["handle_suggestion"],
                                final_reply=result["final_reply"],
                                scene_category=result["scene_category"],
                                business_type=result["business_type"],
                            )
                    except Exception as exc:
                        result = {
                            "file": audio_path.name,
                            "path": str(audio_path),
                            "status": "ERROR",
                            "error": str(exc),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                        }
                        self._post("log", (f"{audio_path.name} 处理失败：{exc}", "error"))
                    self._post("batch_result", result)
                    self._post("batch_progress", (index, len(files), f"已完成 {index}/{len(files)}"))

                stopped = self.stop_event.is_set()
                self._post("batch_finished", (stopped, len(files)))
            except Exception as exc:
                self._post("error", f"批量测试初始化失败：{exc}")
                self._post("log", (traceback.format_exc(), "error"))
                self._post("batch_finished", (False, len(files)))

        self.worker = threading.Thread(target=run, daemon=True, name="yanxi-batch-test")
        self.worker.start()

    def _stop_batch(self) -> None:
        if self.is_busy:
            self.stop_event.set()
            self.batch_stop_btn.configure(state=tk.DISABLED)
            self.batch_progress_label.configure(text="正在停止；当前文件处理完后结束……")
            self._log("收到停止请求，将在当前文件处理结束后停止。", "warning")

    def _add_batch_result(self, result: dict) -> None:
        self.batch_results.append(result)
        if result.get("status") != "OK":
            values = (result.get("file", "?"), f"错误：{result.get('error', '未知')}", "-", "-", "-", "ERROR")
            tag = "ERROR"
        else:
            evaluation = result.get("evaluation", {})
            score = evaluation.get("total_score", "-")
            if isinstance(score, (int, float)):
                score = f"{score:.1f}"
            verdict = evaluation.get("overall_verdict", "未评分")
            text = result.get("call_text", "")
            values = (
                result.get("file", ""),
                text[:80] + ("…" if len(text) > 80 else ""),
                result.get("risk_level", ""),
                result.get("handle_suggestion", ""),
                score,
                verdict,
            )
            tag = verdict if verdict in {"PASS", "WARN", "FAIL"} else ""
        self.batch_tree.insert("", tk.END, values=values, tags=(tag,) if tag else ())

    def _batch_summary(self) -> dict:
        ok = [item for item in self.batch_results if item.get("status") == "OK"]
        errors = [item for item in self.batch_results if item.get("status") != "OK"]
        evaluations = [item.get("evaluation", {}) for item in ok if item.get("evaluation")]
        scores = [item.get("total_score", 0) for item in evaluations]
        verdicts = {name: sum(e.get("overall_verdict") == name for e in evaluations) for name in ("PASS", "WARN", "FAIL")}
        return {
            "total": len(self.batch_results),
            "succeeded": len(ok),
            "errors": len(errors),
            "passed": verdicts["PASS"],
            "warned": verdicts["WARN"],
            "failed": verdicts["FAIL"],
            "average_score": round(sum(scores) / len(scores), 1) if scores else None,
        }

    def _write_batch_report(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source_directory": self.batch_dir_var.get(),
                "recursive": self.batch_recursive_var.get(),
                "ai_judge_enabled": self.batch_judge_var.get(),
            },
            "summary": self._batch_summary(),
            "results": self.batch_results,
        }
        with target.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    def _export_batch_report(self) -> None:
        if not self.batch_results:
            return
        default_name = f"comprehensive_test_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        path = filedialog.asksaveasfilename(
            title="导出批量测试报告",
            initialdir=str(REPORT_DIR),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json")],
        )
        if not path:
            return
        try:
            self._write_batch_report(Path(path))
            self.last_report_path = Path(path)
            messagebox.showinfo("导出成功", f"报告已保存至：\n{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _show_selected_batch_detail(self, _event=None) -> None:
        selection = self.batch_tree.selection()
        if not selection:
            return
        index = self.batch_tree.index(selection[0])
        if index >= len(self.batch_results):
            return
        result = self.batch_results[index]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"测试详情 · {result.get('file', '?')}")
        dialog.geometry("800x680")
        dialog.configure(bg=self.COLORS["bg"])
        text = scrolledtext.ScrolledText(
            dialog, wrap=tk.WORD, bg="white", fg=self.COLORS["text"],
            font=("Microsoft YaHei UI", 10), padx=14, pady=14,
        )
        text.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        text.insert(tk.END, json.dumps(result, ensure_ascii=False, indent=2))
        text.configure(state=tk.DISABLED)

    # --------------------------- 队列与生命周期 ---------------------------

    def _poll_events(self) -> None:
        try:
            while True:
                event_type, payload = self.events.get_nowait()
                if event_type == "log":
                    message, tag = payload
                    self._log(message, tag)
                elif event_type == "analysis_result":
                    source, result = payload
                    self.last_result = result
                    target = self.mic_result_text if source == "microphone" else self.file_result_text
                    self._render_result(target, result)
                    self._log(
                        f"分析完成：{result['file']}，风险={result['risk_level']}，处理={result['handle_suggestion']}",
                        "success",
                    )
                elif event_type == "error":
                    self._set_status("任务失败", "error")
                    self._log(payload, "error")
                    messagebox.showerror("运行失败", payload)
                elif event_type == "job_done":
                    self._set_busy(False)
                elif event_type == "batch_progress":
                    current, total, message = payload
                    self.batch_progress.configure(value=current, maximum=total)
                    self.batch_progress_label.configure(text=message)
                elif event_type == "batch_result":
                    self._add_batch_result(payload)
                elif event_type == "batch_finished":
                    stopped, expected = payload
                    self._set_busy(False)
                    self.batch_stop_btn.configure(state=tk.DISABLED)
                    summary = self._batch_summary()
                    status = "已停止" if stopped else "测试完成"
                    self.batch_progress_label.configure(
                        text=(
                            f"{status}：处理 {summary['total']}/{expected}，成功 {summary['succeeded']}，"
                            f"错误 {summary['errors']}，通过 {summary['passed']}，警告 {summary['warned']}，失败 {summary['failed']}"
                        )
                    )
                    if self.batch_results:
                        self.batch_export_btn.configure(state=tk.NORMAL)
                        try:
                            auto_path = REPORT_DIR / f"comprehensive_test_report_{datetime.now():%Y%m%d_%H%M%S}.json"
                            self._write_batch_report(auto_path)
                            self.last_report_path = auto_path
                            self._log(f"批量报告已自动保存：{auto_path}", "success")
                        except Exception as exc:
                            self._log(f"自动保存报告失败：{exc}", "error")
                    self._log(self.batch_progress_label.cget("text"), "warning" if stopped else "success")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _on_close(self) -> None:
        if self.mic_stream is not None:
            if not messagebox.askyesno("退出", "正在录音，确定停止录音并退出吗？"):
                return
            try:
                self.mic_stream.abort()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
        elif self.is_busy:
            if not messagebox.askyesno("退出", "任务仍在运行，确定请求停止并退出界面吗？"):
                return
        self.stop_event.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app = YanxiComprehensiveGUI()
    app.run()


if __name__ == "__main__":
    main()
