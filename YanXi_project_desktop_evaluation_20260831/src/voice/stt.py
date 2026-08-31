"""
语音识别模块 (Speech-to-Text, STT)
---------------------------------
使用 Faster-Whisper 模型将麦克风输入的语音实时转换为文字。

技术要点：
- Faster-Whisper 基于 CTranslate2，比原版 Whisper 快 3~4 倍，显存占用更低
- VAD（语音活动检测）：通过音量阈值判断说话开始/结束，自动分段
- 支持 GPU (CUDA) 和 CPU 推理

使用方式:
    from src.voice.stt import SpeechRecognizer

    recognizer = SpeechRecognizer(config)
    recognizer.load_model()  # 加载模型（首次会下载，约 1.5GB）

    text = recognizer.listen()  # 开始监听，返回识别文本
"""

import collections
import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SpeechRecognizer:
    """
    语音识别器，封装 Faster-Whisper 和麦克风录音逻辑。

    工作流程:
    1. 打开麦克风，持续监听音量
    2. 音量超过阈值 → 开始录制
    3. 音量持续低于阈值 N 秒 → 结束录制
    4. 将录制的音频送入 Whisper 识别
    5. 返回识别出的文字
    """

    def __init__(self, config: dict):
        """
        初始化语音识别器。

        参数:
            config: 全局配置字典，主要读取 stt 部分
        """
        self.config = config
        stt_cfg = config.get("stt", {})

        self.model_size = stt_cfg.get("model_size", "medium")
        self.device = stt_cfg.get("device", "cuda")
        self.compute_type = stt_cfg.get("compute_type", "float16")
        self.language = stt_cfg.get("language", "zh")
        self.sample_rate = stt_cfg.get("sample_rate", 16000)
        self.vad_threshold = stt_cfg.get("vad_threshold", 0.1)
        self.silence_duration = stt_cfg.get("silence_duration", 2.0)
        self.device_index = stt_cfg.get("device_index", None)  # None=系统默认

        # 音频参数
        self.chunk_size = 1024  # 每次读取的音频帧数
        self.format = pyaudio.paInt16  # 16 位采样
        self.channels = 1  # 单声道

        # 模型和音频流对象（懒加载）
        self._model = None
        self._audio = None
        self._stream = None

        # 存储最新识别结果
        self.last_text: str = ""

    def load_model(self):
        """
        加载 Faster-Whisper 模型。
        首次运行会自动从 HuggingFace 下载模型文件（约 1.5GB，medium 模型），
        下载后会缓存在本地，之后不再需要下载。

        Faster-Whisper 模型大小参考:
        - tiny:    ~75MB, 最快但精度最低
        - small:   ~500MB
        - medium:  ~1.5GB, 推荐平衡精度和速度
        - large-v3: ~3GB, 精度最高但最慢
        """
        if self._model is not None:
            return  # 已加载，跳过

        logger.info(f"正在加载 Faster-Whisper 模型: {self.model_size}...")
        logger.info(f"设备: {self.device}, 计算精度: {self.compute_type}")

        # 自动设置 HuggingFace 镜像
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Faster-Whisper 模型加载完成 ✓")
        except ImportError:
            raise ImportError(
                "请安装 faster-whisper: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _get_audio_stream(self):
        """
        打开 PyAudio 音频输入流。
        """
        if self._audio is None:
            self._audio = pyaudio.PyAudio()

        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()

        # 如果指定了设备索引，打印设备信息
        if self.device_index is not None:
            info = self._audio.get_device_info_by_index(self.device_index)
            logger.info(f"使用指定麦克风: [{self.device_index}] {info['name']}")

        self._stream = self._audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        logger.debug(f"音频流已打开: {self.sample_rate}Hz, {self.channels}声道")

    def _is_speech(self, audio_chunk: bytes) -> float:
        """
        判断当前音频帧是否为语音（基于音量）。
        计算音频数据的 RMS（均方根）能量作为音量指标。

        参数:
            audio_chunk: 1024 帧的原始音频字节数据

        返回:
            float: 归一化音量值 (0~1)
        """
        # 将字节数据转为 numpy int16 数组
        data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        # 计算 RMS 能量，归一化到 0~1 范围
        rms = np.sqrt(np.mean(data ** 2)) / 32768.0
        return float(rms)

    def listen_with_keyboard(self, timeout: float = 60.0) -> str:
        """
        空格键控制录音（比 VAD 更可靠）。

        按空格键开始录音，再按空格键停止。
        使用 msvcrt (Windows) 或 termios (Unix) 实现非阻塞按键检测。

        参数:
            timeout: 最长录音时间（秒）

        返回:
            str: 识别出的中文文本
        """
        self.load_model()
        self._get_audio_stream()

        import sys

        # 跨平台按键检测
        if sys.platform == "win32":
            import msvcrt

            def wait_for_space():
                """等待空格键按下"""
                print("\r  按 [空格键] 开始录音...", end="", flush=True)
                while True:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch == b' ':
                            break
        else:
            # Unix: 用 termios（或提示用户按 Enter）
            def wait_for_space():
                input("\n  按 Enter 开始录音...")

        frames: list[bytes] = []
        chunk_size = self.chunk_size

        try:
            # 等待开始
            wait_for_space()
            print(f"\r  🔴 录音中... 再按 [空格键] 停止{' ' * 30}")

            # 开始录音
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    print(f"\r  录音超时 ({timeout}秒)，自动停止{' ' * 20}")
                    break

                chunk = self._stream.read(chunk_size, exception_on_overflow=False)
                frames.append(chunk)

                # 检查空格键停止
                if sys.platform == "win32":
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch == b' ':
                            print(f"\r  停止录音{' ' * 40}")
                            break

        finally:
            pass

        if not frames:
            logger.warning("未录制到任何音频数据")
            return ""

        recording_duration = len(frames) * chunk_size / self.sample_rate
        logger.info(f"录音完成，时长: {recording_duration:.1f} 秒")

        # 保存 WAV
        wav_path = Path(__file__).resolve().parent.parent.parent / "data" / "temp_recording.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_wav(str(wav_path), frames)

        # 转录
        text = self._transcribe(str(wav_path))
        wav_path.unlink(missing_ok=True)

        self.last_text = text
        logger.info(f"识别结果: {text}")
        return text

    def listen(self, timeout: float = 30.0) -> str:
        """
        监听麦克风，等待用户说话，识别后返回文字。

        工作流程:
        1. 持续监听，当音量超过 VAD 阈值时开始录音
        2. 当连续静音超过 silence_duration 秒时停止录音
        3. 将录音的原始 PCM 数据保存为临时 WAV 文件
        4. 调用 Whisper 模型进行识别
        5. 返回识别出的文字

        参数:
            timeout: 最长等待时间（秒），超时返回空字符串

        返回:
            str: 识别出的中文文本，超时或无语音返回空字符串
        """
        self.load_model()
        self._get_audio_stream()

        logger.info(f"🎤 开始监听... (最长等待 {timeout} 秒, VAD阈值={self.vad_threshold})")

        frames: list[bytes] = []
        is_recording = False
        silence_frames = 0
        # 计算静音需要的帧数: 每秒 sample_rate/chunk_size 帧
        silence_threshold_frames = int(
            self.silence_duration * self.sample_rate / self.chunk_size
        )
        start_time = time.time()
        frame_count = 0

        try:
            while True:
                # 检查超时
                if time.time() - start_time > timeout:
                    if not is_recording:
                        logger.info("监听超时，未检测到语音")
                        return ""
                    else:
                        logger.debug("录音超时，开始识别")
                        break

                # 读取一帧音频数据
                chunk = self._stream.read(self.chunk_size, exception_on_overflow=False)
                volume = self._is_speech(chunk)
                frame_count += 1

                # 实时音量显示（每20帧约1.3秒刷新一次）
                if frame_count % 20 == 0:
                    bar_len = min(int(volume * 100), 60)
                    bar = "█" * bar_len + "░" * (60 - bar_len)
                    status = "🔴录音中" if is_recording else "🟢等待语音"
                    print(f"\r  {status} 音量: [{bar}] {volume:.4f}  (阈值={self.vad_threshold})  ", end="", flush=True)

                if not is_recording:
                    # --- 等待语音开始 ---
                    if volume > self.vad_threshold:
                        is_recording = True
                        frames = [chunk]  # 开始收集帧
                        silence_frames = 0
                        print(f"\r  检测到语音 (音量={volume:.3f})，开始录音...                              ")
                        logger.debug(f"检测到语音 (音量={volume:.3f})，开始录音...")
                else:
                    # --- 正在录音，检测结束条件 ---
                    frames.append(chunk)
                    if volume < self.vad_threshold:
                        silence_frames += 1
                        if silence_frames >= silence_threshold_frames:
                            logger.debug(
                                f"连续静音 {self.silence_duration} 秒，结束录音"
                            )
                            break
                    else:
                        silence_frames = 0  # 有声音，重置静音计数
        finally:
            pass  # 不在这里关闭流，留给后续调用

        if not frames:
            logger.warning("未收集到任何音频数据")
            return ""

        # --- 将录制的 PCM 数据保存为 WAV 文件 ---
        recording_duration = len(frames) * self.chunk_size / self.sample_rate
        logger.info(f"录音完成，时长: {recording_duration:.1f} 秒")

        wav_path = Path(__file__).resolve().parent.parent.parent / "data" / "temp_recording.wav"
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        self._save_wav(str(wav_path), frames)

        # --- 使用 Whisper 识别 ---
        text = self._transcribe(str(wav_path))

        # 清理临时文件
        wav_path.unlink(missing_ok=True)

        self.last_text = text
        logger.info(f"识别结果: {text}")
        return text

    def _save_wav(self, filepath: str, frames: list[bytes]):
        """
        将原始 PCM 音频帧保存为 WAV 格式文件。

        参数:
            filepath: 输出 WAV 文件路径
            frames: 音频帧列表（每个帧是 bytes）
        """
        wf = wave.open(filepath, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self._audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        logger.debug(f"音频已保存: {filepath}")

    def _transcribe(self, audio_path: str) -> str:
        """
        调用 Faster-Whisper 模型将音频文件转录为文字。

        参数:
            audio_path: WAV 音频文件路径

        返回:
            str: 识别出的文本，所有片段用空格拼接
        """
        logger.debug(f"开始转录: {audio_path}")

        # beam_size=5 指定束搜索宽度，值越大精度越高但越慢
        # vad_filter=True 启用内置 VAD 过滤，进一步去除静音段
        segments, info = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )

        logger.debug(
            f"检测到语言: {info.language} (概率={info.language_probability:.2%})"
        )

        # 将所有识别的文本片段拼接
        text_parts = []
        for segment in segments:
            logger.debug(f"  [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
            text_parts.append(segment.text.strip())

        full_text = "".join(text_parts)
        return full_text

    def transcribe_file(self, audio_path: str) -> str:
        """
        直接对已有音频文件进行转录（不通过麦克风录音）。
        用于测试或处理已录制的音频文件。

        参数:
            audio_path: 音频文件路径（支持 WAV, MP3 等格式）

        返回:
            str: 识别出的文本
        """
        self.load_model()
        return self._transcribe(audio_path)

    def close(self):
        """
        释放音频资源，关闭麦克风流。
        程序退出时必须调用，否则可能残留音频进程。
        """
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._audio is not None:
            self._audio.terminate()
            self._audio = None
        logger.debug("音频资源已释放")


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    """
    测试语音识别模块:
        python -m src.voice.stt                 正常识别测试
        python -m src.voice.stt --list          列出麦克风设备
        python -m src.voice.stt --test          实时音量测试（验证哪个设备有声音）
        python -m src.voice.stt --device 1    指定设备号识别
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    # --- 列出设备 ---
    if "--list" in sys.argv:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("可用的输入设备:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate()
        sys.exit(0)

    # --- 实时音量测试 ---
    if "--test" in sys.argv:
        import pyaudio, time
        device_idx = None
        if "--device" in sys.argv:
            idx = sys.argv.index("--device")
            device_idx = int(sys.argv[idx + 1])

        pa = pyaudio.PyAudio()
        if device_idx is not None:
            info = pa.get_device_info_by_index(device_idx)
            print(f"测试设备 [{device_idx}]: {info['name']}")
        else:
            print("测试默认输入设备")

        print("请说话... (Ctrl+C 停止)")
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                         input=True, input_device_index=device_idx,
                         frames_per_buffer=1024)
        try:
            while True:
                data = stream.read(1024, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = np.sqrt(np.mean(arr ** 2)) / 32768.0
                bar = "#" * int(rms * 80) + "-" * (80 - int(rms * 80))
                print(f"\r  音量: [{bar}] {rms:.4f}", end="", flush=True)
        except KeyboardInterrupt:
            print("\n\n测试结束")
        finally:
            stream.close()
            pa.terminate()
        sys.exit(0)

    # --- 正常识别测试 ---
    device_index = None
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        device_index = int(sys.argv[idx + 1])

    test_config = {
        "stt": {
            "model_size": "medium",
            "device": "cuda",
            "compute_type": "float16",
            "language": "zh",
            "sample_rate": 16000,
            "vad_threshold": 0.1,
            "silence_duration": 2.0,
            "device_index": device_index,
        },
        "logging": {"level": "INFO"},
    }

    recognizer = SpeechRecognizer(test_config)
    try:
        print("=" * 50)
        print("语音识别测试 - 请对麦克风说话 (Ctrl+C 退出)")
        print("=" * 50)
        text = recognizer.listen(timeout=60)
        if text:
            print(f"\n识别结果: {text}")
        else:
            print("\n未检测到语音")
    except KeyboardInterrupt:
        print("\n中断测试")
    finally:
        recognizer.close()
