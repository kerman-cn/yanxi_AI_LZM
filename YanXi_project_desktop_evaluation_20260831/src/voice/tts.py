"""
语音合成模块 (Text-to-Speech, TTS)
---------------------------------
使用 Edge-TTS 将文字转换为语音并通过扬声器播放。

技术要点：
- Edge-TTS 调用微软 Edge 浏览器的免费 TTS 引擎，无需 API Key
- 支持多种中文语音角色，默认使用 zh-CN-XiaoxiaoNeural（自然女声）
- 先合成 MP3 音频，再用 pygame 播放，播放后自动清理临时文件

使用方式:
    from src.voice.tts import SpeechSynthesizer

    tts = SpeechSynthesizer(config)
    await tts.speak("您好，请问是张先生的外卖吗？")
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

import pygame

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SpeechSynthesizer:
    """
    语音合成器，封装 Edge-TTS 合成逻辑和音频播放。

    支持的语音角色（中文）:
    - zh-CN-XiaoxiaoNeural: 女声，自然活泼（推荐）
    - zh-CN-YunxiNeural:   男声，温暖
    - zh-CN-XiaoyiNeural:  女声，温柔
    - zh-CN-YunjianNeural: 男声，沉稳
    """

    # 可用的中文语音角色列表
    AVAILABLE_VOICES = [
        "zh-CN-XiaoxiaoNeural",  # 女声（默认）
        "zh-CN-YunxiNeural",     # 男声
        "zh-CN-XiaoyiNeural",    # 女声
        "zh-CN-YunjianNeural",   # 男声
    ]

    def __init__(self, config: dict):
        """
        初始化语音合成器。

        参数:
            config: 全局配置字典，主要读取 tts 部分
        """
        tts_cfg = config.get("tts", {})

        self.voice = tts_cfg.get("voice", "zh-CN-XiaoxiaoNeural")
        self.rate = tts_cfg.get("rate", "+0%")    # 语速，如 "+10%" 加快 10%
        self.pitch = tts_cfg.get("pitch", "+0Hz") # 音调，如 "-10Hz" 降低

        # pygame 混音器是否已初始化
        self._mixer_initialized = False

    def _init_mixer(self):
        """
        初始化 pygame 音频混音器（只初始化一次）。
        """
        if self._mixer_initialized:
            return

        try:
            pygame.mixer.init()
            self._mixer_initialized = True
            logger.debug("pygame 混音器已初始化")
        except Exception as e:
            logger.error(f"pygame 初始化失败: {e}")
            raise RuntimeError(f"音频播放器初始化失败: {e}")

    async def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        将文字合成为语音 MP3 文件（不播放）。

        参数:
            text: 要合成的文本
            output_path: 输出文件路径，不传则使用临时文件

        返回:
            str: 生成的 MP3 文件路径
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("请安装 edge-tts: pip install edge-tts")

        # 如果用户指定了输出路径就用它，否则用临时文件
        if output_path:
            mp3_path = output_path
        else:
            # 创建临时文件（关闭后自动删除）
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            mp3_path = tmp.name
            tmp.close()

        # 创建 TTS 通信对象
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        logger.debug(f"正在合成语音: {text[:50]}...")
        await communicate.save(mp3_path)
        logger.debug(f"语音已合成到: {mp3_path}")

        return mp3_path

    async def speak(self, text: str) -> str:
        """
        将文字合成为语音并立即播放（主要的对外接口）。

        工作流程:
        1. 调用 Edge-TTS 将文本合成 MP3 文件
        2. 用 pygame 播放 MP3 文件
        3. 等待播放完毕
        4. 清理临时文件

        参数:
            text: 要朗读的文本内容

        返回:
            str: 已朗读的文本
        """
        if not text or not text.strip():
            return ""

        self._init_mixer()

        mp3_path = await self.synthesize(text)

        try:
            # 加载并播放 MP3
            pygame.mixer.music.load(mp3_path)
            pygame.mixer.music.play()

            # 等待播放完成
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

            logger.info(f"🔊 已播报: {text[:80]}{'...' if len(text) > 80 else ''}")

        except Exception as e:
            logger.error(f"音频播放失败: {e}")
            raise
        finally:
            # 清理临时文件
            try:
                Path(mp3_path).unlink(missing_ok=True)
            except Exception:
                pass

        return text

    def speak_sync(self, text: str) -> str:
        """
        同步版本的 speak()，方便在非 async 环境中调用。

        参数:
            text: 要朗读的文本内容

        返回:
            str: 已朗读的文本
        """
        return asyncio.run(self.speak(text))

    def close(self):
        """
        释放 pygame 音频资源。
        """
        try:
            pygame.mixer.quit()
            self._mixer_initialized = False
            logger.debug("pygame 混音器已关闭")
        except Exception:
            pass


# ============================================================
# 独立测试入口
# ============================================================
if __name__ == "__main__":
    """
    测试语音合成模块:
        python -m src.voice.tts

    如果能听到语音播报，说明模块正常工作。
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    test_config = {
        "tts": {
            "voice": "zh-CN-XiaoxiaoNeural",
            "rate": "+0%",
            "pitch": "+0Hz",
        },
        "logging": {"level": "INFO"},
    }

    async def main():
        tts = SpeechSynthesizer(test_config)
        try:
            print("=" * 50)
            print("语音合成测试")
            print("=" * 50)
            await tts.speak("您好，这里是AI语音助手，请问有什么可以帮您的？")
            print("\n✅ 播报完成")
        finally:
            tts.close()

    asyncio.run(main())
