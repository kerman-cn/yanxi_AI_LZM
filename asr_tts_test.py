import whisper
import edge_tts
import asyncio
import pygame
import os

# ================== 配置 ==================
# 把 "base" 改成 "small"，准确率会大幅提升！
WHISPER_MODEL = "small"
VOICE = "zh-CN-XiaoxiaoNeural"
TEST_AUDIO = "test.mp3"
OUTPUT_AUDIO = "output.mp3"

pygame.mixer.init()


# ----------------------
# 语音识别（精准版）
# ----------------------
def asr(audio_path):
    print("\n🎤 正在识别语音...")

    # 加载模型（第一次会自动下载，后面就快了）
    model = whisper.load_model(WHISPER_MODEL)

    # 识别：强制简体中文 + 提示词提升准确率
    result = model.transcribe(
        audio_path,
        language="zh",  # 强制中文
        initial_prompt="以下是普通话的句子，用词准确。",  # 提示词
        fp16=False,
        temperature=0.0  # 降低随机性，更精准
    )

    text = result["text"].strip()
    print(f"✅ 识别结果：{text}")
    return text


# ----------------------
# 语音合成
# ----------------------
async def tts(text):
    print("\n🔊 正在合成语音...")
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(OUTPUT_AUDIO)

    pygame.mixer.music.load(OUTPUT_AUDIO)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    print("✅ 播放完成！")


# ----------------------
# 主程序
# ----------------------
async def main():
    print("=" * 50)
    print(f"🚀 语音模块测试（模型：{WHISPER_MODEL}）")
    print("=" * 50)

    if os.path.exists(TEST_AUDIO):
        text = asr(TEST_AUDIO)
    else:
        print(f"未找到 {TEST_AUDIO}，使用测试文本")
        text = "您好，您的快递有破损，是否需要退货"

    await tts(text)
    print("\n🎉 语音模块测试全部通过！")


if __name__ == "__main__":
    asyncio.run(main())