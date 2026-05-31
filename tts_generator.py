"""
语音生成器：输入文字，生成 MP3 语音文件（用于测试 AI 代接系统）
用法：
  python tts_generator.py "来电文字" 文件名.mp3
  python tts_generator.py -f 文本文件.txt    （批量模式，每行一条）
"""
import asyncio
import os
import sys
import edge_tts

VOICE = "zh-CN-XiaoxiaoNeural"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


async def generate(text: str, filename: str, voice: str = VOICE):
    mp3_path = os.path.join(OUTPUT_DIR, filename)
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="+0%")
    await communicate.save(mp3_path)
    size = os.path.getsize(mp3_path) / 1024
    print(f"  已生成: {filename} ({size:.1f} KB)")


async def main():
    if len(sys.argv) < 2:
        print("用法:")
        print('  python tts_generator.py "来电文字" 文件名.mp3')
        print("  python tts_generator.py -f 文本文件.txt")
        print()
        print("文本文件格式（每行一条）:")
        print("  来电文字|文件名.mp3")
        return

    if sys.argv[1] == "-f":
        # 批量模式：从文件读取
        file_path = sys.argv[2]
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        count = 0
        for line in lines:
            parts = line.split("|", 1)
            if len(parts) == 2:
                text, fname = parts[0].strip(), parts[1].strip()
            else:
                text = parts[0].strip()
                fname = f"test_{count + 1:02d}.mp3"

            if not fname.endswith(".mp3"):
                fname += ".mp3"

            await generate(text, fname)
            count += 1
        print(f"\n共生成 {count} 个文件")
    else:
        # 单条模式
        text = sys.argv[1]
        fname = sys.argv[2] if len(sys.argv) > 2 else "test.mp3"
        if not fname.endswith(".mp3"):
            fname += ".mp3"
        await generate(text, fname)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
