import whisper
import edge_tts
import asyncio
import pygame
import os

# ================== 【新增】RAG 相关依赖导入 ==================
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
import shutil

# ================== 全局配置区 ==================
# --- 语音配置 ---
WHISPER_MODEL = "small"
VOICE = "zh-CN-XiaoxiaoNeural"
INPUT_AUDIO = "test.mp3"
OUTPUT_REPLY_AUDIO = "reply_audio.mp3"

# --- 智谱 RAG 配置（直接沿用你原来的配置）---
ZHIPU_API_KEY = "df0ae242b97e4e92beed7bbed62f504e.bOfZ4YoZ3xYVlthi"
KNOWLEDGE_FILE_PATH = "./knowledge.txt"
SIMILARITY_THRESHOLD = 1.5
CHROMA_DB_PATH = "./chroma_db"

# ================== 全局变量（用于缓存模型和数据库）==================
asr_model = None
vector_db = None
pygame.mixer.init()


# ================== 【模块 0】RAG 系统初始化（启动时运行一次）==================
def init_rag_system():
    """
    初始化 RAG 系统：加载文档、构建/加载向量库
    这部分逻辑直接提取自你原来的 zhipu_API_for_txt.py
    """
    global vector_db
    print("\n" + "=" * 70)
    print("🧠 正在初始化智谱 RAG 知识库系统...")
    print("=" * 70)

    # 1. 检查知识文件
    if not os.path.exists(KNOWLEDGE_FILE_PATH):
        raise FileNotFoundError(f"❌ 找不到知识文件：{KNOWLEDGE_FILE_PATH}，请确保文件在同级目录下")

    try:
        # 2. 加载并清洗文档（完全沿用你的清洗逻辑）
        print("📖 正在加载并清洗文档...")
        loader = TextLoader(KNOWLEDGE_FILE_PATH, encoding='utf-8')
        docs = loader.load()
        raw_text = docs[0].page_content

        # 你的特定清洗规则
        cleaned_text = raw_text.replace(
            "|[外卖/快递]|kk上课时段自动代接，告知放在指定地点，提取信息推送kk短信", "")
        cleaned_text = cleaned_text.replace("|[诈骗/推销]|直接拦截，挂断后推送风险提醒给kk", "")

        lines = cleaned_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line:
                clean_content = line.split('|')[0]
                clean_lines.append(clean_content)

        splits = [Document(page_content=line) for line in clean_lines]

        if len(splits) == 0:
            raise ValueError("❌ knowledge.txt 中没有有效内容")

        # 3. 构建向量库（如果已存在则删除重建，保证数据最新）
        print("🔗 正在连接智谱 AI 并构建向量库...")
        embeddings = ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY)

        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)

        vector_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )

        # 分批入库
        batch_size = 32
        total_count = len(splits)
        for i in range(0, total_count, batch_size):
            batch = splits[i: i + batch_size]
            vector_db.add_documents(batch)

        print(f"✅ RAG 系统初始化成功！共加载 {len(splits)} 条规则。")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"❌ RAG 初始化失败：{e}")
        raise


# ================== 【模块 1】ASR 语音识别 ==================
def speech_to_text(audio_path):
    global asr_model
    print("\n🎤 【1/4】正在进行语音识别...")

    if asr_model is None:
        print("   (首次运行正在加载 Whisper 模型，请稍候...)")
        asr_model = whisper.load_model(WHISPER_MODEL)

    result = asr_model.transcribe(
        audio_path,
        language="zh",
        initial_prompt="以下是关于快递、外卖、电话拦截的普通话简体中文句子。",
        fp16=False,
        temperature=0.0
    )
    question_text = result["text"].strip()
    print(f"✅ 用户提问：{question_text}")
    return question_text


# ================== 【模块 2】调用智谱 RAG 检索（核心整合部分）==================
def rag_get_answer(question_text):
    print("\n🔍 【2/4】正在调用智谱 RAG 检索知识库...")

    # 获取数据库总数
    total_docs_in_db = vector_db._collection.count()

    # 检索所有记录
    results_with_score = vector_db.similarity_search_with_score(
        question_text,
        k=total_docs_in_db
    )

    # 基于阈值过滤
    relevant_records = []
    for doc, score in results_with_score:
        if score < SIMILARITY_THRESHOLD:
            relevant_records.append((doc, score))

    # 生成最终的回复文本
    if relevant_records:
        print(f"✅ 找到 {len(relevant_records)} 条相关规则：")
        answer_parts = ["根据知识库，为您找到以下相关信息："]
        for idx, (doc, score) in enumerate(relevant_records, 1):
            content = doc.page_content
            print(f"   {idx}. {content} (匹配度: {score:.4f})")
            answer_parts.append(f"{idx}. {content}")

        final_answer = "。".join(answer_parts)
    else:
        final_answer = "不好意思，我在知识库中没有找到相关的规则。"
        print("❌ 未找到匹配的规则")

    return final_answer


# ================== 【模块 3】TTS 语音合成 ==================
async def text_to_speech(answer_text):
    print("\n🔊 【3/4】正在生成回复语音...")
    communicate = edge_tts.Communicate(answer_text, VOICE)
    await communicate.save(OUTPUT_REPLY_AUDIO)

    print("▶️  正在播放回复语音...")
    pygame.mixer.music.load(OUTPUT_REPLY_AUDIO)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    print(f"✅ 语音播放完成，文件已保存至：{OUTPUT_REPLY_AUDIO}")


# ================== 【主程序】全链路串联 ==================
async def main():
    try:
        # 0. 系统启动：先初始化 RAG
        init_rag_system()

        print("🚀 系统就绪，请开始测试...")

        # 1. 语音识别
        question = speech_to_text(INPUT_AUDIO)

        # 2. RAG 检索
        answer = rag_get_answer(question)

        # 3. 语音合成
        await text_to_speech(answer)

        print("\n🎉 【4/4】全流程测试通过！任务完成！")

    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")


if __name__ == "__main__":
    # 检查是否需要安装新依赖
    try:
        import langchain_community
        import langchain_chroma
    except ImportError:
        print("💡 检测到缺少 RAG 相关依赖，正在自动安装...")
        import subprocess

        subprocess.check_call(["pip", "install", "langchain", "langchain-community", "langchain-chroma", "zhipuai"])
        print("✅ 依赖安装完成，请重新运行程序！")
        exit()

    asyncio.run(main())