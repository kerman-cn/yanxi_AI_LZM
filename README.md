# 言犀 AI 管家｜综合可视化版

言犀 AI 管家用于演示“上课时不方便接电话”的语音代接流程：输入录音或语音文件，在本地识别文字，再通过知识库和多个 AI Agent 分析场景、风险与处理方式，生成文字回复，并可合成语音播放。

当前根目录是 `PythonProject2` 的通义千问综合可视化版，推荐入口为 `comprehensive_gui.py`。它处理麦克风录音和本地音频文件，**尚未直接接管手机或运营商来电**；“拦截”“转人工”“报警”等是模型输出的建议或话术，不代表程序已执行这些现实操作。

仓库中另外保留了独立的 [桌面与评测版本（2026-08-31）](YanXi_project_desktop_evaluation_20260831/README.md)，该版本有自己的配置与启动说明，请勿混用。本文介绍根目录的综合可视化版。

## 1. 环境准备

本说明以 Windows 和 PowerShell 为例。建议使用 Python 3.12 的 64 位版本，安装时启用 Python Launcher 和 Tcl/Tk（界面使用 Tkinter）。

运行前需要：

- Python 3.12：用于运行程序和创建虚拟环境。
- FFmpeg：用于 Whisper 读取、解码音频。
- 可访问百炼服务的网络，以及能够调用对话、向量模型的百炼 API Key。
- 使用语音回复时，需要连接 Edge TTS 的在线语音合成服务；使用麦克风时，需要允许桌面应用访问麦克风。

Whisper 在本地识别音频，首次使用指定模型时可能需要下载模型文件。后续模型分析和向量计算仍需要网络，整套应用不是完全离线运行。

### 1.1 获取工程

新电脑可使用 Git 下载：

```powershell
git clone https://github.com/kerman-cn/yanxi_AI_LZM.git
cd .\yanxi_AI_LZM
```

也可以下载仓库 ZIP 后解压。如果已有本地 `PythonProject2`，直接在该文件夹打开终端，无需重复下载。

下文命令均在包含 `README.md`、`requirements.txt` 和 `comprehensive_gui.py` 的工程根目录执行。克隆后的文件夹默认叫 `yanxi_AI_LZM`，不需要再寻找一层 `PythonProject2`。

### 1.2 创建虚拟环境并安装依赖

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果当前电脑已有可用的 `.venv`，可以跳过创建步骤。不要从另一台电脑直接复制虚拟环境；应使用依赖文件重新安装。

本文直接调用虚拟环境中的 Python，不需要运行 `Activate.ps1`，也无需修改 PowerShell 执行策略。依赖中的 `openai` 是百炼兼容接口所使用的客户端库，本工程无需配置 OpenAI 平台的 API Key。

### 1.3 安装 FFmpeg

从 [FFmpeg 官方下载页](https://www.ffmpeg.org/download.html) 的 Windows EXE Files 区域选择 Windows 构建包。解压后，将包含 `ffmpeg.exe` 的 `bin` 目录添加到用户的 `Path` 环境变量，例如 `C:\tools\ffmpeg\bin`。

重新打开终端；若从 IDE 启动，也需重启 IDE。验证：

```powershell
ffmpeg -version
```

能够显示版本信息即可。如果系统已有 FFmpeg，不需要重复安装。

## 2. 配置百炼 API

### 2.1 获取 API Key

按 [阿里云官方的 API Key 获取说明](https://help.aliyun.com/zh/model-studio/get-api-key/) 在百炼密钥管理页面创建或复制 API Key，并确认所属业务空间有权访问需要使用的模型。

本项目默认使用中国大陆接口 `https://dashscope.aliyuncs.com/compatible-mode/v1`。Key 与接口所属区域应匹配；不要把阿里云 AccessKey ID / AccessKey Secret 当作百炼 API Key。更换服务区域时，请按官方说明同时检查 Key、接口地址和模型可用性。

### 2.2 从模板创建本机配置

仅在还没有 `.env` 时复制模板，避免覆盖已经配置好的密钥：

```powershell
if (-not (Test-Path -LiteralPath .\.env)) {
    Copy-Item -LiteralPath .\.env.example -Destination .\.env
}
notepad .\.env
```

将 `DASHSCOPE_API_KEY` 的示例值换成自己的百炼 API Key，其余配置第一次使用时保持默认：

```dotenv
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_CHAT_MODEL=qwen-plus
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_EMBEDDING_DIMENSIONS=1024
```

| 配置项 | 作用 | 默认值 / 要求 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 百炼鉴权凭证 | 必填，替换示例值；不要上传 |
| `QWEN_BASE_URL` | OpenAI 兼容接口地址 | 上述中国大陆接口 |
| `QWEN_CHAT_MODEL` | 场景、风险、业务处理与质检模型 | `qwen-plus` |
| `QWEN_EMBEDDING_MODEL` | 知识库向量模型 | `text-embedding-v4` |
| `QWEN_EMBEDDING_DIMENSIONS` | 向量维度 | `1024`，必须是所选模型支持的整数 |

保存时确认文件名是 `.env`，不是 `.env.txt`。不要把真实 Key 写入源代码、截图或 README。

程序读取工程根目录的 `.env`，但**已经存在的同名进程环境变量优先**。如果修改 `.env` 后没有生效，请检查系统或 IDE 是否配置了同名变量，并完全退出后重新启动程序。

切换对话模型时，模型还需兼容当前代码使用的 JSON 输出和 `enable_thinking` 参数。切换向量模型、维度，或修改知识库内容后，下次初始化会重建本机向量缓存，并产生新的向量接口调用。

### 2.3 本机自检

检查配置是否已被读取，但不显示密钥、不调用真实 API：

```powershell
.\.venv\Scripts\python.exe -B -c "from qwen_provider import QWEN_API_KEY; print('API_KEY_CONFIGURED' if QWEN_API_KEY and QWEN_API_KEY != 'sk-your-dashscope-api-key' else 'API_KEY_MISSING_OR_PLACEHOLDER')"
```

检查依赖和 FFmpeg：

```powershell
.\.venv\Scripts\python.exe -B -c "import tkinter, whisper, edge_tts, pygame, openai, dotenv, chromadb, langchain_community, langchain_chroma, sounddevice, soundfile, numpy; print('IMPORTS_OK')"
ffmpeg -version
.\启动综合可视化版.bat --check
```

`LAUNCHER_OK` 只说明启动脚本找到了入口文件且虚拟环境 Python 可以启动，**不代表依赖、FFmpeg、API Key 或网络全部通过验证**。真实 API 可用性需要通过一次音频分析确认。

## 3. 启动与使用

### 3.1 打开综合界面

完成配置后，双击 `启动综合可视化版.bat`。也可以在终端运行：

```powershell
.\.venv\Scripts\python.exe comprehensive_gui.py
```

排查错误时优先使用终端启动，以便查看完整报错。启动脚本不会自动安装依赖。

第一次分析可能需要下载或加载 Whisper 模型、调用向量接口建立知识库；等待状态和日志更新即可。后续任务会复用当前进程已经加载的模型和系统对象。

### 3.2 分析一个语音文件

1. 进入“语音文件”页面，点击“浏览文件”。
2. 选择一段音频。支持 MP3、WAV、M4A、FLAC、OGG、AAC。
3. 按需勾选“合成并播放 AI 回复”。首次验证可取消勾选，先检查文字分析链路。
4. 点击“开始分析”，等待识别、知识库检索和多 Agent 分析完成。
5. 查看识别文字、场景、风险等级、处理建议、最终回复和执行轨迹。

可先从 `data/test_audio_batch_200/` 中选择一条附带音频测试。建议先完成单条分析，再启动大批量任务。

### 3.3 使用麦克风录音

1. 进入麦克风页面，选择输入设备；插拔设备后点击“刷新”。
2. 按需勾选“停止后自动分析”和“合成并播放 AI 回复”。
3. 点击“开始录音”，说出模拟来电内容，然后点击“停止录音”。
4. 勾选自动分析时，程序会继续处理录音；未勾选时，仅保存录音，可随后在“语音文件”页面选择它。

单次录音至少 0.5 秒，最长 120 秒；达到上限会自动停止。录音保存在系统临时目录下的 `yanxi_comprehensive_gui/`，具体路径会显示在日志里。建议使用耳机，避免扬声器声音被麦克风再次录入。

### 3.4 批量测试与导出报告

1. 进入“批量自动测试”，使用默认测试目录或点击“选择目录”。
2. 设置“数量上限”：`0` 表示全部；首次可设置为 `3`，少量验证。
3. 音频在子文件夹中时，勾选“扫描子目录”。
4. 按需勾选“启用 AI 质检评分”。开启后，每条音频会增加一次模型评分调用。
5. 点击“开始测试”，查看处理进度与结果列表，双击结果查看详情。
6. 完成后，报告自动保存到 `outputs/test_reports/`；也可以点击“导出报告”另存 JSON。

批量模式只生成文字分析结果，不逐条播放 TTS。点击“停止”后，会等当前文件处理完再结束，不会立即中断正在进行的模型请求。已经完成的结果仍会保留；存在结果时会尝试自动保存报告。

默认目录名中的 `200` 是沿用的名称，实际测试数量以扫描结果为准；同目录的 `.txt` 文件不会被当作音频分析。一次“分析成功”仅表示流程完成，不等于风险判断或回复内容一定正确；AI 质检分数也不能替代人工核验。

### 3.5 其他入口

若只需要原有的文件选择、循环处理方式，可以运行：

```powershell
.\.venv\Scripts\python.exe call_agent.py
```

选择音频后程序进行分析和语音回复，取消文件选择则退出。日常使用仍建议统一从综合界面进入。

## 4. 数据、输出与自定义

主要目录如下（`.env`、缓存和输出文件由本机配置或运行产生，不随 Git 下载）：

```text
工程根目录/
├─ comprehensive_gui.py             # 综合界面：录音、单文件、批量测试
├─ call_agent.py                    # ASR、RAG、多 Agent、TTS
├─ qwen_provider.py                 # 百炼客户端和 Embedding 适配器
├─ auto_test_gui.py                 # 批量质检支持模块及独立测试界面
├─ 启动综合可视化版.bat             # Windows 启动入口
├─ requirements.txt                 # Python 依赖
├─ .env.example                     # 可提交的配置模板
├─ .env                             # 本机真实配置，不提交
├─ data/
│  ├─ knowledge.txt                 # 通话规则知识库
│  └─ test_audio_batch_200/          # 测试音频和部分配套文本
├─ tests/                           # 离线基础测试
├─ runtime/chroma_call_db_qwen/      # 可重建的向量缓存，不提交
├─ outputs/
│  ├─ ai_reply.mp3                  # 最近一次合成的回复，不提交
│  └─ test_reports/                 # 批量报告，不提交
└─ 历史归档/                        # 本机留存旧版本，不上传
```

知识库采用 UTF-8 文本，每行格式为“来电示例|场景类型|处理规则”，例如：

```text
您好，您的外卖到了|[外卖/快递]|告知机主正在上课，请放在约定位置
```

编辑 `data/knowledge.txt` 后重启应用，新内容会在下次初始化时重新建立向量缓存。不要在知识库中填写不适合发送给模型服务的密码或其他敏感信息。

`call_agent.py` 顶部的 `WHISPER_MODEL`、`VOICE` 和 `DEFAULT_LOCATION` 分别用于选择识别模型、合成音色和默认地点。它们不是 `.env` 配置项；修改后需要重启。若改变使用场景或地点，还应一并检查知识库和 Agent 提示词中的相关描述。

`outputs/ai_reply.mp3` 会随后续语音回复覆盖。需要长期保留的音频和报告请自行备份。报告可能包含录音文字、本机路径等信息，分享前请检查。

## 5. 运行离线测试

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

现有测试覆盖向量请求分批与排序、批量文件扫描、分析结果序列化和报告汇总，不调用真实 API，也不加载 Whisper 模型。这些测试不验证真实识别质量、录音设备或服务连通性。

## 6. 常见问题

| 现象 | 排查方法 |
| --- | --- |
| 双击 BAT 没有界面 | 在终端运行启动自检，再直接运行 `comprehensive_gui.py` 查看报错；`LAUNCHER_OK` 不是完整健康检查。 |
| 提示虚拟环境缺失或无效 | 按第 1.2 节用 Python 3.12 创建本机 `.venv`，再安装依赖；不要复用其他电脑复制来的环境。 |
| `ModuleNotFoundError` | 用本文的 `.venv\Scripts\python.exe -m pip` 安装依赖，确认 IDE 解释器也选中了该环境。 |
| `ffmpeg` 找不到或音频无法解码 | 在启动应用的同一终端检查 `ffmpeg -version`；修改 Path 后重启终端或 IDE。 |
| 未配置 API Key、401 或鉴权失败 | 检查 `.env` 文件名、示例值、Key 与接口区域、模型授权，以及覆盖 `.env` 的同名环境变量。 |
| `Connection error`、超时或 SSL/TLS 错误 | 检查网络、代理、证书与接口地址是否可用；不要通过关闭证书校验来绕过问题。 |
| 429 或额度相关错误 | 根据完整错误信息检查百炼限流、可用额度和账户状态；减少批量数量，必要时关闭 AI 质检。 |
| 第一次分析很慢 | 查看日志是否正在下载 / 加载模型或构建知识库；本机识别速度取决于硬件和模型。 |
| 找不到麦克风或没有录到声音 | 检查系统麦克风权限、输入设备和音量，刷新设备列表，并确认录音长度达到 0.5 秒。 |
| 有文字结果但无法播放回复 | 检查是否勾选 TTS、网络和播放设备；查看日志确认语音是否已保存到 `outputs/ai_reply.mp3`。 |
| 改了配置或知识库却没有变化 | 完全退出应用后重开；检查进程环境变量是否覆盖了 `.env`。 |

## 7. 上传范围与安全说明

Git 保留当前源代码、依赖文件、配置模板、测试、知识库和测试音频。`历史归档/` 只保存在本机，不上传；`.env`、`.venv/`、`.idea/`、Python 缓存、`runtime/` 和 `outputs/` 同样被忽略。请勿使用 `git add -f` 强行提交这些内容。

本地 `言犀AI管家_安卓录音版.apk` 包含内嵌鉴权凭证，已单独排除上传，不应直接对外分发。本文的配置与运行步骤面向 Python 工程，不涉及该 APK。

将文件加入 `.gitignore` 不会清除 Git 已有历史中的内容。若密钥曾经被提交，应在对应服务控制台撤销并更换；不要因为当前版本已移除密钥就继续信任旧 Key。

录音在本地做 Whisper 识别，但识别文字、检索相关文本和规则会用于云端模型调用；启用语音回复时，回复文本会发送给在线语音合成服务。使用前请确认有权处理相关录音，并避免提交或分享含私人通话内容的报告。
