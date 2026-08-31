# 言犀桌面与评测分享版

本目录是 2026-08-31 的修改版源码快照，同时包含日常管家桌面程序和独立评测台。
新增目录之外的仓库原文件保持不变。本地原工程及其 Git 远程配置也未因本次分享而修改。

## 版本来源与保留方式

- 修改版的基础工程来自同学分享的 [QM-newer/YanXi_project](https://github.com/QM-newer/YanXi_project)。
- 本地基础提交：`bea7322610c1f6936edbe1f78988820a8972f997`。
- 本次仅向 `kerman-cn/yanxi_AI_LZM` 新增此独立目录；没有向同学仓库推送。
- 沿用原工程文件中的作者、来源及说明；此分享说明不另行授予或变更原项目的许可。

## 同学下载后如何运行

下载整个仓库 ZIP 或克隆仓库，然后进入本目录，保持 `src/`、入口文件和配置文件的相对位置。
不要只下载 `desktop.py` 或 `evaluation.py`。可以放到不同电脑、盘符和文件夹，无须复现作者的绝对路径。

安装 Python 3.12，在本目录打开 PowerShell，执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
```

然后任选一个入口：

- 日常 AI 管家：双击 `启动言犀桌面版.bat`，或运行 `.\.venv\Scripts\python.exe desktop.py`。
- AI 管家评测台：双击 `启动言犀评测台.bat`，或运行 `.\.venv\Scripts\python.exe evaluation.py`。

首次启动填写自己的 API Key，后续可在设置中修改。评测台和日常管家的密钥设置相互独立；本分享版不含作者的密钥。
默认 Whisper 识别需要 FFmpeg 加入 PATH，并需要首次下载识别模型；也可在设置中选择 Faster-Whisper。
Edge TTS 需要联网。Python 依赖和语音模型没有打包，这不是免安装 EXE 或 APK。

## 本次包含的内容

- `desktop.py`、`src/desktop/`：API 设置、麦克风录音、本地录音批处理、文字输入、输出查看与朗读。
- `evaluation.py`、`src/evaluation/`：批量评测、独立大模型评审、新来电生成与去重、报告与历史恢复。
- `test_audio_batch_200/`：262 段测试 MP3，另附 `文本/` 中的 200 份参考原文。
- `tests/`：桌面与评测回归检查、界面检查及显式开启的真实 API 检查脚本。
- [桌面使用说明](DESKTOP_README.md)、[评测使用说明](EVALUATION_README.md)、[评测 Agent 技术报告](AI管家评测Agent技术报告.md)。

目录名虽带 `200`，实际音频数量为 262；其中 62 段没有配套原文，评测时需人工复核。

## 未上传的内容与隐私

排除了 `.env`、`data/`、`.venv/`、原项目 `.git/`、模型缓存、IDE 缓存和本地任务便笺。
因此不包含作者的 API Key、私人麦克风录音、历史转写、评测运行报告或虚拟环境。
只对白名单测试数据目录放行 MP3，其他录音仍由 `.gitignore` 忽略。

新机器运行时，密钥默认在本地明文保存，不能把 `.env` 或整个 `data/` 再分享出去。
音频识别在本地进行；管家和评审会把相关文字发送给所选 API 服务，朗读文字会发送给 Edge TTS。
批量评测、生成新来电及连接检查可能消耗 API 额度，建议先试 2–3 条。

## 验证与功能边界

不联网、不使用真实密钥的回归命令：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_evaluation tests.test_desktop
```

本版本的这两组离线回归共 64 项；离线通过不等于真实大模型质量达标。
本次发布未执行全量 262 段真实 API 评测，也没有随包附带伪装成真实结果的报告。
真实 API 检查须主动执行带 `--live` 参数的脚本，详见评测使用说明。

当前主要覆盖单轮来电内容分析；接听、转接等是处理建议，不操作真实电话。
模型评审可能出错，尤其是诈骗、紧急场景及不确定结果必须人工复核；原工程尚未接通的 RAG 等能力不因此自动完成。
