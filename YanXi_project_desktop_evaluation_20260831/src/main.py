"""AI 语音通话助手 — 主入口 v4 (整合版)
融合 QM-newer 架构 + CC 完整功能实现。
运行方式:
    python src/main.py               # 文本模式（默认）
    python src/main.py --text        # 文本模式
    python src/main.py --test        # 测试模式
"""

import argparse
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestration.orchestrator import CallOrchestrator
from src.core.config import load_and_validate_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def print_banner():
    banner = r"""
=========================================================
  言犀 - AI 智能通话助手 v4 (整合版)
=========================================================
  [诈骗检测] -> 拦截拒接
  [业务来电] -> 对话 -> 摘要卡片
  [紧急来电] -> 转接机主
  [习惯学习] -> 自动调整策略
  [来电画像] -> 信任评分 + 黑白名单
  [对话记忆] -> 多轮上下文
  [多后端 LLM] -> DeepSeek / GLM / Qwen
  [混合检索] -> BM25 + 向量 双路
=========================================================
  输入 /help 查看命令
=========================================================
"""
    print(banner)


def print_help():
    help_text = """
命令帮助:
  来电模拟:
    直接输入文字模拟来电，格式: [号码] 内容

  习惯学习:
    /habit 我要自习一下午
    /habit 我每天晚上11点到7点睡觉
    /habit end 自习
    /habits

  来电画像:
    /profile 13800001111
    /blacklist 13800001111
    /whitelist 13800001111

  统计与记录:
    /stats          今日通话统计
    /recordings     最近录音列表
    /notifications  最近通知卡片

  其他:
    /help  显示帮助
    /quit  退出
"""
    print(help_text)


def handle_command(text: str, orchestrator: CallOrchestrator) -> bool:
    text = text.strip()
    if not text:
        return True

    if text in ("/quit", "/exit", "/q"):
        print("再见！")
        return False

    if text == "/help":
        print_help()
        return True

    if text.startswith("/habit "):
        user_input = text[7:].strip()
        if user_input.lower().startswith("end"):
            keyword = user_input[3:].strip()
            result = orchestrator.end_activity(keyword)
            if result:
                print(f"[OK] 已结束活动: {result}")
            else:
                print("[INFO] 没有正在进行的活动")
        else:
            result = orchestrator.learn_habit(user_input)
            print(f"[HABIT] {result.get('reply', '已记录')}")
            if result.get("mode_change"):
                print(f"   状态变化: {result['mode_change']}")
        return True

    if text == "/habits":
        print(orchestrator.get_habits_summary())
        return True

    if text.startswith("/profile "):
        number = text[9:].strip()
        profile = orchestrator.get_caller_profile(number)
        if profile:
            print(f"[PROFILE] 号码: {profile.phone_number}")
            print(f"   来电次数: {profile.call_count}")
            print(f"   信任分数: {profile.trust_score:.2f}")
            print(f"   标签: {profile.tags}")
            print(f"   黑名单: {'是' if profile.is_blacklisted else '否'}")
            print(f"   白名单: {'是' if profile.is_whitelisted else '否'}")
        else:
            print(f"[INFO] 未找到号码 {number} 的画像")
        return True

    if text.startswith("/blacklist "):
        number = text[11:].strip()
        orchestrator.blacklist_caller(number)
        print(f"[BLACKLIST] 已将 {number} 加入黑名单")
        return True

    if text.startswith("/whitelist "):
        number = text[11:].strip()
        orchestrator.whitelist_caller(number)
        print(f"[WHITELIST] 已将 {number} 加入白名单")
        return True

    if text == "/stats":
        stats = orchestrator.get_today_stats()
        profile_stats = orchestrator.get_profile_stats()
        print(f"[STATS] 今日通话统计:")
        print(f"   总计: {stats.get('total', 0)} 次")
        print(f"   按动作: {stats.get('by_action', {})}")
        print(f"   按类型: {stats.get('by_type', {})}")
        print(f"[STATS] 来电画像统计:")
        print(f"   总画像: {profile_stats.get('total_profiles', 0)} 个")
        print(f"   黑名单: {profile_stats.get('blacklisted', 0)} 个")
        print(f"   白名单: {profile_stats.get('whitelisted', 0)} 个")
        print(f"   平均信任: {profile_stats.get('avg_trust', 0):.2f}")
        return True

    if text == "/recordings":
        recordings = orchestrator.get_recent_recordings(5)
        if recordings:
            print(f"[RECORDINGS] 最近录音:")
            for r in recordings:
                print(f"   {r.call_id} | {r.duration_sec:.1f}s | {r.call_type or '?'} | {r.filepath}")
        else:
            print("[INFO] 暂无录音记录")
        return True

    if text == "/notifications":
        notifications = orchestrator.get_recent_notifications(5)
        if notifications:
            print(f"[NOTIFICATIONS] 最近通知:")
            for n in notifications:
                print(f"   {n.title} | {n.body[:30]} | {n.priority}")
        else:
            print("[INFO] 暂无通知记录")
        return True

    if text.startswith("/"):
        print(f"[?] 未知命令: {text}，输入 /help 查看帮助")
        return True

    return True


def parse_caller_input(text: str) -> tuple[str, str]:
    match = re.match(r"^(\d{5,15})\s+(.+)$", text)
    if match:
        return match.group(1), match.group(2)
    return "", text


def run_text_mode(orchestrator: CallOrchestrator):
    print("\n[TEST] 文本模式已启动（输入 /help 查看命令）\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if not handle_command(user_input, orchestrator):
                break
            continue

        caller_number, call_text = parse_caller_input(user_input)
        if not call_text:
            continue

        result = orchestrator.run(call_text, caller_number=caller_number)

        print(f"\n[RESULT] 处理结果:")
        print(f"   类型: {result.get('call_type_name', '?')} "
              f"(置信度{result.get('confidence', 0):.0%}, "
              f"方法={result.get('classify_method', '?')})")
        print(f"   动作: {result.get('final_action', '?')}")
        presence_mode = result.get('presence_mode', 'free')
        if presence_mode != 'free':
            print(f"   机主状态: {presence_mode} ({result.get('presence_reason', '')})")
        if result.get('agent_reply'):
            print(f"   言犀: {result['agent_reply']}")
        if result.get('notification_card'):
            card = result['notification_card']
            print(f"   卡片: {card.get('title', '')} - {card.get('body', '')[:40]}")
        if caller_number:
            profile = orchestrator.get_caller_profile(caller_number)
            if profile and profile.call_count > 0:
                print(f"   画像: 信任={profile.trust_score:.2f} 来电={profile.call_count}次 标签={profile.tags}")
        print()


def run_test_mode(orchestrator: CallOrchestrator):
    print("\n[TEST] 运行测试用例...\n")

    tests = [
        ("13800001111", "我是美团外卖的，你的餐到楼下了", "food_delivery"),
        ("13800002222", "您好，我是市公安局的，你涉嫌洗钱案件", "scam"),
        ("13800003333", "你的快递到菜鸟驿站了", "express"),
        ("13800004444", "妈，我今天晚上回家吃饭", "family"),
        ("13800005555", "领导，明天下午有个紧急会议", "leader"),
        ("13800001111", "美团外卖，你上一单的餐到了", "food_delivery"),
    ]

    for number, text, expected in tests:
        result = orchestrator.run(text, caller_number=number)
        actual = result.get('type_id', '?')
        action = result.get('final_action', '?')
        reply = result.get('agent_reply', '')[:50]
        card = result.get('notification_card')
        card_info = f'[CARD] {card["title"]}' if card else ''
        match = '[OK]' if expected in actual or actual in expected else '[?]'
        print(f"{match} [{number}] {text[:25]}...")
        print(f"   分类: {actual} | 动作: {action} | {card_info}")
        print(f"   回复: {reply}...")
        profile = orchestrator.get_caller_profile(number)
        if profile:
            print(f"   信任={profile.trust_score:.2f} 来电={profile.call_count}次 标签={profile.tags}")
        print()

    print("=" * 60)
    print("[HABIT] 习惯学习测试:")
    print("=" * 60)
    habit_result = orchestrator.learn_habit("我要自习一下午")
    print(f"回复: {habit_result.get('reply', '')}")
    print(f"\n{orchestrator.get_habits_summary()}")

    print("\n" + "=" * 60)
    print("[BLACKLIST] 黑名单测试:")
    print("=" * 60)
    orchestrator.blacklist_caller("13800002222")
    result = orchestrator.run("你好，我是检察院的", caller_number="13800002222")
    print(f"黑名单号码来电 -> 动作: {result.get('final_action')}")

    print("\n" + "=" * 60)
    print("[STATS] 统计:")
    print("=" * 60)
    stats = orchestrator.get_today_stats()
    print(f"今日通话: {stats.get('total', 0)} 次")
    profile_stats = orchestrator.get_profile_stats()
    print(f"画像总数: {profile_stats.get('total_profiles', 0)}")
    print(f"黑名单: {profile_stats.get('blacklisted', 0)}")
    print(f"白名单: {profile_stats.get('whitelisted', 0)}")
    print("\n[DONE] 测试完成！")


def main():
    parser = argparse.ArgumentParser(
        description="言犀 — AI 智能通话助手 v4 (整合版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python src/main.py              文本模式（默认）
  python src/main.py --text       文本模式
  python src/main.py --test       测试模式
        """,
    )
    parser.add_argument("--text", action="store_true", help="文本模式")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--gui", action="store_true", help="打开桌面工作台")
    args = parser.parse_args()

    if args.gui:
        from src.desktop.app import main as desktop_main
        desktop_main()
        return

    print_banner()

    logger.info("正在加载配置...")
    config = load_and_validate_config("config.yaml")

    logger.info("正在初始化调度器和子模块...")
    try:
        orchestrator = CallOrchestrator(config)
        logger.info("所有模块初始化完成")
        print(f"  LLM 后端: {orchestrator.llm_client.backend}")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.test:
        run_test_mode(orchestrator)
    else:
        run_text_mode(orchestrator)


if __name__ == "__main__":
    main()
