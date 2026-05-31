"""
Batch test: run all MP3 files through the AI call agent pipeline (no TTS playback).
"""
import os
import sys
import json
import asyncio
import time

# Import modules from call_agent (not the TTS, to save time)
from call_agent import (
    init_system, ASREngine, RAGRetriever, DeepSceneAnalyzer,
    RiskControlAgent, BusinessProcessAgent, ManualTransferAgent,
    CallAgentScheduler, LLM_MODEL, WHISPER_MODEL,
)
from zhipuai import ZhipuAI

ZHIPU_API_KEY = "df0ae242b97e4e92beed7bbed62f504e.bOfZ4YoZ3xYVlthi"


async def test_single(asr, rag, scheduler, mp3_path):
    """Test a single MP3 file, return result dict."""
    fname = os.path.basename(mp3_path)
    try:
        call_text = asr.transcribe(mp3_path)
        rag_hint = rag.search(call_text)
        ctx = scheduler.handle(call_text, rag_hint)

        return {
            "file": fname,
            "status": "OK",
            "call_text": call_text,
            "risk_level": ctx.risk_result.get("risk_level", "N/A"),
            "risk_type": ctx.risk_result.get("risk_type", "N/A"),
            "handle": ctx.risk_result.get("handle_suggestion", "N/A"),
            "final_reply": ctx.final_reply,
            "reason": ctx.risk_result.get("reason", ""),
        }
    except Exception as e:
        return {"file": fname, "status": "ERROR", "error": str(e)}


async def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    mp3_files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith(".mp3") and f != "ai_reply.mp3"
    ])

    print(f"\nFound {len(mp3_files)} MP3 files to test\n")
    print("=" * 90)

    # Init system once
    asr, rag, scheduler, _ = init_system()

    results = []
    for i, mp3 in enumerate(mp3_files, 1):
        fname = os.path.basename(mp3)
        print(f"\n[{i}/{len(mp3_files)}] Testing: {fname}")
        print("-" * 50)

        result = await test_single(asr, rag, scheduler, mp3)
        results.append(result)

        if result["status"] == "OK":
            print(f"  Text  : {result['call_text']}")
            print(f"  Level : {result['risk_level']}")
            print(f"  Type  : {result['risk_type']}")
            print(f"  Action: {result['handle']}")
            print(f"  Reply : {result['final_reply']}")
        else:
            print(f"  ERROR : {result['error']}")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'File':<24s} {'Risk Level':<12s} {'Action':<12s} {'Reply':<40s}")
    print("-" * 90)

    for r in results:
        if r["status"] == "OK":
            reply = r["final_reply"][:38] + ".." if len(r["final_reply"]) > 40 else r["final_reply"]
            print(f"{r['file']:<24s} {r['risk_level']:<12s} {r['handle']:<12s} {reply:<40s}")
        else:
            print(f"{r['file']:<24s} {'ERROR':<12s} {'--':<12s} {r['error']:<40s}")

    # Stats
    ok = [r for r in results if r["status"] == "OK"]
    blocked = [r for r in ok if r["handle"] in ("拦截", "拒绝")]
    forwarded = [r for r in ok if r["handle"] == "转人工"]
    processed = [r for r in ok if r["handle"] == "流转业务处理"]

    print("-" * 90)
    print(f"Total: {len(results)} | Intercepted: {len(blocked)} | "
          f"Transferred: {len(forwarded)} | Business: {len(processed)} | "
          f"Errors: {len(results) - len(ok)}")

    # Save detailed results
    out_path = os.path.join(folder, "batch_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
