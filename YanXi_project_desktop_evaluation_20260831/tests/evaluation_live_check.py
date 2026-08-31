"""Explicitly opted-in, small live check; does not modify GUI settings or source audio."""
import argparse
import threading
from dataclasses import replace

from src.desktop.settings import SettingsStore
from src.evaluation.models import discover_cases, metrics, write_json
from src.evaluation.service import EvaluationService, new_run_directory, run_evaluation
from src.evaluation.settings import DEFAULT_DATASET, EvaluationSettings, EvaluationSettingsStore


def main():
    parser = argparse.ArgumentParser(description="Small paid live evaluation check (2 original audio + optional 1 generated audio).")
    parser.add_argument("--live", action="store_true", help="Explicit consent to API requests")
    parser.add_argument("--generate", action="store_true", help="Generate, deduplicate and evaluate one synthetic audio call")
    parser.add_argument("--stt-model", default="small")
    args = parser.parse_args()
    if not args.live:
        parser.error("Requires --live; this test sends test-case texts to Qwen and may incur charges.")
    settings = EvaluationSettingsStore().load()
    if not settings.api_key:
        desktop = SettingsStore().load()
        settings = EvaluationSettings(api_key=SettingsStore().saved_key("qwen"), region=desktop.region)
    settings = replace(settings, stt_model=args.stt_model, remember_key=False)
    settings.validate()
    directory = new_run_directory()
    print("LIVE_CHECK_REPORT:", directory, flush=True)
    dataset = discover_cases([DEFAULT_DATASET])
    print("DATASET:", len(dataset), "WITH_REFERENCE:", sum(bool(c.reference) for c in dataset), flush=True)
    selected = [c for c in dataset if c.name in ("外卖_03_菜品售罄换餐", "诈骗_04_冒充学校急救")]
    if len(selected) != 2:
        raise ValueError("Cannot find the two expected test fixtures; no automatic substitute.")
    service = EvaluationService(settings, directory)
    stop = threading.Event()
    try:
        if args.generate:
            generated = []
            result = service.reviewer.generate(dataset, 1, "不存在于原集合的公共设施或宠物照护场景，切勿只替换姓名和金额", stop,
                lambda m: print(m, flush=True), generated.append)
            write_json(directory / "generation.json", result)
            if not generated:
                raise RuntimeError("生成未得到可接纳的新场景；请查看generation.json")
            generated[0].mode = "tts"
            write_json(directory / "generated_cases.json", [c.export() for c in generated])
            selected.extend(generated)
        def emit(kind, value):
            if kind in ("notice", "progress"):
                print(value, flush=True)
            elif kind == "case" and value.status not in ("待评测", "处理中"):
                print("CASE:", value.name, value.status, value.judgment.get("verdict"),
                    value.judgment.get("score"), value.error, flush=True)
        run_evaluation(selected, service, stop, emit, allow_download=False)
        print("LIVE_CHECK_SUMMARY:", metrics(selected), flush=True)
        if any(c.status != "已评审" for c in selected):
            raise SystemExit(1)
    finally:
        service.close()


if __name__ == "__main__":
    main()
