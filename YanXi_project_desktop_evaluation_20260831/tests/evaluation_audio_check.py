"""Real local ASR + real SUT graph, with explicitly mocked API responses. Not a quality benchmark."""
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from src.evaluation.models import discover_cases
from src.evaluation.service import EvaluationService, run_evaluation
from src.evaluation.settings import DEFAULT_DATASET, EvaluationSettings
from tests.test_evaluation import FakeClient, sample_judgment


class OfflineLLM:
    def __init__(self, config):
        self.last_error = None

    def chat(self, **kwargs):
        if kwargs.get("max_tokens") == 20:
            return "general"
        return "这是模拟API回复，仅用于验证程序连通性，不代表模型评审结果。"

    def close(self):
        pass


def main():
    cases = [c for c in discover_cases([DEFAULT_DATASET]) if c.name in (
        "外卖_03_菜品售罄换餐", "诈骗_04_冒充学校急救")]
    assert len(cases) == 2
    with tempfile.TemporaryDirectory(prefix="yanxi_eval_audio_") as temporary:
        service = EvaluationService(EvaluationSettings(api_key="TEST-NO-NETWORK", stt_model="small"), Path(temporary))
        service._judge_client = FakeClient([sample_judgment(), sample_judgment()])
        def emit(kind, value):
            if kind == "progress":
                print(value, flush=True)
        with patch("src.orchestration.orchestrator.LLMClient", OfflineLLM), \
             patch("src.knowledge.habit_learner.LLMClient", OfflineLLM):
            run_evaluation(cases, service, threading.Event(), emit, allow_download=False)
        for case in cases:
            assert case.status == "已评审", (case.name, case.error)
            assert case.transcript.strip()
            assert case.output.get("final_action")
            print("LOCAL_ASR_OK:", case.name, case.elapsed, "seconds", "transcript_chars=", len(case.transcript), flush=True)
        print("AUDIO_PIPELINE_OK: real ASR + graph + MOCKED LLM/judge; NOT real model evaluation.", flush=True)


if __name__ == "__main__":
    main()
