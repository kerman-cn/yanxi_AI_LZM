"""不加载模型、不访问网络的综合界面基础测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from comprehensive_gui import PipelineService, YanxiComprehensiveGUI


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class ComprehensiveGuiTests(unittest.TestCase):
    def test_pipeline_result_is_report_serializable(self):
        service = PipelineService()
        service.asr = SimpleNamespace(transcribe=lambda path: "您好，外卖到了")
        service.rag = SimpleNamespace(search=lambda text: "外卖规则")
        context = SimpleNamespace(
            final_reply="放在北门就行，谢谢。",
            risk_result={"risk_level": "1级-安全", "risk_type": "正常外卖", "handle_suggestion": "流转业务处理"},
            business_result={"business_type": "外卖", "handle_result": "已处理"},
            transfer_result={},
            logs=[SimpleNamespace(step="场景分析", content="类别=外卖配送, 紧急度=一般")],
        )
        service.scheduler = SimpleNamespace(handle=lambda text, hint: context)
        service.tts = None

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            result = service.analyze(Path(audio.name), False, lambda message: None)

        self.assertEqual(result["scene_category"], "外卖配送")
        self.assertEqual(result["business_type"], "外卖")
        self.assertIn("北门", result["final_reply"])
        json.dumps(result, ensure_ascii=False)

    def test_batch_scan_filters_audio_and_applies_limit(self):
        app = YanxiComprehensiveGUI.__new__(YanxiComprehensiveGUI)
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "a.mp3").touch()
            (folder / "b.wav").touch()
            (folder / "ignore.txt").touch()
            app.batch_dir_var = _Value(str(folder))
            app.batch_recursive_var = _Value(False)
            app.batch_limit_var = _Value("1")
            files = app._scan_batch_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].suffix, ".mp3")

    def test_batch_report_contains_summary(self):
        app = YanxiComprehensiveGUI.__new__(YanxiComprehensiveGUI)
        app.batch_dir_var = _Value("demo")
        app.batch_recursive_var = _Value(False)
        app.batch_judge_var = _Value(True)
        app.batch_results = [
            {
                "file": "a.mp3",
                "status": "OK",
                "evaluation": {"overall_verdict": "PASS", "total_score": 88.0},
            },
            {"file": "b.mp3", "status": "ERROR", "error": "demo"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "report.json"
            app._write_batch_report(target)
            report = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["total"], 2)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["errors"], 1)


if __name__ == "__main__":
    unittest.main()
