"""感知层 golden fixture 的回放测试。

用录制下来的真实游戏帧回放整条感知链路（OCR + 模板匹配 + 区域统计），
断言输出与基线一致。这样「不接触真实游戏也能验证感知层没被改坏」，
是后续拆分 InteractionBGD、合并地图匹配栈这类改动的回归网。

fixtures 不存在时跳过（fixtures 依赖录制时的真实游戏画面，无法凭空生成）。
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import golden_probes as probes  # noqa: E402

FIXTURE_DIR = os.path.join(HERE, "fixtures", "perception")
FRAMES_DIR = os.path.join(FIXTURE_DIR, "frames")
BASELINE_PATH = os.path.join(FIXTURE_DIR, "baseline.json")

# 容差：只吸收浮点/线程调度带来的抖动，不吸收真实的行为变化
MEAN_TOLERANCE = 2.0
REGION_TOLERANCE = 2.0
ICON_TOLERANCE = 0.03


@unittest.skipUnless(os.path.exists(BASELINE_PATH), "缺少感知层 golden fixtures")
class PerceptionGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(BASELINE_PATH, encoding="utf-8") as handle:
            cls.payload = json.load(handle)
        cls.templates = probes.load_icon_templates()

    def test_fixtures_are_not_degenerate(self) -> None:
        """fixtures 本身必须是「有画面」的帧。

        这条断言存在的理由很直接：截图链路曾经在返回全黑帧的同时报告
        fresh=True，如果 fixtures 恰好在那种状态下录下来，整套回归网都会
        对着空气比对。所以先证明 fixtures 非全黑、且 OCR 能读出文字。
        """
        for name in self.payload["frames"]:
            baseline = self.payload["baselines"][name]
            self.assertGreater(baseline["mean"], 5.0, f"{name} 基线亮度异常，疑似全黑帧")
            self.assertTrue(
                any(len(line) >= 2 for line in baseline["ocr_lines"]),
                f"{name} 基线里没有可识别的文字，fixtures 可能录自异常画面",
            )

    def test_replay_matches_baseline(self) -> None:
        for name in self.payload["frames"]:
            baseline = self.payload["baselines"][name]
            frame_path = os.path.join(FRAMES_DIR, name)
            self.assertTrue(os.path.exists(frame_path), f"缺少 fixture 帧 {name}")
            frame = probes.read_frame(frame_path)
            actual = probes.probe_frame(frame, self.templates)

            with self.subTest(frame=name, probe="shape"):
                self.assertEqual(actual["shape"], baseline["shape"])

            with self.subTest(frame=name, probe="mean"):
                self.assertLessEqual(abs(actual["mean"] - baseline["mean"]), MEAN_TOLERANCE)

            with self.subTest(frame=name, probe="regions"):
                for region, expected in baseline["regions"].items():
                    got = actual["regions"][region]
                    self.assertEqual(got["shape"], expected["shape"], f"{region} 裁剪尺寸变了")
                    for key in ("mean", "std"):
                        self.assertLessEqual(
                            abs(got[key] - expected[key]), REGION_TOLERANCE,
                            f"{region}.{key} 漂移过大：{expected[key]} -> {got[key]}",
                        )

            with self.subTest(frame=name, probe="icons"):
                for icon, expected in baseline["icons"].items():
                    got = actual["icons"].get(icon)
                    if not isinstance(expected, float):
                        continue
                    self.assertIsInstance(got, float, f"{icon} 匹配失败：{got}")
                    self.assertLessEqual(
                        abs(got - expected), ICON_TOLERANCE,
                        f"{icon} 匹配分漂移过大：{expected} -> {got}",
                    )

            with self.subTest(frame=name, probe="ocr"):
                # 只断言「基线里的文字没有丢」：OCR 的框选顺序/合并结果在
                # 不同线程调度下可能多出一行，但识别不出原本能识别的字才是真回归。
                got_lines = set(actual["ocr_lines"])
                missing = [line for line in baseline["ocr_lines"] if line not in got_lines]
                self.assertEqual(missing, [], f"{name} 丢失了基线里能识别的文字：{missing}")


if __name__ == "__main__":
    unittest.main()
