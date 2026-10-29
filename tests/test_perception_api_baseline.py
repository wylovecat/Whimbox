"""InteractionBGD 感知层公开 API 的行为基线回放测试。

断言感知层在固定帧上的输出与基线逐键相同。这条测试是「拆分 InteractionBGD、
重构感知链路」的行为等价证明：输出不变 = 行为不变。

基线由 tests/tools/record_perception_api_baseline.py 录制，依赖真实游戏帧，
因此 fixtures 不存在时跳过。
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import perception_api_baseline as baseline  # noqa: E402

FIXTURE_DIR = os.path.join(HERE, "fixtures", "perception")
FRAMES_DIR = os.path.join(FIXTURE_DIR, "frames")
BASELINE_PATH = os.path.join(FIXTURE_DIR, "api_baseline.json")

# 允许的最大差异条数，超过就整体失败（避免一次重构改动过大却只报一条）
MAX_REPORTED_DIFFS = 15


@unittest.skipUnless(os.path.exists(BASELINE_PATH), "缺少感知层 API 基线")
class PerceptionPublicApiBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
            cls.baseline = json.load(handle)
        frame_path = os.path.join(FRAMES_DIR, cls.baseline["frame"])
        cls.frame = baseline.read_fixture_frame(frame_path)
        cls.actual = baseline.collect(cls.frame)

    def test_probe_set_is_unchanged(self):
        """探针集合本身要一致：方法增删属于对外契约变化，必须显式重录基线。"""
        expected_keys = set(self.baseline["probes"])
        actual_keys = set(self.actual)
        self.assertEqual(
            sorted(expected_keys - actual_keys), [],
            "基线里的探针在实现里消失了，说明公开 API 有删改",
        )
        self.assertEqual(
            sorted(actual_keys - expected_keys), [],
            "出现了基线里没有的探针，请在确认这是有意新增后重录基线",
        )

    def test_outputs_match_baseline(self):
        """逐键比对输出。"""
        expected = self.baseline["probes"]
        diffs = []
        for key in sorted(expected):
            if key not in self.actual:
                continue
            if expected[key] != self.actual[key]:
                diffs.append(f"{key}\n  基线: {expected[key]!r}\n  实际: {self.actual[key]!r}")
        if diffs:
            detail = "\n".join(diffs[:MAX_REPORTED_DIFFS])
            more = f"\n（另有 {len(diffs) - MAX_REPORTED_DIFFS} 处未列出）" if len(diffs) > MAX_REPORTED_DIFFS else ""
            self.fail(f"感知层输出与基线不一致 {len(diffs)} 处:\n{detail}{more}")

    def test_baseline_is_not_degenerate(self):
        """基线不能是「全黑帧 + 全 False」这种退化的记录。

        截图链路曾经返回全黑帧却报 fresh=True，如果基线恰在那时录制，
        整套回归网就会对着空气比对——所以这里显式守住「基线有内容」。
        """
        probes = self.baseline["probes"]
        detected = [key for key, value in probes.items() if value is True]
        self.assertGreater(len(detected), 0, "基线里没有任何 True 结果，无法证明感知有效")
        present = probes.get("text.probe_present_text.bool")
        self.assertTrue(present, "基线里连已知存在的文字都没识别出来，帧或探针有问题")


if __name__ == "__main__":
    unittest.main()
