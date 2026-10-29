"""录制 InteractionBGD 感知层公开 API 的行为基线。

前置：已有感知层 golden fixture 帧（tests/fixtures/perception/frames/）。
不需要游戏正在运行——回放的都是落盘的帧。

用法（在仓库根目录）：

    python tests/tools/record_perception_api_baseline.py

产物：tests/fixtures/perception/api_baseline.json

什么时候需要重录：只有「感知行为本来就该变」时才重录（例如换了 OCR 模型、
调整了某个图标的阈值）。重构不应该需要重录——重录就等于把回归网擦掉了。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(TESTS_DIR)
for path in (REPO_ROOT, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import perception_api_baseline as baseline  # noqa: E402

FIXTURE_DIR = os.path.join(TESTS_DIR, "fixtures", "perception")
FRAMES_DIR = os.path.join(FIXTURE_DIR, "frames")
OUTPUT_PATH = os.path.join(FIXTURE_DIR, "api_baseline.json")


def main() -> None:
    frames = sorted(
        name for name in os.listdir(FRAMES_DIR) if name.lower().endswith(".jpg")
    ) if os.path.isdir(FRAMES_DIR) else []
    if not frames:
        raise SystemExit("缺少感知层 golden 帧，请先运行 tests/tools/record_perception_golden.py")

    name = frames[0]
    frame = baseline.read_fixture_frame(os.path.join(FRAMES_DIR, name))
    print(f"使用帧 {name}: shape={frame.shape}")

    probes = baseline.collect(frame)
    payload = {
        "note": "InteractionBGD 感知层公开 API 的输出基线，由 record_perception_api_baseline.py 生成",
        "frame": name,
        "probe_count": len(probes),
        "probes": probes,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)

    errors = {key: value for key, value in probes.items() if isinstance(value, dict) and "error" in value}
    texts = {key: value for key, value in probes.items() if key.startswith("text.") and ".bool" in key}
    print(f"探针 {len(probes)} 项，其中异常 {len(errors)} 项 -> {OUTPUT_PATH}")
    for key, value in errors.items():
        print(f"  异常探针 {key}: {value['error']}")
    for key, value in texts.items():
        print(f"  文字探针 {key}: {value}")


if __name__ == "__main__":
    main()
