"""录制感知层 golden fixture（需要游戏正在运行）。

用法（游戏开着、停在想固化的界面上时运行一次）：

    python tests/tools/record_perception_golden.py

产物：
    tests/fixtures/perception/frames/frame_XX.jpg   采样帧
    tests/fixtures/perception/baseline.json         每帧的感知输出基线

基线是用「落盘后重新读出的 JPEG」算的，不是用内存里那帧算的。这样 JPEG 的
有损压缩已经被吸收进基线本身，回放时不会因为压缩误差而误报。
"""
import json
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
sys.path.insert(0, TESTS_DIR)

import golden_probes as probes  # noqa: E402

FIXTURE_DIR = os.path.join(TESTS_DIR, "fixtures", "perception")
FRAMES_DIR = os.path.join(FIXTURE_DIR, "frames")
BASELINE_PATH = os.path.join(FIXTURE_DIR, "baseline.json")

FRAME_COUNT = 4
FRAME_INTERVAL = 0.6
# 画面静止时（暂停、站立不动）连续采样会拿到完全相同的帧，只保留不同的画面，
# 避免把同一张图存多份。判定方式用像素内容而不是文件哈希——JPEG 编码是确定的，
# 内容相同必然哈希相同，用像素比较更直观。
FRAME_DIFF_THRESHOLD = 0.5


def _is_duplicate(candidate, previous) -> bool:
    if previous is None:
        return False
    if candidate.shape != previous.shape:
        return False
    return float(candidate.astype("float32").mean()) - float(previous.astype("float32").mean()) == 0.0 and \
        bool((candidate == previous).all())


def main() -> None:
    from whimbox.interaction.interaction_core import get_itt

    itt = get_itt()
    templates = probes.load_icon_templates()
    print(f"图标模板 {len(templates)} 个: {', '.join(list(templates)[:6])}...")

    os.makedirs(FRAMES_DIR, exist_ok=True)
    names = []
    previous = None
    for index in range(FRAME_COUNT):
        frame = itt.capture_obj.capture(force=True)
        if _is_duplicate(frame, previous):
            print(f"  第 {index} 帧与上一帧完全相同，跳过")
            time.sleep(FRAME_INTERVAL)
            continue
        previous = frame
        name = f"frame_{len(names):02d}.jpg"
        path = os.path.join(FRAMES_DIR, name)
        probes.write_frame(path, frame)
        stored = probes.read_frame(path)
        print(f"  {name}: shape={stored.shape} mean={float(frame.mean()):.2f} 落盘={os.path.getsize(path) // 1024}KB")
        names.append(name)
        time.sleep(FRAME_INTERVAL)

    if not names:
        raise SystemExit("没有采集到任何帧，请确认游戏窗口可见且未最小化")

    baselines = {}
    for name in names:
        frame = probes.read_frame(os.path.join(FRAMES_DIR, name))
        baselines[name] = probes.probe_frame(frame, templates)
        print(f"  采集基线 {name}: OCR {len(baselines[name]['ocr_lines'])} 行")

    payload = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "note": "由 tests/tools/record_perception_golden.py 自动生成，请勿手工编辑",
        "frames": names,
        "baselines": baselines,
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"基线已写入 {BASELINE_PATH}")


if __name__ == "__main__":
    main()
