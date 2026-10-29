"""InteractionBGD 感知层公开 API 的行为基线（录制与回放共用）。

用途：证明「重构没有改变感知行为」。做法是把 InteractionBGD 的感知方法
（capture / get_screen_center / ocr_* / get_img_existence / get_text_existence /
appear / wait_until_stable）在一张固定帧上的**完整输出**录成 JSON 基线，
重构后再回放同一张帧，逐键比对。

为什么需要它：拆分 InteractionBGD（P3-2）要把方法体搬到别的类里，
一旦有哪一行的语义被改动（裁剪顺序、通道处理、匹配模式、日志分支），
单元测试里的断言未必覆盖到。而这条链路对同一张帧是完全确定的，
因此「输出逐键相同」是一个可以自动验证的强等价条件。

设计取舍：
- 截图用 FakeCapture 桩死，永远返回同一帧，不碰真实窗口，也不依赖游戏是否运行。
- InteractionNormal / PrintWindowCapture 也用桩替换（在 __init__ 内部 import，
  所以只要构造期间替换 sys.modules 即可），构造后立刻恢复，不污染其它测试。
- 图标模板从 ui_assets 按名字自动枚举，不硬编码清单。
- 浮点统一 round 到 3 位：既吸收 OpenCV 多线程归约的末位抖动，
  又足以暴露任何真实的行为变化（匹配率相差 0.001 就已经是不同阈值区间）。
"""
import contextlib
import io
import os
import sys
import types
from unittest import mock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 参与基线比对的图标模板上限（按名字排序取前 N 个，保证可复现）
MAX_ICONS = 12
# 探针用的 OCR 区域（1920x1080 逻辑坐标），落在任务追踪区上，里面确实有文字
PROBE_AREA = (30, 230, 420, 400)
# 一条画面里确实存在的文字，和一条必然不存在的文字
PRESENT_TEXT = "迷雾中的启程"
MISSING_TEXT = "这段文字一定不会出现在画面里XYZ"
FLOAT_DIGITS = 3


class FakeCapture:
    """替换 PrintWindowCapture：永远返回同一帧，不碰真实窗口。"""

    def __init__(self, frame):
        self.frame = frame
        self.resolution = (frame.shape[1], frame.shape[0])

    def capture(self, force=False):
        return self.frame.copy()


def _norm(value):
    """把感知结果转成可 JSON 序列化、且跨进程稳定的表示。"""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        return {"ndarray": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), FLOAT_DIGITS)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_norm(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _norm(item) for key, item in value.items()}
    # AnchorPosi 之类的坐标对象：只记数字，避免 repr 里的内存地址导致基线不稳定
    if all(hasattr(value, attr) for attr in ("x1", "y1", "x2", "y2")):
        return {"x1": int(value.x1), "y1": int(value.y1), "x2": int(value.x2), "y2": int(value.y2)}
    return repr(value)


def load_icon_objects() -> dict:
    """按名字枚举 UI 图标对象，返回 {name: ImgIcon}。"""
    from whimbox.ui import ui_assets
    from whimbox.ui.template.img_manager import ImgIcon

    icons = {}
    for name in sorted(vars(ui_assets)):
        obj = getattr(ui_assets, name)
        if not isinstance(obj, ImgIcon):
            continue
        try:
            image = obj.image
        except Exception:  # noqa: BLE001 - 单个图标加载失败跳过
            continue
        if image is None or getattr(image, "size", 0) == 0:
            continue
        icons[name] = obj
        if len(icons) >= MAX_ICONS:
            break
    return icons


def build_interaction(frame):
    """构造一个不接触真实窗口的 InteractionBGD 实例。

    InteractionBGD.__init__ 是在函数体内 import 依赖的，所以这里先在真实模块上
    把两个实现类替换成桩，构造完再撤掉。用 patch.object 而不是替换 sys.modules，
    因为 `import whimbox.interaction.interaction_normal` 还要求父包上存在同名属性。
    """
    import whimbox.interaction.capture as capture_module
    import whimbox.interaction.interaction_normal as normal_module
    from whimbox.interaction import interaction_core as core

    fake_capture = FakeCapture(frame)
    with mock.patch.object(normal_module, "InteractionNormal",
                           lambda hwnd_handler: mock.Mock(name="InteractionNormal")), \
            mock.patch.object(capture_module, "PrintWindowCapture",
                              lambda hwnd_handler: fake_capture):
        interaction = core.InteractionBGD(mock.Mock(name="hwnd_handler"))
    return interaction, fake_capture


def make_area(name: str, box):
    """构造一个只带坐标、不依赖资源图的 Area。

    Area.__init__ 会用调用者名字去查 assets 里的坐标图，探针不需要真实资源，
    所以直接跳过 __init__ 手动装配基类要用的字段（_position 非 None 就不会再读图）。
    """
    from whimbox.ui.template import posi_manager

    area = posi_manager.Area.__new__(posi_manager.Area)
    area.name = name
    area.print_log = 0
    area.anchor = box.anchor
    area.expand = False
    area._img_path = None
    area._position = box
    return area


def collect(frame) -> dict:
    """在一张固定帧上跑一遍感知层公开 API，返回 {探针名: 归一化结果}。"""
    from whimbox.common.utils.asset_utils import AnchorPosi
    from whimbox.ui.template import text_manager

    from whimbox.interaction import interaction_core as core

    interaction, _ = build_interaction(frame)
    area = make_area("probe_quest_area", AnchorPosi(*PROBE_AREA))
    icons = load_icon_objects()
    texts = [text_manager.TextTemplate(PRESENT_TEXT, area, name="probe_present_text"),
             text_manager.TextTemplate(MISSING_TEXT, area, name="probe_missing_text")]

    results = {}

    def run(label, func):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                results[label] = _norm(func())
        except Exception as exc:  # noqa: BLE001 - 异常本身也是被基线锁定的行为
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}

    run("capture.full_shape", lambda: interaction.capture().shape)
    run("capture.crop_shape", lambda: interaction.capture(AnchorPosi(0, 0, 100, 50)).shape)
    run("get_screen_center", interaction.get_screen_center)

    for name, icon in icons.items():
        run(f"icon.{name}.bool", lambda icon=icon: interaction.get_img_existence(icon))
        run(f"icon.{name}.rate",
            lambda icon=icon: interaction.get_img_existence(icon, ret_mode=core.IMG_RATE))
        run(f"icon.{name}.posi",
            lambda icon=icon: interaction.get_img_existence(icon, ret_mode=core.IMG_POSI))
        run(f"icon.{name}.appear", lambda icon=icon: interaction.appear(icon))

    for text in texts:
        run(f"text.{text.name}.bool", lambda text=text: interaction.get_text_existence(text))
        run(f"text.{text.name}.posi",
            lambda text=text: interaction.get_text_existence(text, ret_mode=core.IMG_POSI))
        run(f"text.{text.name}.appear", lambda text=text: interaction.appear(text))

    run("ocr.single_line", lambda: interaction.ocr_single_line(area, padding=50))
    run("ocr.multiple_lines", lambda: interaction.ocr_multiple_lines(area, padding=50))
    run("ocr.detect_posi", lambda: interaction.ocr_and_detect_posi(area, padding=50))
    run("wait_until_stable", lambda: interaction.wait_until_stable(threshold=0.9995, timeout=2))
    return results


def read_fixture_frame(fixture_path: str) -> np.ndarray:
    """读入 fixtures 里的 BGR JPEG，转成和真实截图一致的 4 通道 BGRA。"""
    buffer = np.fromfile(fixture_path, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"无法读取 golden 帧: {fixture_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
