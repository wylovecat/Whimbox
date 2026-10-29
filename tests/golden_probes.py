"""感知层 golden fixture 的探针定义。

录制脚本（tests/tools/record_perception_golden.py）和回放测试
（tests/test_perception_golden.py）共用这里的逻辑，保证「录什么就比什么」，
避免两边各写一份导致断言漂移。

设计取舍：
- 只依赖「截图 -> 纯感知函数」这条链路上不产生副作用的环节（OCR、模板匹配、
  区域统计），不碰任何鼠标键盘动作，因此回放本身绝对不会影响真实游戏。
- 模板图标从 whimbox.ui.ui_assets 里按名字自动枚举，不硬编码清单；这样即使
  以后新增图标，fixtures 也能重新录制而不必改探针代码。
"""
import os

import cv2
import numpy as np

# 探针用的固定屏幕区域（1920x1080 逻辑坐标）。选的是 HUD 上位置稳定的几块，
# 用区域亮度/对比度做结构指纹：感知链路（裁剪、缩放、色彩通道）一旦被改坏，
# 这些数字会立刻变化。
REGIONS = {
    "minimap": (20, 10, 260, 260),
    "quest_tracker": (30, 230, 420, 400),
    "top_right_menu": (1050, 10, 1900, 120),
    "character": (760, 300, 1160, 760),
    "bottom_bar": (300, 960, 1600, 1075),
    "left_skills": (0, 300, 120, 700),
}

# 参与指纹比对的图标模板上限（按名字排序取前 N 个，保证多次录制结果可比）
MAX_ICON_TEMPLATES = 12


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """统一成 3 通道 BGR，供 cv2.matchTemplate 使用。"""
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_icon_templates() -> dict:
    """按名字枚举 UI 图标模板，返回 {name: image}。

    只取能成功加载的，加载失败的直接跳过（有的图标是延迟加载/依赖运行时状态）。
    """
    from whimbox.ui import ui_assets
    from whimbox.ui.template.img_manager import ImgIcon

    templates = {}
    for name in sorted(vars(ui_assets)):
        obj = getattr(ui_assets, name)
        if not isinstance(obj, ImgIcon):
            continue
        try:
            image = _to_bgr(obj.image)
        except Exception:  # noqa: BLE001 - 单个图标加载失败不影响整体指纹
            continue
        if image is None or image.size == 0:
            continue
        templates[name] = image
        if len(templates) >= MAX_ICON_TEMPLATES:
            break
    return templates


def region_stats(frame: np.ndarray) -> dict:
    """每个固定区域的亮度均值与标准差。"""
    stats = {}
    height, width = frame.shape[:2]
    for name, (x1, y1, x2, y2) in REGIONS.items():
        # 坐标按逻辑 1920x1080 定义，实际帧分辨率不同时按比例换算
        sx, sy = width / 1920.0, height / 1080.0
        crop = frame[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
        if crop.size == 0:
            stats[name] = {"mean": None, "std": None, "shape": list(crop.shape)}
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        stats[name] = {
            "mean": round(float(gray.mean()), 3),
            "std": round(float(gray.std()), 3),
            "shape": list(crop.shape),
        }
    return stats


def icon_scores(frame: np.ndarray, templates: dict) -> dict:
    """每个图标模板在整帧上的最佳匹配分（TM_CCORR_NORMED）。

    这里刻意用「整帧搜索」而不是「按 cap_posi 裁剪后匹配」：cap_posi 是每个图标
    各自的元数据，重建它需要一整套 Area/坐标注册表；而整帧搜索同样会经过
    crop/通道/匹配这一整套原语，对感知链路的回归足够敏感，且不依赖注册表状态。
    """
    from whimbox.common.utils.img_utils import similar_img

    scores = {}
    frame_bgr = _to_bgr(frame)
    for name, template in templates.items():
        # 模板比图还大时 matchTemplate 会直接抛错，跳过
        if template.shape[0] > frame_bgr.shape[0] or template.shape[1] > frame_bgr.shape[1]:
            continue
        try:
            scores[name] = round(float(similar_img(frame_bgr, template)), 4)
        except Exception as exc:  # noqa: BLE001
            scores[name] = f"error: {type(exc).__name__}: {exc}"
    return scores


def ocr_lines(frame: np.ndarray) -> list:
    """整帧 OCR 结果（多行模式）。"""
    from whimbox.api.ocr_rapid import RapidOcr

    try:
        return [str(x) for x in RapidOcr().get_all_texts(_to_bgr(frame), mode=0)]
    except Exception as exc:  # noqa: BLE001
        return [f"error: {type(exc).__name__}: {exc}"]


def probe_frame(frame: np.ndarray, templates: dict = None) -> dict:
    """对一帧跑完整探针，返回可 JSON 序列化的结果。"""
    templates = load_icon_templates() if templates is None else templates
    return {
        "shape": list(frame.shape),
        "mean": round(float(frame.mean()), 3),
        "regions": region_stats(frame),
        "icons": icon_scores(frame, templates),
        "ocr_lines": ocr_lines(frame),
    }


def read_frame(path: str) -> np.ndarray:
    """读取 fixture 帧。用 cv2.imdecode 以兼容 Windows 上的非 ASCII 路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_frame(path: str, frame: np.ndarray, quality: int = 95) -> None:
    """写 fixture 帧（JPEG 以控制仓库体积）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"failed to encode frame: {path}")
    buf.tofile(path)
