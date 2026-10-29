"""截图帧率上限的解析逻辑（零依赖，便于单独测试）。

为什么要单独成模块：``capture.py`` 在模块级就 import 了 cv2 / numpy / loguru，
凡是只想验证「帧率上限怎么算」的场景（单元测试、外部工具读取默认值）都会被这些
重依赖挡在门外。这里把纯逻辑抽出来，只依赖标准库。

开销背景：``Capture.capture()`` 每秒最多抓 30 帧，每帧把 1920x1080x4（约 8MB）
的整帧拷贝一次，30fps 约 240MB/s 的内存带宽。对已经被游戏吃满 CPU / 内存带宽的
机器，把上限降到 15~20fps 能明显减负，而自动化识别通常并不需要 30fps。
"""
import os

# 默认帧率上限（保持与历史行为一致）
DEFAULT_MAX_FPS = 30
MIN_MAX_FPS = 1
MAX_MAX_FPS = 60

# 覆盖用的环境变量名
CAPTURE_MAX_FPS_ENV = "WHIMBOX_CAPTURE_MAX_FPS"


def resolve_max_fps(
    default: int = DEFAULT_MAX_FPS,
    environ=None,
    warn=None,
) -> int:
    """解析截图帧率上限。

    Args:
        default: 未设置环境变量时使用的值。
        environ: 环境变量映射，默认 ``os.environ``（便于测试注入）。
        warn: 可选的告警回调 ``warn(message: str)``，非法/越界时调用。

    Returns:
        落在 [MIN_MAX_FPS, MAX_MAX_FPS] 区间内的整数帧率上限。

    规则：
        - 未设置或空白 -> default
        - 非法数字 -> default（并告警）
        - 低于 MIN -> MIN（并告警）；高于 MAX -> MAX（并告警）
        - 允许 "20.7" 这类小数字符串，按整数截断
    """
    env = os.environ if environ is None else environ
    raw = str(env.get(CAPTURE_MAX_FPS_ENV, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        _warn(warn, f"{CAPTURE_MAX_FPS_ENV}={raw!r} 不是合法数字，回退 {default}fps")
        return default
    if value < MIN_MAX_FPS:
        _warn(warn, f"{CAPTURE_MAX_FPS_ENV}={value} 过低，按 {MIN_MAX_FPS}fps 处理")
        return MIN_MAX_FPS
    if value > MAX_MAX_FPS:
        _warn(warn, f"{CAPTURE_MAX_FPS_ENV}={value} 过高，按 {MAX_MAX_FPS}fps 处理")
        return MAX_MAX_FPS
    return value


def _warn(warn, message: str) -> None:
    if warn is None:
        return
    try:
        warn(message)
    except Exception:
        pass
