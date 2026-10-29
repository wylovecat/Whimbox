"""截图帧率上限（WHIMBOX_CAPTURE_MAX_FPS）的解析逻辑测试。

背景：截图模块每秒最多抓 30 帧，每帧把 1920x1080x4（约 8MB）整帧拷贝一次，
30fps 下约 240MB/s 的内存带宽开销，对已被游戏吃满的机器是可观的浪费。
这里把「帧率上限」做成可配置项，默认行为保持不变（30fps）。

注意：本模块刻意只 import ``whimbox.common.capture_limits``（零第三方依赖），
不 import ``whimbox.interaction.capture``——后者在模块级 import 了 cv2 / numpy /
loguru，缺少这些可选依赖的环境下会让整个测试模块无法收集。对 capture.py 的接线
则用源码级断言覆盖。
"""
import re
from pathlib import Path

import pytest

from whimbox.common.capture_limits import (
    CAPTURE_MAX_FPS_ENV,
    DEFAULT_MAX_FPS,
    MAX_MAX_FPS,
    MIN_MAX_FPS,
    resolve_max_fps,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_PY = REPO_ROOT / "whimbox" / "interaction" / "capture.py"


# ---------------------------------------------------------------- 解析逻辑

def test_default_when_unset():
    assert resolve_max_fps() == DEFAULT_MAX_FPS == 30


def test_default_argument_is_respected():
    assert resolve_max_fps(default=10) == 10


def test_env_override_is_honoured():
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: "15"}) == 15


def test_env_override_accepts_float_string():
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: "20.7"}) == 20


def test_env_override_strips_whitespace():
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: " 12 "}) == 12


@pytest.mark.parametrize("raw", ["0", "-5", "-0.5"])
def test_too_low_is_clamped(raw):
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: raw}) == MIN_MAX_FPS


@pytest.mark.parametrize("raw", ["61", "144", "9999"])
def test_too_high_is_clamped(raw):
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: raw}) == MAX_MAX_FPS


@pytest.mark.parametrize("raw", ["abc", "fast", "12fps"])
def test_invalid_value_falls_back_to_default(raw):
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: raw}) == DEFAULT_MAX_FPS


@pytest.mark.parametrize("raw", ["", "   "])
def test_blank_value_falls_back_to_default(raw):
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: raw}) == DEFAULT_MAX_FPS


def test_boundaries_are_inclusive():
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: str(MIN_MAX_FPS)}) == MIN_MAX_FPS
    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: str(MAX_MAX_FPS)}) == MAX_MAX_FPS


def test_warn_callback_receives_messages():
    messages = []
    resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: "abc"}, warn=messages.append)
    resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: "999"}, warn=messages.append)
    assert len(messages) == 2
    assert all(CAPTURE_MAX_FPS_ENV in m for m in messages)


def test_warn_callback_failure_is_swallowed():
    def boom(_message):
        raise RuntimeError("logger exploded")

    assert resolve_max_fps(environ={CAPTURE_MAX_FPS_ENV: "abc"}, warn=boom) == DEFAULT_MAX_FPS


def test_real_os_environ_is_used(monkeypatch):
    monkeypatch.setenv(CAPTURE_MAX_FPS_ENV, "18")
    assert resolve_max_fps() == 18
    monkeypatch.delenv(CAPTURE_MAX_FPS_ENV)
    assert resolve_max_fps() == DEFAULT_MAX_FPS


# ------------------------------------------------- capture.py 的接线（源码级）

def test_capture_module_wires_resolver():
    src = CAPTURE_PY.read_text(encoding="utf-8")
    assert "resolve_max_fps" in src, "capture.py 应通过 resolve_max_fps 读取帧率上限"
    assert "self.max_fps = resolve_max_fps" in src, "基类应用 resolver 初始化 max_fps"


def test_capture_module_has_no_hardcoded_max_fps():
    """防止以后有人又写死 self.max_fps = 30 把环境变量覆盖掉。"""
    src = CAPTURE_PY.read_text(encoding="utf-8")
    # 只抓真正的赋值（``self.max_fps = ...``），排除 ``>= / <=`` 之类的比较
    assignment = re.compile(r"self\.max_fps\s*=(?!=)")
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        if assignment.search(code):
            assert "resolve_max_fps" in code, f"硬编码帧率上限: {line.strip()}"
