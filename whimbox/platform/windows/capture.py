"""Windows screen-capture implementation using PrintWindow / win32."""
from __future__ import annotations

import ctypes
from typing import Any, Optional

import numpy as np
import win32gui
import win32ui

from whimbox.common.logger import logger
from whimbox.core.interfaces import CaptureManager

# PrintWindow 标志位：PW_CLIENTONLY(1) | PW_RENDERFULLCONTENT(2)
_PRINTWINDOW_FLAGS = 0x00000003
# BitBlt 光栅操作：直接拷贝源像素
_SRCCOPY = 0x00CC0020
# 回退路径的告警节流：30fps 下每 _FALLBACK_WARN_INTERVAL 次回退告警一条，避免刷爆日志
_FALLBACK_WARN_INTERVAL = 150


def _is_blank_frame(img: np.ndarray) -> bool:
    """判断整帧是否全零。

    PrintWindow 对使用翻转链（flip model）/ 硬件覆盖层的游戏会「成功返回但画出全黑」。
    全零帧对下游感知来说等价于没有画面（OCR、模板匹配、地图遮罩都会空转），
    所以必须当作取帧失败上报，而不是伪装成一张合法的新帧。
    用一个 any() 短路判断：真实帧几乎立刻返回 True，只有全黑帧才需要扫完整帧。
    """
    return not img.any()


class WindowsCaptureManager(CaptureManager):
    """Captures a window using the PrintWindow win32 API, with a BitBlt fallback."""

    def __init__(self) -> None:
        # 回退次数统计，仅用于日志节流与现场排查
        self._fallback_calls = 0
        self._fallback_warned = 0

    def capture_window(self, native_handle: Any, pid: Optional[int]) -> Optional[np.ndarray]:
        hwnd: int = native_handle or 0
        if not hwnd:
            return None
        hdc_window = 0
        hdc_mem = None
        hdc_compat = None
        bmp = None
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width = right - left
            height = bottom - top
            if width <= 0 or height <= 0:
                return None

            hdc_window = win32gui.GetWindowDC(hwnd)
            hdc_mem = win32ui.CreateDCFromHandle(hdc_window)
            hdc_compat = hdc_mem.CreateCompatibleDC()

            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(hdc_mem, width, height)
            hdc_compat.SelectObject(bmp)

            # PrintWindow 会返回 0 表示失败。这个返回值必须检查：某些游戏
            # （典型如虚幻引擎的全屏/无边框窗口，走翻转链交换链）会返回 0 并在
            # 位图上留下全黑内容。旧代码丢弃返回值，于是全黑帧被当成正常帧一路
            # 送到 OCR / 模板匹配，整条感知链在纯黑图上静默空转。
            if not ctypes.windll.user32.PrintWindow(hwnd, hdc_compat.GetSafeHdc(), _PRINTWINDOW_FLAGS):
                if not self._bitblt_fallback(hwnd, hdc_compat, hdc_mem, width, height):
                    return None

            bmpinfo = bmp.GetInfo()
            bmpstr = bmp.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)

            # 兜底：即便 PrintWindow 报告成功，也可能画出全黑（有的驱动会这样）。
            if _is_blank_frame(img):
                self._fallback_calls += 1
                if self._fallback_calls - self._fallback_warned >= _FALLBACK_WARN_INTERVAL:
                    self._fallback_warned = self._fallback_calls
                    logger.warning(
                        "截图返回全黑帧，已按取帧失败处理"
                        f"（累计 {self._fallback_calls} 次，hwnd={hwnd}）。"
                        "请确认游戏窗口未最小化、未被完全遮挡，且未使用独占全屏模式。"
                    )
                return None

            return img
        except Exception:
            logger.opt(exception=True).debug("PrintWindow capture failed")
            return None
        finally:
            # 句柄释放必须无条件执行：本函数运行在 30fps 的截图高频路径上，
            # 异常路径漏放一次就会累积一个窗口 DC / 位图，很快耗尽 GDI 句柄配额。
            # 四个释放各自 try/except，避免其中一个失败时掩盖其余释放。
            if bmp is not None:
                try:
                    win32gui.DeleteObject(bmp.GetHandle())
                except Exception:
                    logger.opt(exception=True).debug("failed to delete bitmap object")
            if hdc_compat is not None:
                try:
                    hdc_compat.DeleteDC()
                except Exception:
                    logger.opt(exception=True).debug("failed to delete compatible DC")
            if hdc_mem is not None:
                try:
                    hdc_mem.DeleteDC()
                except Exception:
                    logger.opt(exception=True).debug("failed to delete memory DC")
            if hdc_window:
                try:
                    win32gui.ReleaseDC(hwnd, hdc_window)
                except Exception:
                    logger.opt(exception=True).debug("failed to release window DC")

    def _bitblt_fallback(self, hwnd: int, hdc_compat: Any, hdc_mem: Any, width: int, height: int) -> bool:
        """PrintWindow 失败时，改用窗口 DC 的 BitBlt 直接拷贝屏幕像素。

        BitBlt 的限制决定了它只能作为回退：它读到的是「屏幕上真正显示的内容」，
        因此窗口必须可见且未被遮挡；窗口最小化时拿到的同样是黑图。
        游戏窗口被其它窗口盖住时会抓到覆盖者，这是已知代价——用「可见 + 未最小化」
        两个条件把误用范围压到最小，并且依然靠 _is_blank_frame 兜底。
        """
        try:
            if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
                return False
            hdc_compat.BitBlt((0, 0), (width, height), hdc_mem, (0, 0), _SRCCOPY)
        except Exception:
            logger.opt(exception=True).debug("BitBlt fallback failed")
            return False

        self._fallback_calls += 1
        if self._fallback_calls - self._fallback_warned >= _FALLBACK_WARN_INTERVAL:
            self._fallback_warned = self._fallback_calls
            logger.warning(
                "PrintWindow 取帧失败，已回退到 BitBlt"
                f"（累计 {self._fallback_calls} 次，hwnd={hwnd}）。"
                "回退期间若游戏窗口被其它窗口遮挡，截图会抓到遮挡者。"
            )
        return True
