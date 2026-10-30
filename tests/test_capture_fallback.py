"""截图回退路径的单元测试。

覆盖三件事：
1. PrintWindow 返回 0（失败）时必须回退到 BitBlt，而不是把全黑位图当成成功帧。
2. 即便拿到位图，全零帧也必须按取帧失败处理。
3. 回退的源必须是「屏幕 DC」并按客户区在屏幕上的位置偏移取源矩形。

第 3 条是一条真实缺陷的回归防线：最初的实现直接把窗口 DC 当 BitBlt 源，
而走翻转链（flip model）的全屏游戏——典型是虚幻引擎——窗口 DC 里没有可读像素，
拷出来的是 DWM 重定向表面的陈旧快照：每一帧都是同一幅旧画面、亮度恒定，
连全零帧检查都发现不了。修正后源改为屏幕 DC（与 PIL.ImageGrab 同一路径）。

这些用例全部用桩替换 win32 层，因此不依赖真实游戏窗口，可以在 CI 里跑。
"""
import contextlib
import unittest
from unittest import mock

import numpy as np

from whimbox.platform.windows import capture as capture_module
from whimbox.platform.windows.capture import WindowsCaptureManager, _is_blank_frame

# 桩里约定的几个句柄值，便于断言「用的是哪个 DC」
WINDOW_DC = 3
SCREEN_DC = 5
CLIENT_ORIGIN = (100, 200)


class _FakeBitmap:
    def __init__(self, payload: bytes, width: int, height: int) -> None:
        self._payload = payload
        self._width = width
        self._height = height

    def CreateCompatibleBitmap(self, _dc, _w, _h) -> None:  # noqa: N802 - win32 风格命名
        pass

    def GetInfo(self):  # noqa: N802 - win32 风格命名
        return {"bmWidth": self._width, "bmHeight": self._height}

    def GetBitmapBits(self, _selected):  # noqa: N802 - win32 风格命名
        return self._payload

    def GetHandle(self):  # noqa: N802 - win32 风格命名
        return 1


class _FakeCompatDC:
    def __init__(self) -> None:
        self.bitblt_calls = []

    def SelectObject(self, _bmp) -> None:  # noqa: N802 - win32 风格命名
        pass

    def GetSafeHdc(self):  # noqa: N802 - win32 风格命名
        return 2

    def BitBlt(self, dest, size, src, src_pos, rop):  # noqa: N802 - win32 风格命名
        # 记录源 DC 句柄，用于断言源是屏幕 DC 而不是窗口 DC
        self.bitblt_calls.append(
            {"dest": dest, "size": size, "src_hdc": getattr(src, "hdc", None),
             "src_pos": src_pos, "rop": rop}
        )

    def DeleteDC(self) -> None:  # noqa: N802 - win32 风格命名
        pass


class _FakeDCMem:
    def __init__(self, compat: _FakeCompatDC, hdc: int = 0) -> None:
        self._compat = compat
        self.hdc = hdc

    def CreateCompatibleDC(self):  # noqa: N802 - win32 风格命名
        return self._compat

    def DeleteDC(self) -> None:  # noqa: N802 - win32 风格命名
        pass


class CaptureFallbackTests(unittest.TestCase):
    def _patch_win32(self, payload: bytes, width: int, height: int, printwindow_ret: int, visible: bool = True,
                     minimized: bool = False):
        compat = _FakeCompatDC()
        bitmap = _FakeBitmap(payload, width, height)
        patches = [
            mock.patch.object(capture_module.win32gui, "GetClientRect", return_value=(0, 0, width, height)),
            mock.patch.object(capture_module.win32gui, "GetWindowDC", return_value=WINDOW_DC),
            mock.patch.object(capture_module.win32gui, "GetDC", return_value=SCREEN_DC),
            mock.patch.object(capture_module.win32gui, "ClientToScreen", return_value=CLIENT_ORIGIN),
            mock.patch.object(capture_module.win32gui, "ReleaseDC"),
            mock.patch.object(capture_module.win32gui, "DeleteObject"),
            mock.patch.object(capture_module.win32gui, "IsWindowVisible", return_value=visible),
            mock.patch.object(capture_module.win32gui, "IsIconic", return_value=minimized),
            mock.patch.object(capture_module.win32ui, "CreateDCFromHandle",
                              side_effect=lambda hdc: _FakeDCMem(compat, hdc)),
            mock.patch.object(capture_module.win32ui, "CreateBitmap", return_value=bitmap),
            mock.patch.object(capture_module.ctypes.windll.user32, "PrintWindow", return_value=printwindow_ret),
        ]
        return compat, patches

    @staticmethod
    def _run(patches, manager=None):
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            return (manager or WindowsCaptureManager()).capture_window(1234, None)

    def test_blank_frame_detection(self) -> None:
        self.assertTrue(_is_blank_frame(np.zeros((4, 4, 4), dtype=np.uint8)))
        non_blank = np.zeros((4, 4, 4), dtype=np.uint8)
        non_blank[2, 3, 1] = 1
        self.assertFalse(_is_blank_frame(non_blank))

    def test_falls_back_to_bitblt_when_printwindow_fails(self) -> None:
        """PrintWindow 返回 0 时必须改用 BitBlt，并返回真实内容而不是全黑帧。"""
        payload = np.full((4, 4, 4), 200, dtype=np.uint8).tobytes()
        compat, patches = self._patch_win32(payload, 4, 4, printwindow_ret=0)
        frame = self._run(patches)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (4, 4, 4))
        self.assertEqual(int(frame.max()), 200)
        self.assertEqual(len(compat.bitblt_calls), 1, "PrintWindow 失败后应调用一次 BitBlt")

    def test_fallback_blits_from_screen_dc_with_client_offset(self) -> None:
        """回退必须从屏幕 DC 取像素，源矩形按客户区左上角在屏幕上的位置偏移。

        用窗口 DC 当源会拿到 DWM 重定向表面的陈旧快照（每帧相同、亮度恒定），
        全零帧检查发现不了，下游感知会在「看起来合法」的旧画面上静默空转。
        """
        payload = np.full((4, 4, 4), 77, dtype=np.uint8).tobytes()
        compat, patches = self._patch_win32(payload, 8, 6, printwindow_ret=0)
        self._run(patches)
        self.assertEqual(len(compat.bitblt_calls), 1)
        call = compat.bitblt_calls[0]
        self.assertEqual(call["src_hdc"], SCREEN_DC, "BitBlt 的源必须是屏幕 DC")
        self.assertEqual(call["src_pos"], CLIENT_ORIGIN, "源坐标必须是客户区在屏幕上的偏移")
        self.assertEqual(call["size"], (8, 6), "目标尺寸必须是客户区尺寸")
        self.assertEqual(call["dest"], (0, 0))

    def test_no_fallback_when_window_minimized(self) -> None:
        """窗口最小化时 BitBlt 同样拿到黑图，直接按取帧失败返回 None。"""
        payload = np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
        compat, patches = self._patch_win32(payload, 4, 4, printwindow_ret=0, minimized=True)
        frame = self._run(patches)
        self.assertIsNone(frame)
        self.assertEqual(compat.bitblt_calls, [], "最小化时不应尝试 BitBlt")

    def test_all_black_bitmap_is_a_failure_even_if_printwindow_reports_success(self) -> None:
        """有的驱动会「返回成功但画出全黑」，这种情况也必须判失败。"""
        payload = np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
        _compat, patches = self._patch_win32(payload, 4, 4, printwindow_ret=1)
        self.assertIsNone(self._run(patches))

    def test_invalid_handle_returns_none(self) -> None:
        self.assertIsNone(WindowsCaptureManager().capture_window(0, None))


if __name__ == "__main__":
    unittest.main()
