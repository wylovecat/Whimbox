"""截图回退路径的单元测试。

覆盖两件事：
1. PrintWindow 返回 0（失败）时必须回退到 BitBlt，而不是把全黑位图当成成功帧。
2. 即便拿到位图，全零帧也必须按取帧失败处理。

这些用例全部用桩替换 win32 层，因此不依赖真实游戏窗口，可以在 CI 里跑。
"""
import unittest
from unittest import mock

import numpy as np

from whimbox.platform.windows import capture as capture_module
from whimbox.platform.windows.capture import WindowsCaptureManager, _is_blank_frame


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
        self.bitblt_calls.append((dest, size, src_pos, rop))

    def DeleteDC(self) -> None:  # noqa: N802 - win32 风格命名
        pass


class _FakeDCMem:
    def __init__(self, compat: _FakeCompatDC) -> None:
        self._compat = compat

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
            mock.patch.object(capture_module.win32gui, "GetWindowDC", return_value=3),
            mock.patch.object(capture_module.win32gui, "ReleaseDC"),
            mock.patch.object(capture_module.win32gui, "DeleteObject"),
            mock.patch.object(capture_module.win32gui, "IsWindowVisible", return_value=visible),
            mock.patch.object(capture_module.win32gui, "IsIconic", return_value=minimized),
            mock.patch.object(capture_module.win32ui, "CreateDCFromHandle", return_value=_FakeDCMem(compat)),
            mock.patch.object(capture_module.win32ui, "CreateBitmap", return_value=bitmap),
            mock.patch.object(capture_module.ctypes.windll.user32, "PrintWindow", return_value=printwindow_ret),
        ]
        return compat, patches

    def test_blank_frame_detection(self) -> None:
        self.assertTrue(_is_blank_frame(np.zeros((4, 4, 4), dtype=np.uint8)))
        non_blank = np.zeros((4, 4, 4), dtype=np.uint8)
        non_blank[2, 3, 1] = 1
        self.assertFalse(_is_blank_frame(non_blank))

    def test_falls_back_to_bitblt_when_printwindow_fails(self) -> None:
        """PrintWindow 返回 0 时必须改用 BitBlt，并返回真实内容而不是全黑帧。"""
        payload = np.full((4, 4, 4), 200, dtype=np.uint8).tobytes()
        compat, patches = self._patch_win32(payload, 4, 4, printwindow_ret=0)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            frame = WindowsCaptureManager().capture_window(1234, None)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (4, 4, 4))
        self.assertEqual(int(frame.max()), 200)
        self.assertEqual(len(compat.bitblt_calls), 1, "PrintWindow 失败后应调用一次 BitBlt")

    def test_no_fallback_when_window_minimized(self) -> None:
        """窗口最小化时 BitBlt 同样拿到黑图，直接按取帧失败返回 None。"""
        payload = np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
        compat, patches = self._patch_win32(payload, 4, 4, printwindow_ret=0, minimized=True)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            frame = WindowsCaptureManager().capture_window(1234, None)
        self.assertIsNone(frame)
        self.assertEqual(compat.bitblt_calls, [], "最小化时不应尝试 BitBlt")

    def test_all_black_bitmap_is_a_failure_even_if_printwindow_reports_success(self) -> None:
        """有的驱动会「返回成功但画出全黑」，这种情况也必须判失败。"""
        payload = np.zeros((4, 4, 4), dtype=np.uint8).tobytes()
        _compat, patches = self._patch_win32(payload, 4, 4, printwindow_ret=1)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            frame = WindowsCaptureManager().capture_window(1234, None)
        self.assertIsNone(frame)

    def test_invalid_handle_returns_none(self) -> None:
        self.assertIsNone(WindowsCaptureManager().capture_window(0, None))


if __name__ == "__main__":
    unittest.main()
