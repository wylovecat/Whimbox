"""Windows window management implementation using win32 APIs."""
from __future__ import annotations

import ctypes
import psutil
import time
from typing import Any, Optional

import win32api
import win32con
import win32gui
import win32process

from whimbox.core.interfaces import WindowManager
from whimbox.common.logger import logger


class WindowsWindowManager(WindowManager):
    """Implements WindowManager using win32 APIs."""

    def find_process(self, process_name: Optional[str], pid: Optional[int]) -> Optional[int]:
        """Return an HWND for the given process, or None."""
        if pid is not None:
            return self._get_hwnd_for_pid(pid)
        if process_name is not None:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    return self._get_hwnd_for_pid(proc.info['pid'])
        return None

    def _get_hwnd_for_pid(self, pid: int) -> int:
        hwnds: list[int] = []

        def callback(hwnd: int, _: Any) -> bool:
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid and win32gui.IsWindowVisible(hwnd) and win32gui.GetParent(hwnd) == 0:
                hwnds.append(hwnd)
            return True

        win32gui.EnumWindows(callback, None)
        return hwnds[0] if hwnds else 0

    def get_pid(self, native_handle: Any, process_name: Optional[str]) -> Optional[int]:
        hwnd: int = native_handle or 0
        if hwnd and win32gui.IsWindow(hwnd):
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid:
                    return pid
            except Exception as exc:
                logger.warning(f'通过窗口句柄获取进程ID失败: {exc}')
        if process_name:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == process_name:
                    return proc.info['pid']
        return None

    def is_foreground(self, native_handle: Any, pid: Optional[int]) -> bool:
        return bool(native_handle) and win32gui.GetForegroundWindow() == native_handle

    def is_minimized(self, native_handle: Any) -> bool:
        return bool(native_handle) and win32gui.IsIconic(native_handle)

    # 前台切换是异步的：SetForegroundWindow 返回不代表窗口已经变成前台，
    # 立刻校验会误判失败（历史上表现为偶发「游戏窗口前置失败」）。每次尝试后
    # 轮询等待一小段时间再判定。
    _FOREGROUND_WAIT_SECONDS = 0.5

    def _wait_foreground(self, hwnd: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if win32gui.GetForegroundWindow() == hwnd:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.03)

    def _attach_and_set_foreground(self, hwnd: int) -> None:
        """线程输入挂接 + SetForegroundWindow。对同一输入队列的窗口有效。"""
        fg_hwnd = win32gui.GetForegroundWindow()
        current_tid = win32api.GetCurrentThreadId()
        target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        fg_tid = 0
        if fg_hwnd:
            fg_tid, _ = win32process.GetWindowThreadProcessId(fg_hwnd)

        attached_fg = False
        attached_target = False

        def _safe_attach(src: int, dst: int, attach: bool, label: str) -> bool:
            if not src or not dst or src == dst:
                return False
            try:
                win32process.AttachThreadInput(src, dst, attach)
                return True
            except Exception as exc:
                logger.warning(f'AttachThreadInput {label} failed: {exc}')
                return False

        try:
            attached_fg = _safe_attach(current_tid, fg_tid, True, 'foreground')
            attached_target = _safe_attach(current_tid, target_tid, True, 'target')
            win32gui.SetForegroundWindow(hwnd)
        finally:
            if attached_target:
                _safe_attach(current_tid, target_tid, False, 'target-detach')
            if attached_fg:
                _safe_attach(current_tid, fg_tid, False, 'foreground-detach')

    def _alt_tap_and_set_foreground(self, hwnd: int) -> None:
        """先模拟一次 Alt 按键再前置。

        Windows 只允许「当前前台进程」或「刚刚接收过输入」的进程抢占前台。
        模拟一次按键会让本进程满足后一个条件，这是绕过前台锁定的常用手法。
        Alt 单击不产生可见副作用（不打开菜单、不输入字符）。
        """
        ctypes.windll.user32.keybd_event(win32con.VK_MENU, 0, 0, 0)
        ctypes.windll.user32.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.05)
        win32gui.SetForegroundWindow(hwnd)

    def _switch_to_this_window(self, hwnd: int) -> None:
        """SwitchToThisWindow：任务栏切窗用的未公开接口，不受前台锁定限制。"""
        ctypes.windll.user32.SwitchToThisWindow(hwnd, True)

    def _bring_to_top(self, hwnd: int) -> None:
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetWindowPos(
            hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW,
        )

    def set_foreground(self, native_handle: Any, pid: Optional[int], process_name: Optional[str]) -> None:
        hwnd: int = native_handle or 0
        try:
            if not self.is_alive(hwnd):
                raise Exception('游戏窗口不存在')
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(0.1)
            if self.is_foreground(hwnd, pid):
                return
            # 逐级降级：单条路径在不同窗口状态/前台锁定时会失效，而前置失败会直接
            # 中断整个任务，所以这里宁可多试几种手法，也不要把失败留给上层。
            attempts = (
                ('attach', self._attach_and_set_foreground),
                ('alt-tap', self._alt_tap_and_set_foreground),
                ('switch-to-window', self._switch_to_this_window),
                ('bring-to-top', self._bring_to_top),
            )
            for name, attempt in attempts:
                try:
                    attempt(hwnd)
                except Exception as exc:
                    logger.debug(f'前置尝试 {name} 失败: {exc}')
                    continue
                if self._wait_foreground(hwnd, self._FOREGROUND_WAIT_SECONDS):
                    if name != 'attach':
                        logger.debug(f'游戏窗口前置成功（{name}）')
                    return
                logger.debug(f'前置尝试 {name} 未生效，继续降级')
            raise Exception('无法将游戏窗口前置')
        except Exception as exc:
            logger.error(exc)
            raise Exception('游戏窗口前置失败')

    def is_alive(self, native_handle: Any) -> bool:
        return bool(native_handle) and win32gui.IsWindow(native_handle)

    def close(self, native_handle: Any) -> None:
        if native_handle and self.is_alive(native_handle):
            try:
                win32gui.PostMessage(native_handle, win32con.WM_CLOSE, 0, 0)
            except Exception as exc:
                logger.error(exc)

    def get_window_size(self, native_handle: Any, pid: Optional[int]) -> tuple[int, int]:
        hwnd: int = native_handle or 0
        if not hwnd:
            return 0, 0
        try:
            _, _, width, height = win32gui.GetClientRect(hwnd)
            return width, height
        except Exception:
            return 0, 0

    def get_window_rect(self, native_handle: Any, pid: Optional[int]) -> tuple[int, int, int, int]:
        hwnd: int = native_handle or 0
        if not hwnd:
            return 0, 0, 0, 0
        try:
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            return left, top, width, height
        except Exception:
            return 0, 0, 0, 0

    def get_window_scale_factor(self, native_handle: Any, pid: Optional[int]) -> float:
        hwnd: int = native_handle or 0
        if not hwnd:
            return 1.0
        try:
            get_dpi_for_window = ctypes.windll.user32.GetDpiForWindow
            get_dpi_for_window.argtypes = [ctypes.c_void_p]
            get_dpi_for_window.restype = ctypes.c_uint
            dpi = int(get_dpi_for_window(hwnd))
            return dpi / 96.0 if dpi > 0 else 1.0
        except Exception:
            return 1.0

    def client_to_screen(self, native_handle: Any, x: int, y: int) -> tuple[int, int]:
        return win32gui.ClientToScreen(native_handle, (x, y))
