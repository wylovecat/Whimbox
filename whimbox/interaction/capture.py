"""Cross-platform window capture.

All platform-specific capture logic is delegated to the CaptureManager
obtained from the platform factory.
"""
import os
import threading
import time

import cv2
import numpy as np

from whimbox.common import timer_module
from whimbox.common.logger import logger
from whimbox.common.cvars import DEBUG_MODE
from whimbox.platform.factory import get_capture_manager


# 截图帧率上限的解析逻辑放在 whimbox/common/capture_limits.py（零依赖，便于单测），
# 这里只做一层薄封装，把告警接到项目统一的 loguru logger 上。
from whimbox.common.capture_limits import (
    CAPTURE_MAX_FPS_ENV,
    DEFAULT_MAX_FPS,
    MAX_MAX_FPS,
    MIN_MAX_FPS,
    resolve_max_fps as _resolve_max_fps,
)


def resolve_max_fps(default: int = DEFAULT_MAX_FPS) -> int:
    """解析截图帧率上限，支持用环境变量 WHIMBOX_CAPTURE_MAX_FPS 覆盖。

    默认 30fps；低配机器想让出更多 CPU/内存带宽给游戏时可设为 15~20。
    开销背景：每帧要拷贝约 8MB 的 1080p BGRA 整帧，30fps 约 240MB/s。
    """
    return _resolve_max_fps(default=default, warn=logger.warning)


# 断帧可见性：_capture() 取帧失败时会静默沿用上一帧缓存，调用方需要能区分
# "这是新帧"还是"这是几秒前的旧帧"，否则会基于过期画面做不可逆操作。
# 连续失败 FAIL_WARN_THRESHOLD 次后开始告警，之后每再连续失败
# FAIL_WARN_INTERVAL 次告警一次（30fps 下约 1 秒一条，避免把日志刷爆）。
FAIL_WARN_THRESHOLD = 3
FAIL_WARN_INTERVAL = 30
# 缓存帧被判为 stale 的最小帧龄（毫秒）。只有在"确实取帧失败"时才判 stale：
# 调用方本就可以低频调用 capture()，帧龄大并不等于异常，必须配合失败次数才有意义。
STALE_AGE_MS = 1000.0

FAIL_REASON_CAPTURE = "capture_failed"  # _get_capture() 返回 None（PrintWindow 失败 / 句柄失效）
FAIL_REASON_SHAPE = "shape_rejected"    # 取到了图但分辨率比例校验不通过


class Capture:
    def __init__(self, hwnd_handler):
        self.hwnd_handler = hwnd_handler
        self.capture_cache = np.zeros((1080, 1920, 4), dtype="uint8")
        self.resolution = None
        self.max_fps = resolve_max_fps()
        self.fps_timer = timer_module.Timer(diff_start_time=1)
        self.capture_cache_lock = threading.Lock()
        self.capture_times = 0
        self.cap_per_sec = timer_module.CyclicCounter(limit=3).start()
        self.last_cap_times = 0
        # 帧新鲜度状态，受 _status_lock 保护（与 capture_cache_lock 分开，
        # 避免查询状态时被一次慢速 PrintWindow 阻塞）。
        # 加锁顺序固定为 capture_cache_lock -> _status_lock，反向路径不存在，无死锁风险。
        self._status_lock = threading.Lock()
        self._frame_generation = 0        # 成功取帧的累计次数，用于判断两次读取之间有没有产生新帧
        self._consecutive_failures = 0    # 连续取帧失败次数，一旦成功即清零
        self._last_fail_reason = None     # 最近一次失败原因，取值见 FAIL_REASON_*
        self._last_success_mono = None    # 最近一次成功取帧的 time.monotonic()，用于算帧龄
        self._last_success_wall = None    # 同上，墙钟时间，仅用于日志展示

    def _cover_privacy(self, img: np.ndarray) -> np.ndarray:
        return img

    def _normalize_shape(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if self._check_shape(img):
            self.resolution = img.shape[:2]
            if img.shape[:2] == (1080, 1920) and img.shape[2] == 4:
                return img
            else:
                new_width = 1920
                new_height = int(1920 / self.resolution[1] * self.resolution[0])
                new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                if len(new_img.shape) == 3 and new_img.shape[2] == 3:
                    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2BGRA)
                return new_img
        else:
            self.resolution = None
            return None

    def _get_capture(self) -> np.ndarray:
        pass

    def _check_shape(self, img: np.ndarray):
        return True

    def capture(self, force=False) -> np.ndarray:
        """供外部调用的截图接口

        注意：取帧失败时返回的是**上一帧缓存**（不是 None），调用方无法从返回值
        判断新旧。需要区分时请用 capture_with_status() 或 get_frame_status()。

        Args:
            force: 无视帧率限制，强制截图
        """
        if DEBUG_MODE:
            r = self.cap_per_sec.count_times()
            if r:
                if r != self.last_cap_times:
                    logger.trace(f"capps: {r/3}")
                    self.last_cap_times = r
                elif r >= 10 * 3:
                    logger.trace(f"capps: {r/3}")
                elif r >= 20 * 3:
                    logger.debug(f"capps: {r/3}")
                elif r >= 40 * 3:
                    logger.info(f"capps: {r/3}")
        with self.capture_cache_lock:
            self._capture(force)
            cp = self.capture_cache.copy()
        return cp

    def get_frame_status(self) -> dict:
        """返回当前缓存帧的新鲜度快照（只读，不触发截图）。

        Returns:
            dict:
                fresh: 最近一次取帧尝试是否成功。True 表示 capture() 返回的是新帧；
                    False 表示正在沿用旧帧（或还从未成功取过帧）。
                age_ms: 缓存帧帧龄（毫秒）。从未成功取帧时为 None（此时缓存是初始全零图）。
                stale: 是否"确实取帧失败、且帧龄已超过 STALE_AGE_MS"。上层据此可以
                    放弃不可逆操作（例如盲走、点击确认框），仅 age_ms 大不构成 stale。
                fail_count: 连续失败次数，成功即清零。
                reason: 最近一次失败原因（FAIL_REASON_*），新鲜时为 None。
                generation: 成功取帧的累计次数，可用于判断两次读取之间有没有产生新帧。
                last_success_ts: 上次成功的墙钟时间戳（time.time()），从未成功时为 None。
        """
        with self._status_lock:
            generation = self._frame_generation
            fail_count = self._consecutive_failures
            reason = self._last_fail_reason
            last_success_mono = self._last_success_mono
            last_success_wall = self._last_success_wall
        age_ms = None if last_success_mono is None else (time.monotonic() - last_success_mono) * 1000.0
        return {
            "fresh": bool(fail_count == 0 and last_success_mono is not None),
            "age_ms": age_ms,
            "stale": bool(fail_count > 0 and age_ms is not None and age_ms >= STALE_AGE_MS),
            "fail_count": fail_count,
            "reason": reason,
            "generation": generation,
            "last_success_ts": last_success_wall,
        }

    @property
    def frame_status(self) -> dict:
        """get_frame_status() 的只读属性形式，便于 `itt.capture_obj.frame_status` 直接取用。"""
        return self.get_frame_status()

    def capture_with_status(self, force=False):
        """截图并同时返回本帧新鲜度，语义上等价于 capture() + get_frame_status()。

        Returns:
            (np.ndarray, dict): 帧本身，以及本次取帧后的 get_frame_status() 快照。
        """
        frame = self.capture(force=force)
        return frame, self.get_frame_status()

    def _on_capture_success(self) -> None:
        """记录一次成功取帧。调用方需已持有 capture_cache_lock。"""
        with self._status_lock:
            self._frame_generation += 1
            self._consecutive_failures = 0
            self._last_fail_reason = None
            self._last_success_mono = time.monotonic()
            self._last_success_wall = time.time()

    def _on_capture_failed(self, reason: str) -> None:
        """记录一次取帧失败，必要时告警。调用方需已持有 capture_cache_lock。"""
        with self._status_lock:
            self._consecutive_failures += 1
            self._last_fail_reason = reason
            fail_count = self._consecutive_failures
            last_success_mono = self._last_success_mono
            last_success_wall = self._last_success_wall
        # 日志放到锁外，避免 IO 拉长持锁时间（这条路径在 30fps 下会被频繁走到）
        if fail_count < FAIL_WARN_THRESHOLD:
            return
        if fail_count > FAIL_WARN_THRESHOLD and (fail_count - FAIL_WARN_THRESHOLD) % FAIL_WARN_INTERVAL != 0:
            return
        age_text = "N/A" if last_success_mono is None else f"{int((time.monotonic() - last_success_mono) * 1000)}ms"
        last_ok_text = "从未成功" if last_success_wall is None else time.strftime("%H:%M:%S", time.localtime(last_success_wall))
        logger.warning(
            f"连续取帧失败 {fail_count} 次（原因: {reason}），已沿用旧帧"
            f"（帧龄 {age_text}，上次成功时间 {last_ok_text}）"
        )

    def _capture(self, force) -> None:
        if (self.fps_timer.get_diff_time() >= 1 / self.max_fps) or force:
            self.fps_timer.reset()
            self.capture_times += 1
            raw_img = self._get_capture()
            if raw_img is None:
                # 取帧本身失败（PrintWindow 失败 / 窗口句柄失效），沿用旧帧但记下失败
                self._on_capture_failed(FAIL_REASON_CAPTURE)
                return
            normalized_img = self._normalize_shape(raw_img)
            if normalized_img is None:
                # 取到图但分辨率比例校验不通过（_check_shape 里已打过日志）
                self._on_capture_failed(FAIL_REASON_SHAPE)
                return
            self.capture_cache = normalized_img
            self._on_capture_success()


class PrintWindowCapture(Capture):
    """Platform-agnostic capture class.

    Delegates the actual pixel acquisition to the CaptureManager so that
    no platform-specific code lives here.
    """

    def __init__(self, hwnd_handler):
        super().__init__(hwnd_handler)
        # 帧率上限已在基类通过 resolve_max_fps() 解析，这里不再覆盖硬编码值，
        # 否则环境变量 WHIMBOX_CAPTURE_MAX_FPS 会被冲掉。
        self._cap_mgr = get_capture_manager()

    def _check_shape(self, img: np.ndarray):
        if img is None:
            return False
        if len(img.shape) >= 3 and img.shape[2] >= 3 and img.shape[1] > 0 and 1.55 < img.shape[1] / img.shape[0] < 1.80:
            # 支持16:9和16:10分辨率
            # 有些用户在特定显示器和缩放下会生成奇怪的1920x1081分辨率，增加宽容度
            return True
        else:
            # 降到 trace：这条每帧都会命中，由 _on_capture_failed() 按连续失败次数汇总告警，
            # 否则 30fps 下会把日志刷爆，真正的故障反而被淹没
            logger.trace("游戏分辨率异常: " + str(img.shape))
            return False

    def _get_capture(self):
        return self._cap_mgr.capture_window(
            self.hwnd_handler.get_handle(),
            self.hwnd_handler.pid,
        )


if __name__ == '__main__':
    pass
