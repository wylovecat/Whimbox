"""动作层与感知层的组合门面（历史上的「神对象」）。

拆分背景
--------
本模块原本同时实现感知（截图/OCR/模板匹配）与动作（点击/按键/滚轮）。
感知部分已抽到 :mod:`whimbox.interaction.perception`，这里保留同名方法作为
**委托**，因此 61 处 ``from whimbox.interaction.interaction_core import itt``
以及所有 ``itt.capture()`` / ``itt.get_img_existence()`` 调用点都无需改动。

为什么要拆：感知是纯读操作、不需要锁、不需要窗口焦点，而动作会抢焦点并且必须
串行化；混在一起导致「只想截一帧」的代码（地图遮罩检测线程、离线分析、单元测试）
也不得不构造输入注入器。详见 perception.py 的模块文档。

分工约定
--------
- 感知方法（capture / ocr_* / get_*_existence / appear / wait_until_stable /
  save_snapshot）在本类里只有一行委托，实现全部在 ``Perception``；
- 动作方法（appear_then_click / click / key_* / move_* / delay）留在本类，
  因为它们要经过 ``before_operation``（抢焦点 + 商城抽卡界面拦截）与
  ``operation_lock``（串行化）。

行为等价：感知层有一份逐键的输出基线（tests/fixtures/perception/api_baseline.json），
由 ``tests/test_perception_api_baseline.py`` 回放比对，拆分前后必须完全一致。
"""
import cv2
import inspect
import math
import os
import random
import threading
import time

from whimbox.common import path_lib
from whimbox.common.cvars import *
from whimbox.common.logger import logger, get_logger_format_date
from whimbox.common.path_lib import ROOT_PATH
from whimbox.common.timer_module import AdvanceTimer, TimeoutTimer
from whimbox.common.utils.asset_utils import AnchorPosi
from whimbox.common.utils.img_utils import add_padding, crop, process_with_hsv_limit, similar_img
from whimbox.config.config import global_config
from whimbox.interaction.perception import Perception
from whimbox.interaction.perception import ocr  # noqa: F401 - 保持历史导入路径可用
from whimbox.ui.template import img_manager, posi_manager, text_manager
from whimbox.ui.ui_assets import IconGachaFeature, IconShopFeature


class InteractionBGD:
    """
    thanks for https://zhuanlan.zhihu.com/p/361569101
    """

    def __init__(self, hwnd_handler):
        self.hwnd_handler = hwnd_handler
        logger.info("InteractionBGD created")
        self.WHEEL_DELTA = 120
        self.DEFAULT_DELAY_TIME = 0.05
        self.itt_exec = None
        self.capture_obj = None
        self.operation_lock = threading.Lock()
        import whimbox.interaction.interaction_normal
        self.itt_exec = whimbox.interaction.interaction_normal.InteractionNormal(self.hwnd_handler)
        from whimbox.interaction.capture import PrintWindowCapture
        self.capture_obj = PrintWindowCapture(self.hwnd_handler)
        # 感知能力独立成对象：只依赖 capture_obj，构造它不会创建任何输入注入器
        self.perception = Perception(self.capture_obj)

    # ------------------------------------------------------------------
    # 感知层委托（实现见 whimbox/interaction/perception.py）
    # ------------------------------------------------------------------
    def capture(self, anchor_posi: AnchorPosi=None, jpgmode=NORMAL_CHANNELS):
        """窗口客户区截图，实现见 Perception.capture。"""
        return self.perception.capture(anchor_posi=anchor_posi, jpgmode=jpgmode)

    def get_screen_center(self):
        return self.perception.get_screen_center()

    def ocr_single_line(self, area: posi_manager.Area, padding=50, hsv_limit=None, cap=None) -> str:
        return self.perception.ocr_single_line(area, padding=padding, hsv_limit=hsv_limit, cap=cap)

    def ocr_multiple_lines(self, area: posi_manager.Area, padding=50, hsv_limit=None) -> list:
        return self.perception.ocr_multiple_lines(area, padding=padding, hsv_limit=hsv_limit)

    def ocr_and_detect_posi(self, area: posi_manager.Area, padding=50, hsv_limit=None, cap=None, show_res=False):
        return self.perception.ocr_and_detect_posi(
            area, padding=padding, hsv_limit=hsv_limit, cap=cap, show_res=show_res)

    def get_img_existence(self, imgicon: img_manager.ImgIcon, is_gray=False, ret_mode = IMG_BOOL, show_res = False, cap = None):
        """检测图片是否存在。

        这里显式把调用者函数名传给感知层：日志里记的是「谁调用了感知」，一旦经过
        委托层，感知层里 inspect 拿到的栈帧就会变成这个委托方法本身。
        """
        caller_name = inspect.getframeinfo(inspect.currentframe().f_back)[2]
        return self.perception.get_img_existence(
            imgicon, is_gray=is_gray, ret_mode=ret_mode, show_res=show_res, cap=cap, caller_name=caller_name)

    def get_text_existence(self, textobj: text_manager.TextTemplate, ret_mode=IMG_BOOL, cap=None):
        return self.perception.get_text_existence(textobj, ret_mode=ret_mode, cap=cap)

    def appear(self, obj):
        return self.perception.appear(obj)

    def wait_until_stable(self, threshold = 0.9995, timeout = 5):
        return self.perception.wait_until_stable(threshold=threshold, timeout=timeout)

    def save_snapshot(self, reason:str = ''):
        return self.perception.save_snapshot(reason=reason)

    # ------------------------------------------------------------------
    # 动作层
    # ------------------------------------------------------------------

    def appear_then_click(self, inputvar, is_gray=False, key_name="left_mouse"):
        """appear then click

        Args:
            inputvar (img_manager.ImgIcon/text_manager.TextTemplate/button_manager.Button)
            is_gray (bool, optional): 是否启用灰度匹配. Defaults to False.
            key_name (str, optional): 按键名称. Defaults to "left_mouse".

        Returns:
            bool: bool,点击操作是否成功
        """

        upper_func_name = inspect.getframeinfo(inspect.currentframe().f_back)[2]
        match_position = None

        if isinstance(inputvar, img_manager.ImgIcon):
            match_position = self.get_img_existence(inputvar, is_gray=is_gray, ret_mode=IMG_POSI)
            anchor = inputvar.cap_posi.anchor

        elif isinstance(inputvar, text_manager.TextTemplate):
            match_position = self.get_text_existence(inputvar, ret_mode=IMG_POSI)
            anchor = inputvar.cap_area.anchor

        if match_position:
            logger.trace(f"appear then click: True: {inputvar.name} func: {upper_func_name}")
            if key_name == "left_mouse":
                self.move_and_click(match_position, anchor=anchor)
            elif key_name == "right_mouse":
                self.move_and_click(match_position, anchor=anchor, type='right')
            else:
                self.key_press(key_name)
            return True
        else:
            return False


    def delay(self, x, randtime=False, is_log=True, comment=''):
        """延迟一段时间，单位为秒

        Args:
            x : 延迟时间/key words
            randtime (bool, optional): 是否启用加入随机秒. Defaults to True.
            is_log (bool, optional): 是否打印日志. Defaults to True.
            comment (str, optional): 日志注释. Defaults to ''.
        """
        if x  == "animation":
            time.sleep(0.3)
            return
        if x  == "2animation":
            time.sleep(0.6)
            return
        upper_func_name = inspect.getframeinfo(inspect.currentframe().f_back)[2]
        a = random.randint(-10, 10)
        if randtime:
            a = a * x * 0.02
            if x > 0.2 and is_log:
                logger.debug('delay: ' + str(x) + ' rand: ' +
                             str(x + a) + ' |function name: ' + upper_func_name + ' |comment: ' + comment)
            time.sleep(x + a)
        else:
            if x > 0.2 and is_log:
                logger.debug('delay: ' + str(x) + ' |function name: ' + upper_func_name + ' |comment: ' + comment)
            time.sleep(x)

    def _can_interact(self, func_name: str):
        # 判断是否在商城和抽卡界面，在的话禁止操作
        if func_name in ["left_click", "left_down", "left_double_click", "move_and_click"]:
            if self.get_img_existence(IconShopFeature) or self.get_img_existence(IconGachaFeature):
                return False
        return True

    @staticmethod
    def before_operation(print_log=False):
        """装饰器方法：在操作前进行检查"""
        def outwrapper(func):
            def wrapper(self, *args, **kwargs):
                if print_log:
                    func_name = inspect.getframeinfo(inspect.currentframe().f_back)[2]
                    func_name_2 = inspect.getframeinfo(inspect.currentframe().f_back.f_back)[2]
                    logger.trace(f" operation: {func.__name__} | args: {args} | {kwargs} | function name: {func_name} & {func_name_2}")

                if not self._can_interact(func.__name__):
                    raise Exception("中断操作：误入商城和抽卡界面")

                if not self.hwnd_handler.is_foreground():
                    # stop_flag = get_current_stop_flag()
                    # while True:
                    #     if stop_flag.is_set():
                    #         return None
                    #     if self.hwnd_handler.is_foreground():
                    #         logger.info("恢复操作")
                    #         break
                    #     logger.info(f"前台窗口不是目标窗口，操作暂停 {str(5 - (time.time()%5))} 秒")
                    #     time.sleep(5 - (time.time()%5))
                    self.hwnd_handler.set_foreground()
                    logger.info("游戏窗口丢失焦点，已强制恢复")
                return func(self, *args, **kwargs)
            return wrapper
        return outwrapper

    @before_operation()
    def left_click(self):
        """左键点击"""
        self.itt_exec.left_click()

    @before_operation()
    def left_down(self):
        """左键按下"""
        self.itt_exec.left_down()

    @before_operation()
    def left_up(self):
        """左键抬起"""
        self.itt_exec.left_up()

    @before_operation()
    def left_double_click(self, dt=0.05):
        """左键双击

        Args:
            dt (float, optional): 间隔时间. Defaults to 0.05.
        """
        self.itt_exec.left_double_click(dt=dt)

    @before_operation()
    def right_down(self):
        """右键按下"""
        self.itt_exec.right_down()

    @before_operation()
    def right_up(self):
        """右键抬起"""
        self.itt_exec.right_up()

    @before_operation()
    def right_click(self):
        """右键单击"""
        self.itt_exec.right_click()

    @before_operation()
    def middle_click(self):
        """点击鼠标中键"""
        self.itt_exec.middle_click()

    @before_operation()
    def middle_scroll(self, distance):
        """滚动鼠标中键"""
        with self.operation_lock:
            self.itt_exec.middle_scroll(distance)

    @before_operation()
    def key_down(self, key):
        """按下按键

        Args:
            key (str): 按键代号。查阅vkCode.py
        """
        if key == "mouse_left":
            self.itt_exec.left_down()
        elif key == "mouse_right":
            self.itt_exec.right_down()
        elif key == "mouse_middle":
            self.itt_exec.middle_down()
        else:
            self.itt_exec.key_down(key)

    @before_operation()
    def key_up(self, key):
        """松开按键

        Args:
            key (str): 按键代号。查阅vkCode.py
        """
        if key == "mouse_left":
            self.itt_exec.left_up()
        elif key == "mouse_right":
            self.itt_exec.right_up()
        elif key == "mouse_middle":
            self.itt_exec.middle_up()
        else:
            self.itt_exec.key_up(key)

    @before_operation()
    def key_press(self, key):
        """点击按键

        Args:
            key (str): 按键代号。查阅vkCode.py
        """
        if key == "mouse_left":
            self.itt_exec.left_click()
        elif key == "mouse_right":
            self.itt_exec.right_click()
        elif key == "mouse_middle":
            self.itt_exec.middle_click()
        else:
            self.itt_exec.key_press(key)

    @before_operation(print_log=False)
    def move_to(self, position, anchor=ANCHOR_TOP_LEFT, relative=False, smooth=False, smooth_duration=0.2):
        """移动鼠标到坐标

        Args:
            position (list): 坐标
            relative (bool): 是否为相对移动。
        """
        with self.operation_lock:
            self.itt_exec.move_to(
                int(position[0]),
                int(position[1]),
                resolution=self.capture_obj.resolution,
                anchor=anchor,
                relative=relative,
                smooth=smooth,
                smooth_duration=smooth_duration)


    @before_operation()
    def move_and_click(self, position, anchor=ANCHOR_TOP_LEFT, type='left', delay=0.3):
        """移动鼠标到坐标并点击

        Args:
            position (list): 坐标
            type (str, optional): 点击类型。 Defaults to 'left'.
            delay (float, optional): 延迟时间. Defaults to 0.2.
        """
        # 移动与点击必须在同一把锁内完成：否则其它线程可能在两者之间插入移动，
        # 导致点击落在错误坐标上。因此这里的 delay 也留在临界区内
        # （delay 为 0.3s 量级，且输入线程本身已由该锁串行化）。
        with self.operation_lock:
            self.itt_exec.move_to(
                int(position[0]),
                int(position[1]),
                resolution=self.capture_obj.resolution,
                anchor=anchor,
                relative=False)
            time.sleep(delay)

            if type == 'left':
                self.itt_exec.left_click()
            elif type == 'right':
                self.itt_exec.right_click()
            elif type == 'middle':
                self.itt_exec.middle_click()


# ---------------------------------------------------------------------------
# 模块级单例 itt（延迟构造）
#
# 历史实现是在 import 阶段直接 `itt = InteractionBGD(HANDLE_OBJ)`，而
# InteractionBGD.__init__ 会构造 InteractionNormal 与 PrintWindowCapture，
# 后者又会经平台工厂创建截图管理器 —— 也就是说「import 这个模块」本身就
# 已经在碰游戏窗口了。单元测试、静态检查、只想复用 capture() 的工具都会
# 被这些副作用牵连（游戏没开时更是直接失败）。
#
# 改为 PEP 562 的模块级 __getattr__ 懒加载：第一次真正访问 itt 时才构造。
# 61 处 `from whimbox.interaction.interaction_core import itt` 无需改动，
# 因为 from-import 同样走模块属性查找，会命中 __getattr__。
# ---------------------------------------------------------------------------
_itt_lock = threading.Lock()
_itt_singleton = None


def get_itt():
    """返回 InteractionBGD 单例，首次调用时才构造（线程安全）。"""
    global _itt_singleton
    if _itt_singleton is None:
        with _itt_lock:
            if _itt_singleton is None:
                from whimbox.common.handle_lib import HANDLE_OBJ

                _itt_singleton = InteractionBGD(HANDLE_OBJ)
    return _itt_singleton


def __getattr__(name):
    """PEP 562：把模块属性 `itt` 的访问转成懒加载。"""
    if name == "itt":
        return get_itt()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == '__main__':
    pass
