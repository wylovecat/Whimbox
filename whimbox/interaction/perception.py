"""感知层：只读的「看屏幕」能力。

从 InteractionBGD 中拆出来的原因
--------------------------------
InteractionBGD 一直同时承担两件事：

1. **感知**：截图、裁剪、OCR、模板匹配 —— 纯读操作，不改焦点、不按鼠标键盘、
   不需要持有 ``operation_lock``；
2. **动作**：点击/按键/滚轮 —— 会抢窗口焦点（``before_operation``）、必须串行化。

两者混在一个 516 行的类里带来两个实际问题：
- 任何只想「截一帧 / 识别一下」的代码（比如地图遮罩的检测线程、离线分析脚本、
  单元测试）都必须先构造整个 InteractionBGD，而它的构造会连带创建输入注入器；
- 感知代码无法单独测试：想喂一张固定帧进去，只能把整个对象构造出来再替换内部成员。

拆分后 ``Perception`` 只依赖一个「能返回 BGRA 帧」的对象（``Capture`` 实现），
因此可以完全脱离输入设备构造和测试。InteractionBGD 保留同名方法作为委托，
61 处 ``from whimbox.interaction.interaction_core import itt`` 调用点无需改动。

边界约定
--------
- 本模块**不得**导入 ``interaction_normal``（输入注入），也不得持有任何锁；
- 本模块的方法对同一张帧必须给出确定的结果（这是 golden 基线回放的前提）；
- 日志里的「调用者函数名」由调用方传入（``caller_name``），因为一旦经过委托层，
  ``inspect`` 拿到的栈帧会变成委托方法本身，语义会漂移。
"""
import inspect
import time

import cv2

from whimbox.common import path_lib
from whimbox.common.cvars import *
from whimbox.common.logger import logger
from whimbox.common.timer_module import AdvanceTimer, TimeoutTimer
from whimbox.common.utils.asset_utils import AnchorPosi
from whimbox.common.utils.img_utils import add_padding, crop, process_with_hsv_limit, similar_img
from whimbox.config.config import global_config
from whimbox.ui.template import img_manager, posi_manager, text_manager

ocr_type = global_config.get('General', 'ocr')
if ocr_type == 'rapid':
    from whimbox.api.ocr_rapid import ocr
else:
    raise ValueError(f"ocr配置错误：{ocr_type}")


class Perception:
    """只读感知能力。

    Args:
        capture_obj: 具备 ``capture() -> np.ndarray``（BGRA）与 ``resolution`` 的对象，
            生产环境是 ``PrintWindowCapture``，测试里可以换成任何返回固定帧的桩。
    """

    def __init__(self, capture_obj):
        self.capture_obj = capture_obj

    def capture(self, anchor_posi: AnchorPosi = None, jpgmode=NORMAL_CHANNELS):
        """窗口客户区截图

        Args:
            posi ( AnchorPosi ): 截图区域的坐标, y2>y1,x2>x1. 全屏截图时为None。
            jpgmode(int):
                0:return jpg (3 channels, delete the alpha channel)
                1:return nikki background channel, background color is black
                2:return nikki ui channel, background color is black

        Returns:
            numpy.ndarray: 图片数组
        """
        ret = self.capture_obj.capture()
        if anchor_posi is not None:
            ret = crop(ret, anchor_posi)
        if ret.shape[2] == 3:
            pass
        elif jpgmode == NORMAL_CHANNELS:
            ret = ret[:, :, :3]
        elif jpgmode == FOUR_CHANNELS:
            return ret
        return ret

    def get_screen_center(self):
        resolution = self.capture_obj.resolution
        if resolution:
            x = 1920 / 2
            y = 1920 / 2 * resolution[0] / resolution[1]
            print(x, y)
            return (int(x), int(y))
        else:
            return (1920 / 2, 1080 / 2)

    def ocr_single_line(self, area: posi_manager.Area, padding=50, hsv_limit=None, cap=None) -> str:
        if cap is None:
            cap = self.capture(anchor_posi=area.position)
        else:
            cap = crop(cap, area.position)
        if hsv_limit:
            cap = process_with_hsv_limit(cap, hsv_limit[0], hsv_limit[1])
        if padding:
            cap = add_padding(cap, padding)
        res = ocr.get_all_texts(cap, mode=1)
        return res

    def ocr_multiple_lines(self, area: posi_manager.Area, padding=50, hsv_limit=None) -> list:
        cap = self.capture(anchor_posi=area.position)
        if hsv_limit:
            cap = process_with_hsv_limit(cap, hsv_limit[0], hsv_limit[1])
        if padding:
            cap = add_padding(cap, padding)
        res = ocr.get_all_texts(cap, mode=0)
        return res

    def ocr_and_detect_posi(self, area: posi_manager.Area, padding=50, hsv_limit=None, cap=None, show_res=False):
        if cap is None:
            cap = self.capture(anchor_posi=area.position)
        else:
            cap = crop(cap, area.position)
        if hsv_limit:
            cap = process_with_hsv_limit(cap, hsv_limit[0], hsv_limit[1])
        if padding:
            cap = add_padding(cap, padding)
        res = ocr.detect_and_ocr(cap, show_res)
        if padding:
            for text, box in res.items():
                res[text] = [
                    box[0] - padding,
                    box[1] - padding,
                    box[2] - padding,
                    box[3] - padding]
        return res

    def get_img_existence(self, imgicon: img_manager.ImgIcon, is_gray=False, ret_mode=IMG_BOOL,
                          show_res=False, cap=None, caller_name: str = ""):
        """检测图片是否存在

        Args:
            imgicon (img_manager.ImgIcon): imgicon对象
            is_gray (bool, optional): 是否启用灰度匹配. Defaults to False.
            is_log (bool, optional): 是否打印日志. Defaults to False.
            caller_name (str, optional): 调用者函数名，仅用于日志。经 InteractionBGD 委托
                调用时必须由调用方传入，否则日志里记的会是委托方法名。

        Returns:
            bool: bool
        """
        if not caller_name:
            caller_name = inspect.getframeinfo(inspect.currentframe().f_back)[2]
        if cap is None:
            cap = self.capture(anchor_posi=imgicon.cap_posi)
        if imgicon.hsv_limit is not None:
            cap = process_with_hsv_limit(cap, imgicon.hsv_limit[0], imgicon.hsv_limit[1])
        elif imgicon.gray_limit is not None:
            cap = cv2.cvtColor(cap, cv2.COLOR_BGRA2GRAY)
            _, cap = cv2.threshold(cap, imgicon.gray_limit[0], imgicon.gray_limit[1], cv2.THRESH_BINARY)
        matching_rate = similar_img(cap, imgicon.image, is_gray=is_gray, is_show_res=show_res)

        if imgicon.is_print_log(matching_rate >= imgicon.threshold):
            logger.trace(
                'imgname: ' + imgicon.name + ' matching_rate: ' + str(
                    matching_rate) + ' |function name: ' + caller_name)

        if ret_mode == IMG_BOOLRATE:
            if matching_rate >= imgicon.threshold:
                return matching_rate
            else:
                return False
        elif ret_mode == IMG_RATE:
            return matching_rate
        elif ret_mode == IMG_POSI:
            if matching_rate >= imgicon.threshold:
                return imgicon.cap_center_position_xy
            else:
                return None
        else:
            return matching_rate >= imgicon.threshold

    def get_text_existence(self, textobj: text_manager.TextTemplate, ret_mode=IMG_BOOL, cap=None):
        if cap == None:
            cap = self.capture(anchor_posi=textobj.cap_area.position)
        cap = add_padding(cap, 50)
        res = ocr.get_all_texts(cap)
        is_exist = textobj.match_results(res)
        if textobj.is_print_log(is_exist):
            logger.trace(f"get_text_existence: text: {textobj.text} {'Found' if is_exist else 'Not Found'}")
        if ret_mode == IMG_POSI:
            if is_exist:
                return textobj.cap_area.center_position()
            else:
                return None
        else:
            return is_exist

    def appear(self, obj):
        if isinstance(obj, text_manager.TextTemplate):
            return self.get_text_existence(obj)
        elif isinstance(obj, img_manager.ImgIcon):  # Button is also an Icon
            return self.get_img_existence(obj)

    def wait_until_stable(self, threshold=0.9995, timeout=5):
        """等待画面稳定（连续两帧相似度超过阈值且持续一段时间）。"""
        timeout_timer = TimeoutTimer(timeout)
        last_cap = self.capture()

        pt = time.time()
        t = AdvanceTimer(0.25, 3).start()
        while 1:
            time.sleep(0.1)
            if timeout_timer.istimeout():
                logger.warning("TIMEOUT")
                break
            curr_img = self.capture()
            simi = similar_img(last_cap, curr_img)
            if simi > threshold:
                pass
            else:
                t.reset()
            if t.reached():
                if DEBUG_MODE:
                    print('wait time: ', time.time() - pt)
                break
            last_cap = curr_img.copy()

    def save_snapshot(self, reason: str = ''):
        """把当前画面存到日志目录（排障用）。"""
        img = self.capture()
        img_path = path_lib.LOG_PATH + "/" + f"{time.time()}.jpg"
        logger.warning(f"Snapshot saved to {img_path}")
        cv2.imwrite(img_path, img)
