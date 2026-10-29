"""P2 回归：import 副作用隔离（PEP 562 懒加载 itt）。

背景：``whimbox/interaction/interaction_core.py`` 原本在模块尾部直接执行

    from whimbox.common.handle_lib import HANDLE_OBJ
    itt = InteractionBGD(HANDLE_OBJ)

而 ``InteractionBGD.__init__`` 会构造 ``InteractionNormal`` 与
``PrintWindowCapture``（后者经平台工厂创建截图管理器）。结果是「import 这个模块」
本身就已经碰过游戏窗口了：没开游戏时失败，写单测时也无从下手。

两个断言都用子进程做隔离，避免被同一个 pytest 会话里其它用例污染 ``sys.modules``
（例如 ``test_map_mask.py`` 就直接 import 了 ``HANDLE_OBJ``）。
单例语义那个用例用桩替换 ``InteractionBGD`` 与 ``HANDLE_OBJ``，因此完全不碰
真实窗口，可以在无游戏的 CI 机器上跑。
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_MARKER = "MARKER:"


def _run_probe(code: str) -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"子进程执行失败\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr[-4000:]}"
    )
    for line in completed.stdout.splitlines():
        if line.startswith(_MARKER):
            body = line[len(_MARKER) :]
            return dict(part.split("=", 1) for part in body.split(";"))
    raise AssertionError(f"未找到 MARKER 输出\nstdout:\n{completed.stdout}")


# 只 import，观察两件事：
#   1) 模块 __dict__ 里还没有 itt —— 说明没有在 import 期构造单例
#   2) handle_lib 还没被 import —— 说明 ProcessHandler 的进程/窗口扫描也没被触发
IMPORT_PROBE = """
import sys
import whimbox.interaction.interaction_core as m

print("MARKER:eager_itt=%s;handle_lib=%s" % (
    "itt" in vars(m),
    "whimbox.common.handle_lib" in sys.modules,
))
"""


# 用桩替换 InteractionBGD / HANDLE_OBJ 后再访问 itt，验证单例语义与 from-import 兼容性。
SINGLETON_PROBE = """
import sys
import types
import whimbox.interaction.interaction_core as m

calls = []


class StubInteractionBGD:
    def __init__(self, hwnd_handler):
        calls.append(hwnd_handler)


fake_handle_lib = types.ModuleType("whimbox.common.handle_lib")
fake_handle_lib.HANDLE_OBJ = object()
sys.modules["whimbox.common.handle_lib"] = fake_handle_lib

m.InteractionBGD = StubInteractionBGD

first = m.get_itt()
second = m.get_itt()
from whimbox.interaction.interaction_core import itt as via_from_import

print("MARKER:same=%s;same_as_from_import=%s;constructed=%d" % (
    first is second,
    first is via_from_import,
    len(calls),
))
"""


def test_interaction_core_import_has_no_side_effects() -> None:
    """import interaction_core 不应该构造 itt，也不应该去找游戏窗口。"""
    result = _run_probe(IMPORT_PROBE)
    assert result["eager_itt"] == "False", "import 阶段就构造了 InteractionBGD，副作用回来了"
    assert result["handle_lib"] == "False", "import 阶段就 import 了 handle_lib（会扫描进程/窗口）"


def test_itt_is_lazy_singleton_and_from_import_compatible() -> None:
    """懒加载必须保持单例语义，且 61 处 ``from ... import itt`` 照旧可用。"""
    result = _run_probe(SINGLETON_PROBE)
    assert result["same"] == "True", "两次 get_itt() 不是同一个实例"
    assert result["same_as_from_import"] == "True", "from-import 的 itt 与模块属性不是同一实例"
    assert result["constructed"] == "1", f"InteractionBGD 构造了 {result['constructed']} 次，应为 1 次"
