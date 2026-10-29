"""P3 回归：分层依赖方向（任务层不许反向依赖接入层 RPC）。

背景：``whimbox/rpc_server.py``（接入层）→ ``whimbox/rpc_method_groups.py`` →
``whimbox/task/*``（任务层）。而任务层曾经在函数内 inline
``from whimbox.rpc_server import notify_event`` 反向回头 import 接入层，
于是「import 任务层」与「import 接入层」互为前置，形成循环导入。

修法不是新造机制：``whimbox/event_bus.py`` 早就提供了 ``set_notifier`` /
``emit_event`` 的晚绑定钩子，``rpc_server`` 启动时也一直有
``set_notifier(notify_event)``，只是任务层几个调用点没走它。改成 ``emit_event``
之后依赖方向变成单向：任务层 → event_bus（无依赖叶子），接入层 → event_bus。

两个用例分别是静态与动态检查：
- 静态：全仓没有其它模块 import rpc_server（只有入口 ``whimbox/main.py`` 可以）；
- 动态：真的 import 任务层模块，确认 ``whimbox.rpc_server`` 没有被拖进 sys.modules。
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_MARKER = "MARKER:"

# 唯一允许 import 接入层的模块：程序入口（它负责把 server 拉起来）。
_ALLOWED_IMPORTERS = {"main.py"}


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


def test_only_entrypoint_imports_rpc_server() -> None:
    """除了 whimbox/main.py，任何模块都不应该 import rpc_server。"""
    package_root = REPO_ROOT / "whimbox"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        if path.name in _ALLOWED_IMPORTERS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in (
            "from whimbox.rpc_server import",
            "import whimbox.rpc_server",
        ):
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()} -> {needle}")
    assert not offenders, (
        "任务层/业务层不允许反向 import 接入层 rpc_server，请改用 "
        "whimbox.event_bus.emit_event：\n" + "\n".join(offenders)
    )


def test_importing_task_layer_does_not_load_rpc_server() -> None:
    """真的 import 任务层，确认接入层没有被顺带拖进来。"""
    probe = """
import sys
import whimbox.task.task_template  # noqa: F401

print("MARKER:rpc=%s" % ("whimbox.rpc_server" in sys.modules))
"""
    result = _run_probe(probe)
    assert result["rpc"] == "False", (
        "import whimbox.task.task_template 把 whimbox.rpc_server 也拉进来了，"
        "说明任务层又反向依赖了接入层"
    )
