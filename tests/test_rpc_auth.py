"""RPC 鉴权的端到端测试。

P0-1 给 RPC 加了令牌校验，但静态检查证明不了「带 token 能连、不带 token 被拒」
这种交互行为，所以这里起一个真实的 websocket 服务、用真实客户端握手。

做法：直接挂 ``rpc_server._ws_handler`` 到临时端口，绕开 ``start_rpc_server``
（后者会启动 scripts_watcher、微信恢复、全局热键监听，与本测试无关，而且端口
固定 8350 会与正在运行的客户端冲突）。鉴权开关与令牌都用 patch 注入，
不读也不写用户的真实配置与 token 文件。

覆盖的行为契约：
1. 无令牌连接：回 -32001，随后以 4401 关闭，且不会进入 _clients（收不到事件推送）
2. 查询参数令牌正确：连接可用，业务方法能派发
3. 查询参数令牌错误：拒绝
4. 首帧 auth 握手：回 {"ok": true}，之后业务方法可用
5. 首帧不是 auth（直接调业务方法）：拒绝
6. Rpc.require_auth=false：不带令牌也放行（旧前端兼容路径）
"""
import asyncio
import json
import os
import sys
import unittest
from unittest import mock

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from whimbox import rpc_server  # noqa: E402

TEST_TOKEN = "test-token-1234567890"
UNKNOWN_METHOD = "__definitely_not_a_method__"


class RpcAuthTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._patches = [
            mock.patch.object(rpc_server, "_rpc_token", TEST_TOKEN),
            # 失败退避 1 秒会让每个拒绝用例都慢 1 秒；退避本身不是本测试的对象
            mock.patch.object(rpc_server, "_RPC_AUTH_FAILURE_BACKOFF_SECONDS", 0.02),
        ]
        for patcher in self._patches:
            patcher.start()
        rpc_server._clients.clear()
        rpc_server._client_send_locks.clear()
        self.server = await websockets.serve(rpc_server._ws_handler, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        rpc_server._clients.clear()
        rpc_server._client_send_locks.clear()
        for patcher in reversed(self._patches):
            patcher.stop()

    def url(self, query: str = "") -> str:
        return f"ws://127.0.0.1:{self.port}/{query}"

    async def _recv_json(self, websocket, timeout: float = 5.0) -> dict:
        message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        return json.loads(message)

    async def _expect_rejected(self, websocket) -> None:
        """断言被拒：先收到 -32001，再以 4401 关闭。"""
        payload = await self._recv_json(websocket)
        self.assertEqual(payload.get("error", {}).get("code"), rpc_server.RPC_AUTH_ERROR_CODE)
        with self.assertRaises(websockets.exceptions.ConnectionClosed) as ctx:
            await asyncio.wait_for(websocket.recv(), timeout=5.0)
        close_frame = ctx.exception.rcvd
        self.assertIsNotNone(close_frame, "服务端应发送关闭帧而不是直接断开 TCP")
        self.assertEqual(close_frame.code, rpc_server.RPC_AUTH_CLOSE_CODE)

    async def _wait_for_no_clients(self, timeout: float = 2.0) -> int:
        """等客户端从 _clients 中摘除：服务端的清理在连接协程的 finally 里，
        与测试断言之间天然有竞态，所以轮询而不是立刻断言。"""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while rpc_server._clients and loop.time() < deadline:
            await asyncio.sleep(0.02)
        return len(rpc_server._clients)

    async def test_connection_without_token_times_out_and_is_rejected(self):
        """不带查询参数令牌、也不发首帧：等到握手超时后拒绝。"""
        with mock.patch.object(rpc_server, "_rpc_auth_required", return_value=True), \
                mock.patch.object(rpc_server, "_RPC_AUTH_TIMEOUT_SECONDS", 0.3):
            async with websockets.connect(self.url()) as websocket:
                await self._expect_rejected(websocket)
        # 关键：被拒的连接不得登记为客户端，否则能收到 event.* 推送
        self.assertEqual(await self._wait_for_no_clients(), 0)

    async def test_connection_with_wrong_token_is_rejected(self):
        with mock.patch.object(rpc_server, "_rpc_auth_required", return_value=True):
            async with websockets.connect(self.url("?token=wrong-token")) as websocket:
                await self._expect_rejected(websocket)
        self.assertEqual(await self._wait_for_no_clients(), 0)

    async def test_connection_with_correct_query_token_can_call_methods(self):
        with mock.patch.object(rpc_server, "_rpc_auth_required", return_value=True):
            async with websockets.connect(self.url(f"?token={TEST_TOKEN}")) as websocket:
                await websocket.send(json.dumps(
                    {"jsonrpc": "2.0", "id": 7, "method": UNKNOWN_METHOD}))
                payload = await self._recv_json(websocket)
                # 收到「方法不存在」说明鉴权已通过、请求已进入派发层
                self.assertEqual(payload.get("error", {}).get("code"), -32601)
                self.assertEqual(payload.get("id"), 7)
            self.assertEqual(await self._wait_for_no_clients(), 0)

    async def test_first_frame_auth_handshake_then_calls_allowed(self):
        with mock.patch.object(rpc_server, "_rpc_auth_required", return_value=True):
            async with websockets.connect(self.url()) as websocket:
                await websocket.send(json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "auth", "params": {"token": TEST_TOKEN}}))
                payload = await self._recv_json(websocket)
                self.assertEqual(payload.get("result"), {"ok": True})

                await websocket.send(json.dumps(
                    {"jsonrpc": "2.0", "id": 8, "method": UNKNOWN_METHOD}))
                payload = await self._recv_json(websocket)
                self.assertEqual(payload.get("error", {}).get("code"), -32601)
                self.assertEqual(payload.get("id"), 8)

    async def test_first_frame_must_be_auth_not_business_call(self):
        with mock.patch.object(rpc_server, "_rpc_auth_required", return_value=True):
            async with websockets.connect(self.url()) as websocket:
                await websocket.send(json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": UNKNOWN_METHOD}))
                await self._expect_rejected(websocket)
        self.assertEqual(await self._wait_for_no_clients(), 0)

    async def test_auth_disabled_still_allows_unauthenticated_clients(self):
        """旧前端兼容路径：require_auth=false 时不要求令牌。"""
        with mock.patch.object(rpc_server, "_rpc_auth_required", return_value=False):
            async with websockets.connect(self.url()) as websocket:
                await websocket.send(json.dumps(
                    {"jsonrpc": "2.0", "id": 9, "method": UNKNOWN_METHOD}))
                payload = await self._recv_json(websocket)
                self.assertEqual(payload.get("error", {}).get("code"), -32601)
                self.assertEqual(payload.get("id"), 9)


if __name__ == "__main__":
    unittest.main()
