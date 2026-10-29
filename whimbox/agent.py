import asyncio
import threading
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from whimbox.agent_workspace import AgentWorkspace, ContextBuilder, MemoryStore, ChatSessionManager
from whimbox.agent_workspace.session import MessageContent, content_to_model_content, has_content
from whimbox.agent_workspace.tools import build_workspace_tools
from whimbox.common.logger import logger
from whimbox.config.config import global_config
from whimbox.event_bus import emit_event
from whimbox.plugin_runtime import get_registry
from whimbox.plugin_tools import build_tools
from whimbox.common.cvars import DEBUG_MODE


class Agent:
    _instance = None
    _initialized = False
    _memory_window = 64

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(Agent, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.langchain_agent = None
        self.err_msg = ""
        self.tools_ready = False
        self.llm = None
        self.tools = None
        self._registry = None
        self._active_session_id = "default"
        self._session_stop_events = {}
        self._tool_running_sessions = set()
        self._current_tool_by_session = {}
        self._session_stream_tasks = {}
        self._consolidation_locks = {}
        self._consolidation_tasks = {}
        self._status = "starting"
        self.workspace = AgentWorkspace()
        self.context_builder = None
        self.memory_store = None
        self.session_manager = None
        self._model_name = ""
        self._model_provider = ""
        self._vision_warned = set()

        self._initialized = True

    def _set_status(self, status: str, message: str = "") -> None:
        self._status = status
        self.err_msg = message
        emit_event("event.agent.status", self.get_status())

    def get_status(self) -> dict[str, Any]:
        return {
            "ready": self.langchain_agent is not None,
            "status": self._status,
            "message": self.err_msg,
        }

    def _get_active_stop_event(self):
        return self._session_stop_events.get(self._active_session_id)

    async def start(self):
        logger.info("开始初始化agent")
        self._set_status("starting", "AI助手正在启动，请稍等...")
        self.workspace.ensure()
        if self.context_builder is None:
            self.context_builder = ContextBuilder(self.workspace.root)
        if self.memory_store is None:
            self.memory_store = MemoryStore(self.workspace.root)
        if self.session_manager is None:
            self.session_manager = ChatSessionManager(self.workspace.sessions_dir)
        logger.info("agent wokrspace准备就绪")

        api_key = global_config.get("Agent", "api_key")
        if not api_key:
            self.langchain_agent = None
            err_msg = "请先前往设置，配置大模型的api密钥。不会配置？点开通知，里面有详细白嫖教程！"
            self.llm = None
            self._set_status("missing_api_key", err_msg)
            logger.error(err_msg)
        else:
            try:
                self._model_name = str(global_config.get("Agent", "model") or "")
                self._model_provider = str(global_config.get("Agent", "model_provider") or "")
                if self._model_provider.startswith("deepseek"):
                    self._model_provider = "deepseek"
                self.llm = init_chat_model(
                    model=self._model_name,
                    model_provider=self._model_provider,
                    base_url=global_config.get("Agent", "base_url"),
                    api_key=api_key,
                )
            except Exception:
                self.llm = None
                err_msg = "AI初始化失败。请前往设置，检查大模型相关配置。"
                self._set_status("error", err_msg)
                logger.error(err_msg)

        self._rebuild_tools()
        logger.info("agent tools准备就绪")

        if self.llm and self.tools and self.context_builder and self.session_manager:
            self.langchain_agent = create_agent(
                model=self.llm,
                tools=self.tools,
                # debug=DEBUG_MODE,
            )
            self._set_status("ready", "")
            logger.info("AGENT 初始化完成")
        else:
            self.langchain_agent = None
            if self._status == "starting":
                self._set_status("error", "AI助手初始化失败")
            logger.error("AGENT 初始化失败")

    def is_ready(self):
        agent_ready = self.langchain_agent is not None
        return agent_ready, self.tools_ready, self.err_msg

    def request_stop(self, session_id: str):
        logger.debug(f"request_stop: session_id={session_id}")
        sid = session_id or "default"
        stop_event = self._session_stop_events.get(sid)
        stream_task = self._session_stream_tasks.get(sid)
        if stop_event is None and stream_task is None:
            return {"ok": False, "tool_running": sid in self._tool_running_sessions}
        if stop_event is not None:
            stop_event.set()
        return {"ok": True, "tool_running": sid in self._tool_running_sessions}

    def request_stop_all(self):
        session_ids = set(self._session_stop_events.keys()) | set(self._session_stream_tasks.keys())
        results = []
        for sid in session_ids:
            stop_event = self._session_stop_events.get(sid)
            if stop_event is not None:
                stop_event.set()
            results.append(
                {
                    "session_id": sid,
                    "tool_running": sid in self._tool_running_sessions,
                }
            )
        return results

    def is_tool_running(self, session_id: str | None = None) -> bool:
        if session_id is None:
            return bool(self._tool_running_sessions)
        sid = session_id or "default"
        return sid in self._tool_running_sessions

    def get_running_tool(self, session_id: str | None = None) -> str:
        if session_id is None:
            return ""
        sid = session_id or "default"
        return str(self._current_tool_by_session.get(sid) or "")

    async def query_agent(
        self,
        message_content: MessageContent,
        thread_id="default",
        stream_callback=None,
        status_callback=None,
        model_turn_callback=None,
    ):
        if not self.langchain_agent:
            err_msg = self.err_msg or "Agent 未就绪，请先完成初始化。"
            if status_callback:
                status_callback("error", err_msg, None)
            raise RuntimeError(err_msg)
        if not has_content(message_content):
            raise ValueError("message is required")
        if self._contains_image_content(message_content) and not self.supports_multimodal_input():
            raise RuntimeError("当前模型不支持图片输入，请更换支持视觉的模型配置。")

        session_id = thread_id or "default"
        self._active_session_id = session_id
        session = self.session_manager.get_or_create(session_id)
        input_payload = {
            "messages": self.context_builder.build_messages(
                history=session.get_history(max_messages=self._memory_window),
                current_message=message_content,
                session_id=session_id,
            )
        }

        logger.debug("开始调用大模型")
        stop_event = threading.Event()
        self._session_stop_events[session_id] = stop_event

        full_response = ""
        tools_used = []
        stream_cancelled = False
        active_tool_calls = 0

        if status_callback:
            status_callback("thinking", "", None)

        async def _run_stream():
            nonlocal full_response, active_tool_calls
            current_model_response = ""
            async for event in self.langchain_agent.astream_events(input_payload):
                event_type = event.get("event")
                data = event.get("data", {})

                if event_type == "on_chat_model_stream":
                    if active_tool_calls > 0:
                        continue
                    if stop_event.is_set():
                        break
                    content = self._extract_chunk_text(data.get("chunk"))
                    if content:
                        full_response += content
                        current_model_response += content
                        if stream_callback and content.strip():
                            stream_callback(content)

                elif event_type == "on_tool_start":
                    active_tool_calls += 1
                    tool_meta = event.get("metadata") or {}
                    tool_name = tool_meta.get("display_name") or event.get("name", "")
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)
                    self._tool_running_sessions.add(session_id)
                    self._current_tool_by_session[session_id] = tool_name
                    if status_callback:
                        status_callback("on_tool_start", tool_name, tool_meta)

                elif event_type == "on_tool_end":
                    active_tool_calls = max(0, active_tool_calls - 1)
                    tool_name = event.get("name", "")
                    self._tool_running_sessions.discard(session_id)
                    self._current_tool_by_session.pop(session_id, None)
                    if status_callback:
                        status_callback("on_tool_end", tool_name, {"output": data.get("output")})

                elif event_type == "on_tool_error":
                    active_tool_calls = max(0, active_tool_calls - 1)
                    error = data.get("error", "")
                    tool_name = event.get("name", "")
                    self._tool_running_sessions.discard(session_id)
                    self._current_tool_by_session.pop(session_id, None)
                    if stream_callback:
                        stream_callback(f"❌ 任务失败: {error}\n")
                    if status_callback:
                        status_callback("on_tool_error", tool_name, {"error": error})

                elif event_type == "on_chat_model_start":
                    current_model_response = ""
                    if status_callback:
                        status_callback("generating", "", None)

                elif event_type == "on_chat_model_end":
                    if model_turn_callback and current_model_response.strip():
                        model_turn_callback(current_model_response)
                    current_model_response = ""

                elif event_type == "on_chain_end":
                    final_content = self._extract_output_text(data.get("output"))
                    if final_content and not full_response.strip():
                        full_response += final_content
                        if stream_callback:
                            stream_callback(final_content)

        stream_task = asyncio.create_task(_run_stream())
        self._session_stream_tasks[session_id] = stream_task
        try:
            await stream_task
        except asyncio.CancelledError:
            logger.info(f"Agent stream cancelled: session={session_id}")
            stream_cancelled = True
        except Exception as exc:
            if status_callback:
                status_callback("error", "agent_error", {"error": str(exc)})
            raise
        finally:
            self._session_stream_tasks.pop(session_id, None)
            self._session_stop_events.pop(session_id, None)
            self._tool_running_sessions.discard(session_id)
            self._current_tool_by_session.pop(session_id, None)

        if stop_event.is_set() and not full_response.strip():
            full_response = "已停止当前对话。"

        self._save_session_turn(
            session,
            user_content=message_content,
            assistant_text=full_response,
            tools_used=tools_used,
        )
        self.session_manager.save(session)
        self._schedule_consolidation_if_needed(session_id=session_id)

        if status_callback and (stop_event.is_set() or stream_cancelled):
            status_callback("cancelled", "cancelled", None)
        elif status_callback:
            status_callback("completed", "completed", None)

        logger.debug("大模型调用完成")
        return full_response

    def get_ai_message(self, resp):
        ai_msgs = []
        for msg in resp["messages"]:
            if msg.type == "ai":
                ai_msgs.append(msg.content)
        return "\n".join(ai_msgs)

    def reload_tools(self):
        self._rebuild_tools()
        if self.tools:
            self.tools_ready = True
            self.err_msg = ""
            logger.debug(f"插件工具重载完成: {len(self.tools)}")
        else:
            self.tools_ready = False
            self.err_msg = "未加载任何插件工具"
            logger.error(self.err_msg)

    def _rebuild_tools(self):
        self._registry = get_registry()
        plugin_tools = build_tools(
            self._registry,
            session_id_getter=lambda: self._active_session_id,
            stop_event_getter=self._get_active_stop_event,
        )
        workspace_tools = build_workspace_tools(
            self.workspace.root,
            session_id_getter=lambda: self._active_session_id,
            stop_event_getter=self._get_active_stop_event,
            image_analyzer=self._analyze_image,
        )
        self.tools = [*plugin_tools, *workspace_tools]
        self.tools_ready = bool(plugin_tools)

    def _save_session_turn(
        self,
        session,
        user_content: MessageContent,
        assistant_text: str,
        tools_used: list[str],
    ) -> None:
        session.add_message("user", user_content)
        if assistant_text:
            session.add_message("assistant", assistant_text, tools_used=tools_used)

    def _schedule_consolidation_if_needed(self, *, session_id: str) -> None:
        if not self.memory_store or not self.llm or not self.session_manager:
            return
        session = self.session_manager.get_or_create(session_id)
        if not self.memory_store.should_consolidate(session=session, memory_window=self._memory_window):
            return
        existing_task = self._consolidation_tasks.get(session_id)
        if existing_task and not existing_task.done():
            return
        task = asyncio.create_task(self._run_consolidation(session_id))
        self._consolidation_tasks[session_id] = task

        def _cleanup(done_task):
            if self._consolidation_tasks.get(session_id) is done_task:
                self._consolidation_tasks.pop(session_id, None)

        task.add_done_callback(_cleanup)

    async def _run_consolidation(self, session_id: str) -> None:
        if not self.memory_store or not self.llm or not self.session_manager:
            return
        lock = self._consolidation_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self.session_manager.get_or_create(session_id)
            if not self.memory_store.should_consolidate(session=session, memory_window=self._memory_window):
                return
            ok = await self.memory_store.consolidate(
                session=session,
                llm=self.llm,
                memory_window=self._memory_window,
            )
            if ok:
                self.session_manager.save(session)
            else:
                logger.warning(f"后台记忆压缩失败: session={session_id}")

    def _extract_chunk_text(self, chunk) -> str:
        if hasattr(chunk, "content"):
            return self._extract_output_text(chunk)
        return ""

    def _extract_output_text(self, output) -> str:
        if output is None:
            return ""
        if isinstance(output, dict):
            messages = output.get("messages")
            if isinstance(messages, list):
                for item in reversed(messages):
                    item_type = getattr(item, "type", "")
                    if item_type in {"ai", "assistant"}:
                        return self._extract_output_text(item)
                    if isinstance(item, dict) and item.get("role") == "assistant":
                        return self._extract_output_text(item.get("content"))
            if "content" in output:
                return self._extract_output_text(output.get("content"))
            return ""
        content = getattr(output, "content", output)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)

    def _analyze_image(
        self,
        *,
        image_path: str,
        prompt: str,
        session_id: str,
        source_mode: str,
    ) -> dict[str, Any]:
        if not self.llm:
            return {
                "status": "error",
                "message": "Agent model is not ready.",
                "analysis": "",
                "image_source": image_path,
            }
        if not prompt.strip():
            return {
                "status": "error",
                "message": "prompt is required.",
                "analysis": "",
                "image_source": image_path,
            }
        if not self.supports_multimodal_input():
            return {
                "status": "error",
                "message": "当前模型不支持图片输入，请更换支持视觉的模型配置。",
                "analysis": "",
                "image_source": image_path,
            }

        try:
            content = content_to_model_content(
                [
                    {"type": "text", "text": prompt.strip()},
                    {"type": "image_file", "path": image_path},
                ]
            )
            response = self.llm.invoke([HumanMessage(content=content)])
            analysis = self._extract_output_text(response).strip()
            return {
                "status": "success",
                "message": "图片分析完成",
                "analysis": analysis,
                "image_source": image_path,
                "source_mode": source_mode,
                "session_id": session_id,
            }
        except Exception as exc:
            logger.warning(f"image analysis failed: {exc}")
            return {
                "status": "error",
                "message": str(exc),
                "analysis": "",
                "image_source": image_path,
                "source_mode": source_mode,
                "session_id": session_id,
            }

    # 已知支持图片输入的 provider（其主流模型均可接收 image block）
    _VISION_PROVIDERS = frozenset({"anthropic", "google_genai", "google", "google-genai"})
    # 已知支持图片输入的模型名关键字（小写子串匹配；name 需能被 provider 识别）
    _VISION_MODEL_TOKENS = (
        "gpt-4o",
        "gpt-4.1",
        "gpt-4-turbo",
        "gpt-5",
        "o1",
        "o3",
        "o4",
        "vision",
        "vl",  # qwen2-vl / qwen2.5-vl / 其它以 -vl 结尾的视觉模型
        "glm-4v",
        "internvl",
        "llava",
        "bakllava",
        "minicpm-v",
        "moondream",
        "pixtral",
        "step-1v",
    )
    _VISION_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
    _VISION_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

    def _vision_config_override(self) -> bool | None:
        """读取 Agent.supports_vision 覆盖项；留空或无法识别时返回 None（走自动判定）。"""
        raw = global_config.get("Agent", "supports_vision")
        text = str(raw or "").strip().lower()
        if text in self._VISION_TRUE_VALUES:
            return True
        if text in self._VISION_FALSE_VALUES:
            return False
        if text:
            logger.warning(f"Agent.supports_vision 取值无法识别: {raw!r}，本次改用自动判定")
        return None

    def supports_multimodal_input(self) -> bool:
        """当前模型能否接收图片输入（保守判定：宁可不发，也不乱发）。

        判定顺序：
        1. 配置覆盖：Agent.supports_vision=true/false 直接生效；
        2. 自动判定：provider 或 model 命中已知视觉组合才返回 True；
        3. 其余（含未配置模型、未知模型）返回 False，并打一条 warning，
           提示用户可用 Agent.supports_vision 手动打开。
        """
        override = self._vision_config_override()
        if override is not None:
            return override

        provider = str(self._model_provider or "").strip().lower()
        model = str(self._model_name or "").strip().lower()
        if provider in self._VISION_PROVIDERS:
            return True
        if provider == "deepseek":
            # deepseek 只有 VL / vision 系列支持图片输入
            if any(token in model for token in ("vl", "vision")):
                return True
        elif any(token in model for token in self._VISION_MODEL_TOKENS):
            return True

        self._warn_unknown_vision_model(provider, model)
        return False

    def _warn_unknown_vision_model(self, provider: str, model: str) -> None:
        """同一个 (provider, model) 只告警一次，避免每轮对话刷屏。"""
        key = (provider, model)
        if key in self._vision_warned:
            return
        self._vision_warned.add(key)
        logger.warning(
            "当前模型未登记为支持图片输入，已按纯文本模型处理"
            f"（provider={provider or '未配置'}, model={model or '未配置'}）；"
            "如需向模型发送截图/图片，请设置 Agent.supports_vision=true。"
        )

    def _contains_image_content(self, content: MessageContent) -> bool:
        if isinstance(content, str):
            return False
        if not isinstance(content, list):
            return False
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") == "image_file":
                return True
        return False


whimbox_agent = Agent()
