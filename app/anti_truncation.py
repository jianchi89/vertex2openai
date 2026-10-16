"""Vertex 抗截断传输（synthetic tool transport）。

移植自 ken050210/vertex-streaming-anti-truncation 的 src/anti-truncation.mjs、
src/vertex-text-stream.mjs 与 src/completion-integrity.mjs，用 Python 重写，
直接对接本代理的 Express 通道（多 Key 轮询、location 钉定、预填充兼容全部保留）。

机制（四步）
  1. 合成工具：注入一个随机命名的函数 ``router_emit_<24hex>``，只收一个 content
     字符串；并在消息末尾追加一条 user 指令，要求模型把整段回答放进这个参数。
  2. 强制走函数通道：``toolConfig.functionCallingConfig.mode = "ANY"`` +
     ``allowedFunctionNames = [该函数]``。
  3. 增量传输：流式时再打开 ``streamFunctionCallArguments = true``，Google 会用
     ``partialArgs``（``{jsonPath, stringValue, willContinue}``）分片吐参数。
  4. 还原：本端把分片重拼成合法 JSON，再用逐字符状态机只把 content 的值实时交付，
     对客户端表现为普通 ``delta.content``；合成 tool_calls 一律不外泄。

为什么必须自己发原生 REST
    google-genai SDK 会把 function call 参数反序列化成 dict，**分片边界在反序列化
    时就丢了**：每个分片都会被重新序列化成"看起来完整"的 JSON，第二片必然对不上，
    也就无法增量还原。所以本模块用 httpx 直接请求 ``:streamGenerateContent?alt=sse``，
    读原始 partialArgs 文本。

"抗截断"到底保住了什么
    当结束原因是 length / content_filter（模型被输出上限或安全过滤打断）时，容忍
    半截 JSON，把**已经收到的正文照常交付**——而不是当成"损坏的工具调用"整段丢弃。
    它不是重试，也不能变出模型没生成的内容。

触发方式（任一命中即启用）
    1. 模型名后缀（推荐：不用落盘状态，Render 免费层冷启动也不丢）
       ``gemini-4-argon-抗截断`` / ``gemini-4-argon-antitrunc``
    2. 控制台设置 ``anti_truncation``（默认关闭，可经 /api/settings 写入）
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
import time
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi.responses import JSONResponse, StreamingResponse
from google.genai import types

import config as app_config
from logger import stats
from message_processing import PrefillDeduper
from models import OpenAIRequest
from runtime_state import app_state
import model_capabilities as mc

# ---------------------------------------------------------------- 常量

MODEL_SUFFIXES = ("-antitruncation", "-antitrunc", "-抗截断")
TOOL_PREFIX = "router_emit_"
DEFAULT_API_BASE = "https://aiplatform.googleapis.com"
BODY_LIMIT = 8 * 1024 * 1024

#: 上游结束原因白名单：认不出来的一律当异常，不假装正常结束
VALID_FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls", "function_call"}
#: 这两种结束原因说明"上游主动截断"，此时半截 JSON 也要把已收正文交出去
CUT_SHORT_REASONS = {"length", "content_filter"}

NATIVE_TO_OPENAI_FINISH = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "RECITATION": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_calls",
    "UNEXPECTED_TOOL_CALL": "tool_calls",
    "TOOL_CODE": "tool_calls",
    "FUNCTION_CALL": "tool_calls",
}

INSTRUCTION = (
    "Transport format for this reply: call {tool} exactly once. Put the entire "
    "user-visible answer in its content string, preserving all requested formatting, "
    "markup, language and sections. The function only transports text and performs no "
    "external action."
)
CONTINUATION_INSTRUCTION = (
    "Follow the existing continuation context and extend the message being continued "
    "from its exact end. Output only the new text to append; do not repeat, summarize, "
    "rewrite or restart the existing text. Preserve its language, point of view, "
    "formatting and any unfinished sentence or markup. Do not add a preamble, "
    "continuation label, speaker prefix or enclosing quotes unless they belong to the "
    "continuation itself."
)

CONNECT_TIMEOUT = 20.0
READ_TIMEOUT = 600.0


def _log(msg: str) -> None:
    print(f"🛡️ [抗截断] {msg}")


# ---------------------------------------------------------------- 触发与准入


def split_trigger(model_name: str) -> Tuple[str, bool]:
    """识别模型名后缀。返回 (去掉后缀的基础模型名, 是否要求抗截断)。"""
    name = (model_name or "").strip()
    low = name.lower()
    for suffix in MODEL_SUFFIXES:
        if low.endswith(suffix.lower()):
            base = name[: len(name) - len(suffix)].strip().strip("-_ ")
            if base:
                return base, True
    return name, False


def settings_enabled() -> bool:
    """控制台开关（默认关）。"""
    try:
        return bool(app_state.get_setting("anti_truncation", False))
    except Exception:
        return False


def skip_reason(request_obj: OpenAIRequest, base_model_name: str, *,
                is_grounded_search: bool = False, is_image_model: bool = False) -> Optional[str]:
    """不满足条件就跳过（返回原因字符串）。

    与上游项目一致的跳过清单：请求里已经有工具/工具历史/结构化输出/多候选时
    绝不能包装，否则会和客户端自己的工具调用打架。
    """
    if getattr(request_obj, "tools", None):
        return "已有工具调用"
    if getattr(request_obj, "tool_choice", None) is not None:
        return "已指定 tool_choice"
    fmt = getattr(request_obj, "response_format", None)
    if isinstance(fmt, dict) and str(fmt.get("type") or "").lower() not in ("", "text"):
        return "结构化输出"
    n = getattr(request_obj, "n", None)
    if n is not None and n != 1:
        return "多候选（n>1）"
    if is_image_model:
        return "生图模型"
    if is_grounded_search:
        return "联网搜索"
    if not re.match(r"^gemini-[\w.\-]+$", base_model_name or ""):
        return "非 Gemini 文本模型"
    if re.search(r"image|embedding", base_model_name or "", re.I):
        return "生图/向量模型"
    for msg in request_obj.messages or []:
        if getattr(msg, "role", "") in ("tool", "function"):
            return "工具历史"
        if getattr(msg, "tool_calls", None):
            return "工具历史"
        if getattr(msg, "tool_call_id", None):
            return "工具历史"
    return None


def build_instruction(tool_name: str, continuation: bool = False) -> str:
    text = INSTRUCTION.format(tool=tool_name)
    if continuation:
        text += "\n" + CONTINUATION_INSTRUCTION
    return text


def new_tool_name() -> str:
    return TOOL_PREFIX + secrets.token_hex(12)


# ---------------------------------------------------------------- 模型列表变体
# 很多客户端（酒馆手机版、TauriTavern 等）的模型是**下拉选择**、不能手输模型名，
# 所以抗截断不能只靠"自己在模型名后面加后缀"。这里把变体直接并进 /v1/models，
# 客户端刷新模型列表后就能直接选。

#: 暴露给客户端的抗截断版本后缀
VARIANT_SUFFIX = "-抗截断"
#: 默认给哪些模型生成抗截断版本（改 config.DEFAULT_SETTINGS["anti_truncation_models"] 即可）
DEFAULT_VARIANT_MODELS = ["gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash"]


def variant_eligible(base_id: str) -> bool:
    """该模型是否适合生成抗截断版本。

    必须与 skip_reason 的准入规则保持一致，否则下拉里会出现"选了却静默回落普通通道"
    的选项：生图/向量模型排除；``-search`` 变体排除（联网搜索请求本来就会跳过抗截断）。

    生图判定一律走能力档案，**不能只靠名字**：``gemini-nano-banana-2.1`` 名字里既没有
    image 也没有 embedding（见 model_capabilities.get_profile 的注释），用名字匹配会漏判，
    结果就是在列表里给生图模型挂上一个永远不会生效的抗截断版本。
    """
    name = (base_id or "").strip()
    if not name or name.endswith("-search"):
        return False
    if not re.match(r"^gemini-[\w.\-]+$", name):
        return False
    try:
        if mc.get_profile(name).get("is_image"):
            return False
    except Exception:
        pass
    if re.search(r"image|embedding|nano-banana", name, re.I):
        return False
    return True


def variant_base_models(available: List[str]) -> List[str]:
    """要生成抗截断版本的基础模型名（结果一定在 available 里，避免下拉出现上游没有的模型）。

    设置 ``anti_truncation_models`` 的语义：
      - ``["*"]``            → 所有可用的文本 Gemini 模型
      - ``["a", "b"]``       → 指定模型（与可用列表求交集）
      - ``[]``               → 不生成任何抗截断版本
      - 缺省 / 非法值         → 用 DEFAULT_VARIANT_MODELS
    """
    avail = [str(m).strip() for m in (available or []) if variant_eligible(str(m))]
    try:
        configured = app_state.get_setting("anti_truncation_models", DEFAULT_VARIANT_MODELS)
    except Exception:
        configured = DEFAULT_VARIANT_MODELS
    if not isinstance(configured, (list, tuple)):
        configured = DEFAULT_VARIANT_MODELS
    wanted = [str(m).strip() for m in configured if str(m).strip()]
    if not wanted:
        return []
    if any(w == "*" or w.lower() == "all" for w in wanted):
        return avail
    return [m for m in avail if m in set(wanted)]


# ---------------------------------------------------------------- 增量 JSON 解码器


class ContentDecoder:
    """流式 JSON 识别器：只增量吐出顶层 ``content`` 字符串的值。

    与上游实现同构——不保存原始参数串，只保留 JSON 语法状态、当前 key（最多 32 字符）
    和一个待决的高代理项（surrogate）。转义与代理项各延后一拍，其余字符立刻交出。

    喂入的是**合法 JSON 文本片段**（分片已按 JSON 规则重新转义），因此跨片的
    转义/多字节字符不会破帧。

    与 JS 的一处有意差异：不成对的高代理项在交出时替换成 U+FFFD，避免把无法
    UTF-8 编码的孤立代理项写进 SSE（上游是直接拼进 JS 字符串）。
    """

    __slots__ = ("stack", "mode", "root_started", "complete", "invalid", "found",
                 "closed_content", "high", "_out", "string_kind", "key", "escape",
                 "hexbuf", "token", "expected")

    def __init__(self) -> None:
        self.stack: List[Dict[str, Any]] = []
        self.mode = "normal"
        self.root_started = False
        self.complete = False
        self.invalid = False
        self.found = False
        self.closed_content = False
        self.high = ""
        self._out: List[str] = []
        self.string_kind = ""
        self.key = ""
        self.escape = False
        self.hexbuf: Optional[str] = None
        self.token = ""
        self.expected = ""

    # ---- 内部 ----

    def _fail(self) -> None:
        self.invalid = True

    def _value_done(self) -> None:
        if self.stack:
            self.stack[-1]["state"] = "commaOrEnd"
        else:
            self.complete = True

    def _safe_high(self, ch: str) -> str:
        code = ord(ch)
        return ch if not (0xD800 <= code <= 0xDBFF) else "\ufffd"

    @staticmethod
    def _combine_surrogates(high: str, low: str) -> str:
        """把 \\uD83D\\uDE00 这样的代理项对还原成真实字符（😀）。

        JS 里两个 code unit 拼在一起就自动成对；Python 的 str 是码点序列，
        必须显式合并，否则会吐出两个孤立代理项（无法 UTF-8 编码）。
        """
        return chr(0x10000 + ((ord(high) - 0xD800) << 10) + (ord(low) - 0xDC00))

    def _emit_char(self, ch: str) -> None:
        if self.string_kind == "key":
            if len(self.key) < 32:
                self.key += ch
            return
        if self.string_kind != "content":
            return
        code = ord(ch)
        if self.high:
            pending = self.high
            self.high = ""
            if 0xDC00 <= code <= 0xDFFF:
                self._out.append(self._combine_surrogates(pending, ch))
                return
            # 高代理项后跟的不是低代理项：落单，替换成 U+FFFD 保证可编码
            self._out.append("\ufffd")
        if 0xD800 <= code <= 0xDBFF:
            self.high = ch
        else:
            self._out.append(ch)

    def _start_value(self, ch: str, content: bool) -> None:
        if content:
            if self.found or ch != '"':
                self._fail()
                return
            self.found = True
        if ch == '"':
            self.mode = "string"
            self.string_kind = "content" if content else "value"
            self.escape = False
            self.hexbuf = None
        elif ch in "{[":
            if len(self.stack) >= 64:
                self._fail()
                return
            self.stack.append({"type": ch, "state": "keyOrEnd" if ch == "{" else "valueOrEnd",
                               "key": ""})
        elif ch in "tfn":
            self.mode = "literal"
            self.token = ch
            self.expected = "true" if ch == "t" else "false" if ch == "f" else "null"
        elif ch == "-" or ch.isdigit():
            self.mode = "number"
            self.token = ch
        else:
            self._fail()

    # ---- 对外 ----

    def feed(self, text: str) -> str:
        """喂入一段（可能不完整的）JSON 文本，返回本次能确定交付的 content 增量。"""
        self._out = []
        i = 0
        n = len(text)
        while i < n and not self.invalid:
            ch = text[i]

            if self.mode == "string":
                if self.hexbuf is not None:
                    if ch not in "0123456789abcdefABCDEF":
                        self._fail()
                        i += 1
                        continue
                    self.hexbuf += ch
                    if len(self.hexbuf) == 4:
                        self._emit_char(chr(int(self.hexbuf, 16)))
                        self.hexbuf = None
                elif self.escape:
                    self.escape = False
                    if ch == "u":
                        self.hexbuf = ""
                    else:
                        escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                                   "n": "\n", "r": "\r", "t": "\t"}
                        if ch not in escapes:
                            self._fail()
                        else:
                            self._emit_char(escapes[ch])
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.mode = "normal"
                    if self.string_kind == "key":
                        if self.stack:
                            self.stack[-1]["key"] = self.key
                            self.stack[-1]["state"] = "colon"
                    else:
                        if self.string_kind == "content":
                            if self.high:
                                self._out.append(self._safe_high(self.high))
                                self.high = ""
                            self.closed_content = True
                        self._value_done()
                elif ord(ch) < 32:
                    self._fail()
                else:
                    self._emit_char(ch)
                i += 1
                continue

            if self.mode == "literal":
                self.token += ch
                if not self.expected.startswith(self.token):
                    self._fail()
                elif self.token == self.expected:
                    self.mode = "normal"
                    self._value_done()
                i += 1
                continue

            if self.mode == "number":
                if ch in "0123456789eE+-.":
                    self.token += ch
                    if len(self.token) > 128:
                        self._fail()
                    i += 1
                    continue
                if not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", self.token):
                    self._fail()
                    i += 1
                    continue
                self.mode = "normal"
                self._value_done()
                continue  # 当前字符不消耗，交给正常状态处理

            if ch in " \t\r\n":
                i += 1
                continue
            if self.complete:
                self._fail()
                i += 1
                continue
            if not self.root_started:
                self.root_started = True
                if ch != "{":
                    self._fail()
                else:
                    self.stack.append({"type": "{", "state": "keyOrEnd", "key": ""})
                i += 1
                continue
            frame = self.stack[-1] if self.stack else None
            if frame is None:
                self._fail()
                i += 1
                continue
            if frame["type"] == "{" and frame["state"] in ("keyOrEnd", "key"):
                if ch == "}" and frame["state"] == "keyOrEnd":
                    self.stack.pop()
                    self._value_done()
                elif ch == '"':
                    self.mode = "string"
                    self.string_kind = "key"
                    self.key = ""
                    self.escape = False
                    self.hexbuf = None
                else:
                    self._fail()
            elif frame["state"] == "colon":
                if ch != ":":
                    self._fail()
                else:
                    frame["state"] = "value"
            elif frame["state"] == "commaOrEnd":
                closer = "}" if frame["type"] == "{" else "]"
                if ch == closer:
                    self.stack.pop()
                    self._value_done()
                elif ch == ",":
                    frame["state"] = "key" if frame["type"] == "{" else "value"
                else:
                    self._fail()
            elif frame["type"] == "[" and frame["state"] == "valueOrEnd" and ch == "]":
                self.stack.pop()
                self._value_done()
            else:
                is_content = (len(self.stack) == 1 and frame["type"] == "{"
                              and frame["key"] == "content")
                self._start_value(ch, is_content)
            i += 1

        return "".join(self._out)

    @property
    def valid(self) -> bool:
        return self.complete and self.found and self.closed_content and not self.invalid


def decode_full_arguments(args_json: str, finish_reason: Optional[str]) -> Tuple[str, bool]:
    """非流式：解析完整的 arguments JSON。返回 (正文, 是否完整)。"""
    decoder = ContentDecoder()
    text = decoder.feed(args_json)
    if decoder.invalid:
        return "", False
    if not decoder.valid and (finish_reason or "") not in CUT_SHORT_REASONS:
        return text, False
    return text, True


# ---------------------------------------------------------------- REST 请求体构造


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


#: 这些键下面装的是"用户自定义的 JSON Schema"，键名属于用户数据，绝不能被 camelCase 改写
_SCHEMA_KEYS = {"response_schema", "response_json_schema", "parameters", "parameters_json_schema",
                "properties", "defs", "$defs"}


def _rest_value(value: Any, *, camel: bool = True) -> Any:
    """把 SDK 类型/字典统一转成原生 REST 可用的 JSON 值（camelCase + bytes→base64）。

    ``camel=False`` 时只做类型转换，不改键名——用于 JSON Schema 这类键名属于用户数据的字典。
    """
    if isinstance(value, Enum):
        return _rest_value(value.value, camel=camel)
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k)
            child_camel = camel and key not in _SCHEMA_KEYS
            out[_camel(key) if camel else key] = _rest_value(v, camel=child_camel)
        return out
    if isinstance(value, (list, tuple)):
        return [_rest_value(v, camel=camel) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _rest_value(value.model_dump(exclude_none=True, by_alias=False), camel=camel)
        except Exception:
            return _rest_value(str(value), camel=camel)
    return value


def build_rest_body(contents: List[Any], gen_config_dict: Dict[str, Any], tool_name: str,
                    *, stream: bool, continuation: bool = False) -> Dict[str, Any]:
    """把 SDK 的 contents + 生成配置翻译成原生 REST 请求体。

    tools / toolConfig / safetySettings / systemInstruction 在 REST 里是**顶层字段**，
    不能留在 generationConfig 里（SDK 才收在一个 config 对象里）。
    """
    config = dict(gen_config_dict or {})

    system_instruction = config.pop("system_instruction", None)
    safety_settings = config.pop("safety_settings", None)
    config.pop("tools", None)
    config.pop("tool_config", None)

    body: Dict[str, Any] = {
        "contents": _rest_value(contents),
        "generationConfig": _rest_value(config),
        "tools": [{
            "functionDeclarations": [{
                "name": tool_name,
                "description": ("Deliver only the new continuation text to append to the existing "
                                "message." if continuation else
                                "Deliver the complete user-visible reply as text."),
                "parametersJsonSchema": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                    "additionalProperties": False,
                },
            }],
        }],
        "toolConfig": {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [tool_name],
            },
        },
    }

    if stream:
        # 预览字段：让 Google 用 partialArgs 增量吐函数参数
        body["toolConfig"]["functionCallingConfig"]["streamFunctionCallArguments"] = True

    if isinstance(system_instruction, str) and system_instruction.strip():
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    elif system_instruction:
        body["systemInstruction"] = _rest_value(system_instruction)

    if safety_settings:
        body["safetySettings"] = _rest_value(safety_settings)

    return body


def build_url(model_path: str, *, stream: bool) -> str:
    """拼原生端点。model_path 可以是裸模型名，也可以是完整 projects/.../models/... 路径。

    base URL 优先取环境变量 VERTEX_BASE_URL（本模块自带的逃生口，Node 版没有这个能力），
    否则用 Google 官方端点——Express Key 模式与官方 SDK 一致走全局端点，区域由后端路由；
    填了 express_location 时用带项目/区域的完整路径钉定（复用反代既有逻辑）。
    """
    base = (app_config.VERTEX_BASE_URL or DEFAULT_API_BASE).rstrip("/")
    if not base.endswith(("/v1", "/v1beta")):
        base += "/v1"
    method = "streamGenerateContent" if stream else "generateContent"
    if model_path.startswith(("projects/", "publishers/")):
        path = model_path
    elif model_path.startswith("models/"):
        path = "publishers/google/" + model_path
    else:
        path = f"publishers/google/models/{model_path}"
    url = f"{base}/{path}:{method}"
    if stream:
        url += "?alt=sse"
    return url


# ---------------------------------------------------------------- SSE 输出


def _sse(payload: Any) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


class _ChunkWriter:
    """按本代理既有信封输出 OpenAI 流式分片（与 convert_chunk_to_openai 同形）。"""

    def __init__(self, response_id: str, model: str) -> None:
        self.response_id = response_id
        self.model = model
        self.created = int(time.time())

    def chunk(self, delta: Dict[str, Any], finish_reason: Optional[str] = None,
              native_reason: Optional[str] = None, index: int = 0) -> str:
        choice: Dict[str, Any] = {"index": index, "delta": delta, "finish_reason": finish_reason}
        if native_reason:
            choice["native_finish_reason"] = native_reason
        return _sse({"id": self.response_id, "object": "chat.completion.chunk",
                     "created": self.created, "model": self.model, "choices": [choice]})

    def error(self, code: str, message: str) -> str:
        return _sse({"error": {"code": code, "message": message, "type": "upstream_error"}})


class _StreamFail(Exception):
    """流内协议异常（分片损坏、结束原因缺失、结束之后还有数据等）。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class NativeStreamRestorer:
    """原生 SSE → OpenAI SSE，并顺手把合成工具调用还原成普通正文。

    上游项目分两段做（vertex-text-stream.mjs 先拼成 OpenAI 工具参数，再由
    anti-truncation.mjs 还原）；这里既然两段都归自己管，就合成一遍，少一次
    重新转义与再解析，行为等价：

      - ``partialArgs`` 分片 → 逐片重转义后喂 ContentDecoder → 立刻吐 delta.content
      - ``args``（完整参数，非流式分片路径）→ 同样经解码器
      - ``part.text`` → 普通 delta.content，但只在"正文通道"未被合成工具占用时交付
      - ``part.thought == true`` → reasoning_content（不参与通道竞争）
    """

    def __init__(self, tool_name: str, writer: _ChunkWriter, *, include_usage: bool,
                 deduper: Optional[PrefillDeduper] = None) -> None:
        self.tool_name = tool_name
        self.writer = writer
        self.include_usage = include_usage
        self.deduper = deduper
        self.decoder = ContentDecoder()

        self.started = False          # 合成工具调用已开始
        self.string_started = False   # content 字符串已开头
        self.string_closed = False    # content 字符串已闭合
        self.call_closed = False      # 合成工具调用已闭合
        self.full_args = False        # 收到的是完整 args（非分片）
        self.terminal = False         # 已经见过结束原因
        self.failed = False
        self.channel: Optional[str] = None   # "plain" | "synthetic"
        self.restored = False
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.got_any = False
        self.delivered_chars = 0      # 已交付给客户端的正文字数
        self.lenient_closures = 0     # 主动收口次数（上游没给分片结束标记时）
        self.recent: List[Dict[str, Any]] = []   # 最近几个分片的**形状**（不含正文）

    # ---- 诊断信息（只含元数据，不含正文） ----

    def _note_part(self, fc: Dict[str, Any]) -> None:
        """记下分片形状，用来核对 Google 实际给的字段与 willContinue 语义。

        只记键名、willContinue 和片段长度——**不记 stringValue**，避免把正文带进日志。
        """
        partials = fc.get("partialArgs")
        shape: Dict[str, Any] = {
            "keys": sorted(fc.keys()),
            "willContinue": fc.get("willContinue"),
            "args": "object" if isinstance(fc.get("args"), dict) else None,
            "partial": None if not isinstance(partials, list) else [
                {"jsonPath": p.get("jsonPath"), "willContinue": p.get("willContinue"),
                 "len": len(p.get("stringValue") or "")}
                for p in partials[:8] if isinstance(p, dict)
            ],
        }
        self.recent.append(shape)
        del self.recent[:-3]

    def shape_summary(self) -> str:
        return json.dumps(self.recent, ensure_ascii=False) if self.recent else "(无分片)"

    def progress(self) -> str:
        return (f"started={self.started} str={self.string_started}/{self.string_closed} "
                f"call={self.call_closed} full_args={self.full_args} "
                f"已交付={self.delivered_chars}字 收口={self.lenient_closures}次")

    # ---- 正文交付（预填充去重统一在这一处） ----

    def _text_delta(self, text: str) -> str:
        if not text:
            return ""
        if self.deduper is not None:
            text = self.deduper.feed(text)
            if not text:
                return ""
        self.got_any = True
        self.delivered_chars += len(text)
        return self.writer.chunk({"content": text})

    def flush_deduper(self) -> str:
        if self.deduper is not None and not self.deduper.done:
            tail = self.deduper.flush()
            if tail:
                self.got_any = True
                return self.writer.chunk({"content": tail})
        return ""

    # ---- 协议处理 ----

    def _call(self, fc: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        self._note_part(fc)
        name = fc.get("name")
        if self.call_closed or (name is not None and name != self.tool_name):
            raise _StreamFail("unexpected_call")
        if not self.started:
            if name != self.tool_name:
                raise _StreamFail("missing_name")
            self.started = True

        if fc.get("args") is not None:
            if self.string_started or self.full_args:
                raise _StreamFail("invalid_args")
            args = fc.get("args")
            if not isinstance(args, dict) or not isinstance(args.get("content"), str):
                raise _StreamFail("invalid_args")
            self.full_args = True
            self.string_closed = True
            text = self.decoder.feed(json.dumps(args, ensure_ascii=False))
            if self.decoder.invalid:
                raise _StreamFail("invalid_args")
            self._claim_synthetic(text, out)

        partials = fc.get("partialArgs")
        if partials is not None and not isinstance(partials, list):
            raise _StreamFail("invalid_partial_args")
        for part in partials or []:
            if self.full_args or self.string_closed:
                raise _StreamFail("invalid_partial_args")
            if part.get("jsonPath") != "$.content" or not isinstance(part.get("stringValue"), str):
                raise _StreamFail("invalid_partial_args")
            if not self.string_started:
                self.decoder.feed('{"content":"')
                self.string_started = True
            value = part.get("stringValue") or ""
            if value:
                fragment = json.dumps(value, ensure_ascii=False)[1:-1]
                text = self.decoder.feed(fragment)
                if self.decoder.invalid:
                    raise _StreamFail("invalid_partial_args")
                self._claim_synthetic(text, out)
            if part.get("willContinue") is not True:
                self.decoder.feed('"')
                self.string_closed = True

        if fc.get("willContinue") is not True:
            if not self.string_closed:
                self._close_string_leniently("functionCall 声明的参数结束")
            if not self.full_args:
                self.decoder.feed("}")
            self.call_closed = True
        return out

    def _close_string_leniently(self, why: str) -> None:
        """上游声明"参数结束"、却没给分片级结束标记时，主动把 content 字符串收口。

        背景（2026-10 用 gemini-4-argon 实测）：``partialArgs[].willContinue`` 在
        最后一个分片上**仍可能是 true**，参数是否结束只体现在 ``functionCall.willContinue``
        或 ``finishReason`` 上。上游 Node 项目按 gemini-3.x 的语义写死了"最后一片必须
        带 willContinue=false"，于是把一次**正文已经完整交付**的回复错判成
        ``incomplete_args``——客户端看到的是"内容全对，最后弹一个失败"。

        为什么主动收口是安全的：
          1. 本方法只在**传输已经确认结束**的上下文里被调用——HTTP 流正常读完并收到了
             结束原因或 functionCall 结束标记。TCP 不会静默跳过中间字节：要么数据完整
             到达，要么连接报错（那条路径会被 stream_chat 标成 finish_reason=length）；
          2. 收口之后 decoder 仍要过完整结构校验（见 _finish），真正被截断、转义残缺的
             流依然会判失败。

        分片一个都没收到时不收口：那是真的异常，不能让空回复假装成功。
        """
        if not (self.string_started and not self.full_args):
            raise _StreamFail("incomplete_args")
        self.decoder.feed('"')
        self.string_closed = True
        self.lenient_closures += 1
        _log(f"⚠️ 上游未给分片结束标记（{why}），已主动收口：正文已交付 "
             f"{self.delivered_chars} 字，收口后仍会做结构校验。"
             f"分片形状：{self.shape_summary()}")

    def _claim_synthetic(self, text: str, out: List[str]) -> None:
        if not text:
            return
        if self.channel is None:
            self.channel = "synthetic"
        if self.channel != "synthetic":
            return  # 正文已经先走了普通 text 通道，丢弃重复的合成通道内容
        self.restored = True
        chunk = self._text_delta(text)
        if chunk:
            out.append(chunk)

    def _finish(self, reason: Optional[str], *, block: bool = False) -> List[str]:
        if reason is not None and reason not in NATIVE_TO_OPENAI_FINISH:
            self.failed = True
            self.terminal = True
            raise _StreamFail("invalid_finish_reason")
        if self.terminal:
            self.failed = True
            raise _StreamFail("duplicate_finish")
        mapped = NATIVE_TO_OPENAI_FINISH.get("SAFETY" if block else (reason or ""), "stop")
        if self.started and not self.call_closed and mapped not in CUT_SHORT_REASONS:
            # 正文早已逐片交付，只是上游没给收口标记（见 _close_string_leniently）：
            # 主动收口后交给下面的结构校验裁决，而不是直接判失败。
            self._close_string_leniently(f"finishReason={reason}")
            if not self.full_args:
                self.decoder.feed("}")
            self.call_closed = True
        if self.decoder.found and not self.decoder.valid and mapped not in CUT_SHORT_REASONS:
            self.failed = True
            raise _StreamFail("incomplete_arguments")
        if mapped == "tool_calls" and self.restored and not self.failed:
            # 唯一的工具调用就是合成通道，而且正文已经还原出来了：对客户端而言这是一次
            # 正常文本回复。若照搬上游结束原因，客户端会看到 finish_reason=tool_calls
            # 却没有任何 tool_calls，容易触发它自己的工具分支。
            mapped = "stop"
        self.terminal = True
        self.finish_reason = mapped
        return [self.writer.chunk({}, finish_reason=mapped, native_reason=reason)]

    def process(self, payload: Dict[str, Any]) -> List[str]:
        """处理一条原生 SSE data 负载，返回要立刻下发的 SSE 片段。"""
        out: List[str] = []
        if isinstance(payload.get("usageMetadata"), dict):
            self.usage = _rest_value(payload["usageMetadata"])
        candidates = payload.get("candidates") or []
        if len(candidates) > 1:
            raise _StreamFail("multiple_candidates")
        candidate = candidates[0] if candidates else None
        for part in ((candidate or {}).get("content") or {}).get("parts") or []:
            if self.terminal and (part.get("functionCall") or part.get("text")):
                raise _StreamFail("data_after_finish")
            if part.get("functionCall"):
                out.extend(self._call(part["functionCall"]))
            if isinstance(part.get("text"), str) and part["text"]:
                if part.get("thought") is True:
                    out.append(self.writer.chunk({"reasoning_content": part["text"]}))
                elif self.channel in (None, "plain"):
                    self.channel = "plain"
                    chunk = self._text_delta(part["text"])
                    if chunk:
                        out.append(chunk)
        block_reason = (payload.get("promptFeedback") or {}).get("blockReason") if not candidates else None
        reason = (candidate or {}).get("finishReason") or block_reason
        if reason:
            out.extend(self._finish(reason, block=bool(block_reason)))
        return out


# ---------------------------------------------------------------- usage / 收尾


def _usage_dict(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not isinstance(usage, dict):
        return None
    p = int(usage.get("promptTokenCount") or 0)
    c = int(usage.get("candidatesTokenCount") or 0)
    t = int(usage.get("totalTokenCount") or (p + c))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


def _extract_native_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or "")[:400]
    return ""


def _parse_error_body(raw: bytes) -> str:
    """从错误响应体里取上游 message（不回显密钥/提示词，只取 Google 的 error.message）。"""
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or "")[:400]
    if isinstance(data, dict) and isinstance(data.get("message"), str):
        return data["message"][:400]
    return ""


def _retry_settings() -> Tuple[int, float]:
    from api_helpers import get_retry_settings
    return get_retry_settings()


def _wants_usage(request_obj: OpenAIRequest) -> bool:
    from api_helpers import wants_usage
    return wants_usage(request_obj)


# ---------------------------------------------------------------- 上游调用


def _headers(api_key: str) -> Dict[str, str]:
    return {"content-type": "application/json", "x-goog-api-key": api_key}


async def stream_chat(*, api_key: str, model_path: str, body: Dict[str, Any],
                      request_obj: OpenAIRequest, tool_name: str,
                      prefill_text: str = "", display_model: str = "") -> AsyncIterator[str]:
    """流式抗截断：产生与既有通道完全同形的 SSE 片段，最后以 [DONE] 收尾。"""
    url = build_url(model_path, stream=True)
    response_id = f"chatcmpl-antitrunc-{int(time.time())}"
    writer = _ChunkWriter(response_id, display_model or request_obj.model)
    include_usage = _wants_usage(request_obj)
    deduper = PrefillDeduper(prefill_text) if (prefill_text and (request_obj.n or 1) == 1) else None
    retry_max, backoff = _retry_settings()
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)

    _log(f"流式请求上游：{url.split('?')[0]}（合成函数 {tool_name}）")
    yield ": keep-alive\n\n"

    restorer = NativeStreamRestorer(tool_name, writer, include_usage=include_usage,
                                    deduper=deduper)
    if prefill_text:
        # 预填充文本先交给客户端，模型若复述开头由去重器裁掉（与普通通道一致）
        yield writer.chunk({"role": "assistant", "content": prefill_text})

    async def _one_pass() -> AsyncIterator[str]:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        try:
            async with client.stream("POST", url, json=body, headers=_headers(api_key)) as resp:
                if resp.status_code >= 400:
                    detail = ""
                    try:
                        raw = await resp.aread()
                        detail = _parse_error_body(raw)
                    except Exception:
                        pass
                    raise _UpstreamHTTP(resp.status_code, detail or f"HTTP {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        raise _StreamFail("invalid_sse")
                    if isinstance(payload, dict) and payload.get("error"):
                        raise _UpstreamHTTP(
                            int(payload["error"].get("code") or 502),
                            str(payload["error"].get("message") or "")[:400])
                    for piece in restorer.process(payload):
                        yield piece
        finally:
            await client.aclose()

    attempt = 0
    yielded_any = False   # 是否已经向客户端吐出过任何片段（含思考）；吐过就不再重试，避免重复正文
    while True:
        try:
            async for piece in _one_pass():
                yielded_any = True
                yield piece
            break
        except _StreamFail as exc:
            _log(f"❌ 协议异常 {exc.code}，已停止本轮（不再重发，避免正文重复）。"
                 f"状态：{restorer.progress()}；最近分片形状：{restorer.shape_summary()}")
            yield writer.error(exc.code,
                               f"抗截断传输失败（{exc.code}）；已交付 "
                               f"{restorer.delivered_chars} 字。")
            yield "data: [DONE]\n\n"
            return
        except _UpstreamHTTP as exc:
            if exc.status in (429, 500, 502, 503, 504) and attempt < retry_max and not yielded_any:
                attempt += 1
                stats.add_retry()
                _log(f"⚠️ 上游 HTTP {exc.status}，第 {attempt} 次退避重试（等待 {backoff} 秒）。")
                await asyncio.sleep(max(0.0, backoff))
                continue
            _log(f"❌ 上游 HTTP {exc.status}：{exc.detail[:200]}")
            yield writer.error("upstream_http_error",
                               f"上游返回 HTTP {exc.status}：{exc.detail[:200]}")
            yield "data: [DONE]\n\n"
            return
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            if attempt < retry_max and not yielded_any:
                attempt += 1
                stats.add_retry()
                _log(f"⚠️ 上游连接异常（{type(exc).__name__}），第 {attempt} 次退避重试。")
                await asyncio.sleep(max(0.0, backoff))
                continue
            _log(f"❌ 上游连接异常：{type(exc).__name__}: {exc}")
            # 已经输出过正文：不要假装正常结束（这正是"后半段整段丢失"却看不出来的成因）
            if yielded_any:
                yield writer.chunk({}, finish_reason="length", native_reason="UPSTREAM_ABORTED")
            else:
                yield writer.error("upstream_unreachable",
                                   f"连不上上游：{type(exc).__name__}")
            yield "data: [DONE]\n\n"
            return

    # ---- 正常读完：补去重尾巴、usage、结束标记 ----
    tail = restorer.flush_deduper()
    if tail:
        yield tail
    if not restorer.terminal:
        # 流在没有结束原因的情况下就断了：明确标记为截断，不要静默收尾
        _log("❌ 上游流未给出结束原因（可能被中途掐断），已按截断标记 length。")
        yield writer.chunk({}, finish_reason="length", native_reason="STREAM_INTERRUPTED")
        restorer.finish_reason = "length"
    usage = _usage_dict(restorer.usage)
    if usage and (usage["prompt_tokens"] or usage["completion_tokens"]):
        stats.add_tokens(usage["prompt_tokens"], usage["completion_tokens"])
        _log(f"💰 tokens 提示词={usage['prompt_tokens']} 生成={usage['completion_tokens']}")
    if include_usage:
        yield _sse({"id": response_id, "object": "chat.completion.chunk", "created": writer.created,
                    "model": writer.model, "choices": [], "usage": usage or {
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
    yield "data: [DONE]\n\n"
    _log(f"✅ 完成：还原={'是' if restorer.restored else '否'} "
         f"结束原因={restorer.finish_reason}")


class _UpstreamHTTP(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


async def complete_chat(*, api_key: str, model_path: str, body: Dict[str, Any],
                        request_obj: OpenAIRequest, tool_name: str,
                        prefill_text: str = "", display_model: str = "") -> JSONResponse:
    """非流式抗截断：等上游完整返回后还原成普通正文。"""
    url = build_url(model_path, stream=False)
    response_id = f"chatcmpl-antitrunc-{int(time.time())}"
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    _log(f"非流式请求上游：{url}（合成函数 {tool_name}）")

    last_error = "未知错误"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        try:
            resp = await client.post(url, json=body, headers=_headers(api_key))
        except httpx.HTTPError as exc:
            return _error_response(502, f"连不上上游：{type(exc).__name__}: {exc}",
                                   "upstream_unreachable")
        if resp.status_code >= 400:
            detail = _extract_native_error(resp) or f"HTTP {resp.status_code}"
            last_error = detail
            return _error_response(resp.status_code if 400 <= resp.status_code < 600 else 502,
                                   detail, "upstream_http_error")
        try:
            payload = resp.json()
        except Exception:
            return _error_response(502, "上游返回的不是合法 JSON。", "upstream_protocol_error")

    candidates = payload.get("candidates") or []
    if not candidates:
        block = (payload.get("promptFeedback") or {}).get("blockReason")
        if block:
            return _error_response(400, f"提示词被上游拦截：{block}", "content_filter")
        return _error_response(502, "上游没有返回任何候选。", "empty_completion")

    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    native_reason = candidates[0].get("finishReason")
    mapped = NATIVE_TO_OPENAI_FINISH.get(native_reason or "", "stop")
    if native_reason and native_reason not in NATIVE_TO_OPENAI_FINISH:
        return _error_response(502, f"认不出的上游结束原因：{native_reason}",
                               "invalid_finish_reason")

    text = ""
    reasoning = ""
    synthetic_text: Optional[str] = None
    for part in parts:
        if isinstance(part.get("text"), str) and part["text"]:
            if part.get("thought") is True:
                reasoning += part["text"]
            else:
                text += part["text"]
        fc = part.get("functionCall")
        if fc:
            if fc.get("name") != tool_name:
                return _error_response(502, "上游返回了非合成的工具调用。", "unexpected_tool_call")
            args = fc.get("args")
            if not isinstance(args, dict):
                return _error_response(502, "上游工具参数不是对象。", "invalid_arguments")
            decoded, ok = decode_full_arguments(
                json.dumps(args, ensure_ascii=False), mapped)
            if not ok:
                return _error_response(502, "抗截断参数不完整（半截 JSON）。",
                                       "incomplete_arguments")
            synthetic_text = decoded
            break

    # 合成通道是正文的权威来源：它一旦有内容就顶掉同名普通 text，避免两路重复交付
    if synthetic_text is not None:
        text = synthetic_text

    if not text and not reasoning:
        return _error_response(502, "上游回复为空（只有工具调用外壳）。", "empty_completion")
    if prefill_text:
        text = prefill_text + text

    usage = _usage_dict(_rest_value(payload.get("usageMetadata"))) if payload.get("usageMetadata") else None
    if usage:
        stats.add_tokens(usage["prompt_tokens"], usage["completion_tokens"])

    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = reasoning
    result = {"id": response_id, "object": "chat.completion", "created": int(time.time()),
              "model": display_model or request_obj.model,
              "choices": [{"index": 0, "message": message,
                           "finish_reason": mapped if native_reason else "stop"}]}
    if usage:
        result["usage"] = usage
    _log(f"✅ 非流式完成：结束原因={result['choices'][0]['finish_reason']} 正文 {len(text)} 字")
    return JSONResponse(content=result)


def _error_response(status: int, message: str, code: str) -> JSONResponse:
    from api_helpers import create_openai_error_response
    return JSONResponse(status_code=status,
                        content=create_openai_error_response(status, message, code))


async def handle(*, request_obj: OpenAIRequest, base_model_name: str, tool_name: str,
                 contents: List[Any], gen_config_dict: Dict[str, Any], api_key: str,
                 model_path: str, prefill_text: str = "",
                 continuation: bool = False, display_model: str = "") -> Any:
    """入口：构造请求体并返回 StreamingResponse / JSONResponse。"""
    body = build_rest_body(contents, gen_config_dict, tool_name, stream=bool(request_obj.stream),
                           continuation=continuation)
    if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) > BODY_LIMIT:
        return _error_response(413, "请求体超过 8 MiB，抗截断模式拒绝发送。", "request_too_large")
    if request_obj.stream:
        return StreamingResponse(
            stream_chat(api_key=api_key, model_path=model_path, body=body,
                        request_obj=request_obj, tool_name=tool_name,
                        prefill_text=prefill_text, display_model=display_model),
            media_type="text/event-stream")
    return await complete_chat(api_key=api_key, model_path=model_path, body=body,
                               request_obj=request_obj, tool_name=tool_name,
                               prefill_text=prefill_text, display_model=display_model)
