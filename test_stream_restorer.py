"""NativeStreamRestorer 验证：模拟上游原生 SSE 负载，核对还原行为。

同样只抽取纯逻辑类（加上 _rest_value 与 _camel），不需要 fastapi/google-genai。
"""
import ast
import base64
import json
import re
import sys
import time
import typing
from enum import Enum
from pathlib import Path

SRC = Path(r"D:\DeepSeekH客户端\游戏安装\_v2o\vertex2openai-main\app\anti_truncation.py")
tree = ast.parse(SRC.read_text(encoding="utf-8"))
WANT_CLASSES = {"ContentDecoder", "_ChunkWriter", "_StreamFail", "NativeStreamRestorer"}
WANT_FUNCS = {"_camel", "_rest_value", "_sse"}
WANT_CONSTS = {"NATIVE_TO_OPENAI_FINISH", "CUT_SHORT_REASONS", "VALID_FINISH_REASONS",
               "_SCHEMA_KEYS"}
body = [n for n in tree.body
        if (isinstance(n, ast.ClassDef) and n.name in WANT_CLASSES)
        or (isinstance(n, ast.FunctionDef) and n.name in WANT_FUNCS)
        or (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in WANT_CONSTS for t in n.targets))]
ns = {"re": re, "json": json, "time": time, "base64": base64, "Enum": Enum,
      "Any": typing.Any, "Dict": typing.Dict, "List": typing.List,
      "Optional": typing.Optional, "Tuple": typing.Tuple,
      "AsyncIterator": typing.AsyncIterator, "PrefillDeduper": object}
exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
Restorer, Writer, StreamFail = ns["NativeStreamRestorer"], ns["_ChunkWriter"], ns["_StreamFail"]

TOOL = "router_emit_abcdefabcdefabcdefabcdef"
fails = []


def check(label, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + f"  {label}" + (f"   {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(label)


def contents(chunks):
    out = ""
    for raw in chunks:
        payload = json.loads(raw[6:])
        for ch in payload.get("choices") or []:
            if ch.get("delta", {}).get("content"):
                out += ch["delta"]["content"]
    return out


def finish_of(chunks):
    for raw in chunks:
        payload = json.loads(raw[6:])
        for ch in payload.get("choices") or []:
            if ch.get("finish_reason"):
                return ch["finish_reason"]
    return None


def reasoning_of(chunks):
    out = ""
    for raw in chunks:
        payload = json.loads(raw[6:])
        for ch in payload.get("choices") or []:
            rc = ch.get("delta", {}).get("reasoning_content")
            if rc:
                out += rc
    return out


def new_restorer():
    writer = Writer("chatcmpl-test", "gemini-4-argon")
    return Restorer(TOOL, writer, include_usage=False, deduper=None)


def native_partial(text, will_continue=True, name=TOOL):
    return {"candidates": [{"content": {"parts": [{"functionCall": {
        "name": name,
        "partialArgs": [{"jsonPath": "$.content", "stringValue": text,
                         "willContinue": will_continue}],
        "willContinue": will_continue}}]}}]}


print("1) partialArgs 分片 → 增量 delta.content，且不泄漏合成 tool_calls")
r = new_restorer()
chunks = []
for frag in ["你", "好", "，世界"]:
    chunks += r.process(native_partial(frag))
chunks += r.process(native_partial("", will_continue=False, name=None))
chunks += r.process({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]})
check("正文还原", contents(chunks) == "你好，世界", f"got={contents(chunks)!r}")
check("结束原因 stop", finish_of(chunks) == "stop")
check("首片就交付（不是攒到最后）", contents(chunks[:1]) == "你", f"got={contents(chunks[:1])!r}")
check("无 tool_calls 泄漏", all("tool_calls" not in raw for raw in chunks))

print("2) 思考部件走 reasoning_content，正文走 content")
r = new_restorer()
chunks = r.process({"candidates": [{"content": {"parts": [{"text": "内部推理", "thought": True}]}}]})
chunks += r.process(native_partial("正文", will_continue=False))
chunks += r.process({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]})
check("reasoning_content", reasoning_of(chunks) == "内部推理", f"got={reasoning_of(chunks)!r}")
check("content 只有正文", contents(chunks) == "正文", f"got={contents(chunks)!r}")

print("3) 普通 text 通道先到 → 合成通道内容被抑制（只保留一个通道）")
r = new_restorer()
chunks = r.process({"candidates": [{"content": {"parts": [{"text": "普通"}]}}]})
chunks += r.process(native_partial("合成", will_continue=False))
chunks += r.process({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]})
check("保留普通通道", contents(chunks) == "普通", f"got={contents(chunks)!r}")

print("4) 合成通道先到 → 普通 text 被抑制")
r = new_restorer()
chunks = r.process(native_partial("合成", will_continue=False))
chunks += r.process({"candidates": [{"content": {"parts": [{"text": "普通"}]}}]})
chunks += r.process({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]})
check("保留合成通道", contents(chunks) == "合成", f"got={contents(chunks)!r}")

print("5) MAX_TOKENS 截断：半截 JSON 也要把已收正文交出去（抗截断的核心价值）")
r = new_restorer()
chunks = r.process(native_partial("前半段正文", will_continue=True))
chunks += r.process({"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})
check("半截正文已交付", contents(chunks) == "前半段正文", f"got={contents(chunks)!r}")
check("结束原因 length", finish_of(chunks) == "length", f"got={finish_of(chunks)!r}")

print("6) STOP 却没收口 → 必须判失败（不能假装成功）")
for payload, label in [
    ({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}, "合成调用未闭合"),
]:
    r = new_restorer()
    try:
        r.process(native_partial("正文", will_continue=True))
        r.process(payload)
        check(label + " → 抛错", False, "没有抛错")
    except StreamFail as exc:
        check(label + f" → {exc.code}", exc.code in ("incomplete_args", "incomplete_arguments"))

print("7) 结束之后还有数据 → data_after_finish")
r = new_restorer()
r.process(native_partial("正文", will_continue=False))
r.process({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]})
try:
    r.process(native_partial("再来"))
    check("data_after_finish → 抛错", False, "没有抛错")
except StreamFail as exc:
    check(f"data_after_finish → {exc.code}", exc.code == "data_after_finish")

print("8) 认不出的结束原因 / 重复结束 / 多候选 → 一律失败")
for payload, want in [
    ({"candidates": [{"content": {"parts": []}, "finishReason": "WHATEVER"}]}, "invalid_finish_reason"),
    ({"candidates": [{"content": {"parts": []}, "finishReason": "STOP"},
                     {"content": {"parts": []}, "finishReason": "STOP"}]}, "multiple_candidates"),
]:
    r = new_restorer()
    try:
        r.process(payload)
        check(f"{want} → 抛错", False, "没有抛错")
    except StreamFail as exc:
        check(f"{want} → {exc.code}", exc.code == want)

print("9) 提示词被拦截（无候选 + promptFeedback）→ content_filter")
r = new_restorer()
chunks = r.process({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}})
check("结束原因 content_filter", finish_of(chunks) == "content_filter", f"got={finish_of(chunks)!r}")

print("10) 完整 args（非分片）路径同样能还原")
r = new_restorer()
chunks = r.process({"candidates": [{"content": {"parts": [{"functionCall": {
    "name": TOOL, "args": {"content": "完整参数正文"}}}]},
    "finishReason": "STOP"}]})
check("正文还原", contents(chunks) == "完整参数正文", f"got={contents(chunks)!r}")
check("结束原因 stop", finish_of(chunks) == "stop")

print("11) usage 元数据被读取（camelCase → 内部保存）")
r = new_restorer()
r.process({"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 34,
                             "totalTokenCount": 46}})
check("usage 已解析", r.usage is not None and r.usage.get("promptTokenCount") == 12,
      f"got={r.usage!r}")

print()
if fails:
    print(f"❌ {len(fails)} 项未通过：" + ", ".join(fails))
    sys.exit(1)
print("✅ 全部通过")
