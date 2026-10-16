"""模型列表变体验证：确认 /v1/models 里出现的抗截断版本与准入规则完全一致。

抽取 anti_truncation.py 里的纯函数 + 用假的 app_state 注入设置，
并复刻 models_api.add_model 的后缀合成逻辑来核对最终 id 集合。
"""
import ast
import json
import re
import sys
import typing
from pathlib import Path

SRC = Path(r"D:\DeepSeekH客户端\游戏安装\_v2o\vertex2openai-main\app\anti_truncation.py")
MODELS_JSON = Path(r"D:\DeepSeekH客户端\游戏安装\_v2o\vertex2openai-main\vertexModels.json")

tree = ast.parse(SRC.read_text(encoding="utf-8"))
WANT_FUNCS = {"variant_eligible", "variant_base_models"}
WANT_CONSTS = {"VARIANT_SUFFIX", "DEFAULT_VARIANT_MODELS"}


class FakeState:
    def __init__(self, settings):
        self.settings = settings

    def get_setting(self, key, default=None):
        return self.settings.get(key, default)


class FakeCapabilities:
    """只复刻 model_capabilities.get_profile 的生图判定（model_capabilities.py:143）。"""

    @staticmethod
    def get_profile(name):
        low = (name or "").lower()
        return {"is_image": ("image" in low or "nano-banana" in low)}


ns = {"re": re, "List": typing.List, "Dict": typing.Dict, "Any": typing.Any,
      "Optional": typing.Optional, "Tuple": typing.Tuple, "app_state": None,
      "mc": FakeCapabilities}
body = [n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in WANT_FUNCS)
        or (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in WANT_CONSTS for t in n.targets))]
exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
variant_eligible = ns["variant_eligible"]
variant_base_models = ns["variant_base_models"]
SUFFIX = ns["VARIANT_SUFFIX"]
DEFAULTS = ns["DEFAULT_VARIANT_MODELS"]

RAW = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"]
fails = []


def check(label, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + f"  {label}" + (f"   {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(label)


def list_models(settings, raw=RAW):
    """复刻 models_api.add_model 的后缀合成逻辑。"""
    ns["app_state"] = FakeState(settings)
    targets = set(variant_base_models(raw))
    out, seen = [], set()
    for base_id in raw:
        suffixes = [""]
        if "gemini" in base_id.lower() and "image" not in base_id.lower():
            suffixes.append("-search")
        if base_id in targets:
            suffixes.append(SUFFIX)
        for suffix in suffixes:
            final_id = f"{base_id}{suffix}"
            if final_id not in seen:
                out.append(final_id)
                seen.add(final_id)
    return out


print(f"目录里的模型（{len(RAW)} 个）：{', '.join(RAW)}")

print("\n1) 默认设置（config 里那三个）→ 只有这三个多出抗截断版本")
ids = list_models({"anti_truncation_models": DEFAULTS})
wanted = {f"{m}{SUFFIX}" for m in DEFAULTS}
got = {i for i in ids if i.endswith(SUFFIX)}
check("恰好三个变体", got == wanted, f"got={sorted(got)}")
check("没有 -search-抗截断 这种组合", not any(i.endswith(f"-search{SUFFIX}") for i in ids))
check("生图模型没有变体", not any("image" in i for i in got))
check("原有模型一个都没丢", set(RAW).issubset(set(ids)), f"missing={set(RAW) - set(ids)}")
check("原有 -search 变体还在", "gemini-3.7-flash-search" in ids and "gemini-4-argon-search" in ids)

print("\n2) 设置 [\"*\"] → 所有可用的文本模型都有变体，生图模型没有")
ids = list_models({"anti_truncation_models": ["*"]})
got = {i for i in ids if i.endswith(SUFFIX)}
expect = {f"{m}{SUFFIX}" for m in RAW if variant_eligible(m)}
check("变体 = 全部合格文本模型", got == expect, f"got={sorted(got)}")
check("数量合理", len(got) == 9, f"got={len(got)}")
check("gemini-3.1-flash-image 被排除", f"gemini-3.1-flash-image{SUFFIX}" not in got)
check("gemini-nano-banana-2.1 被排除", f"gemini-nano-banana-2.1{SUFFIX}" not in got)

print("\n3) 设置 [] → 一个变体都不生成（功能从列表里隐藏）")
ids = list_models({"anti_truncation_models": []})
check("没有任何变体", not any(i.endswith(SUFFIX) for i in ids))
check("基础模型照常列出", "gemini-4-argon" in ids)

print("\n4) 设置了上游没有的模型 → 不出现（避免选了必然 404）")
ids = list_models({"anti_truncation_models": ["gemini-9-fake", "gemini-4-argon"]})
got = {i for i in ids if i.endswith(SUFFIX)}
check("只保留目录里真实存在的", got == {f"gemini-4-argon{SUFFIX}"}, f"got={sorted(got)}")

print("\n5) 准入规则本身")
for name, want in [
    ("gemini-4-argon", True),
    ("gemini-3.7-flash", True),
    ("gemini-3.5-flash-lite", True),
    ("gemini-3.7-flash-search", False),      # 联网搜索请求会被跳过
    ("gemini-3.1-flash-image", False),       # 生图
    ("gemini-nano-banana-2.1", False),       # 生图（名字里没有 image）
    ("gemini-3-pro-image", False),
    ("gemini-embedding-001", False),
    ("gpt-4o", False),                       # 非 Gemini
    ("", False),
    ("-search", False),
]:
    check(f"{name or '(空)'} → {want}", variant_eligible(name) is want)

print("\n6) 后缀名与解析函数一致（发出的变体必须能被 split_trigger 剥掉）")
mod = ast.parse(SRC.read_text(encoding="utf-8"))
ns2 = {"re": re, "secrets": __import__("secrets"), "Tuple": typing.Tuple}
ns2.update({n.name: None for n in mod.body if isinstance(n, ast.ClassDef)})
body2 = [n for n in mod.body
         if (isinstance(n, ast.FunctionDef) and n.name in {"split_trigger", "new_tool_name"})
         or (isinstance(n, ast.Assign) and any(
             isinstance(t, ast.Name) and t.id in {"MODEL_SUFFIXES", "TOOL_PREFIX"}
             for t in n.targets))]
exec(compile(ast.Module(body=body2, type_ignores=[]), str(SRC), "exec"), ns2)
split_trigger = ns2["split_trigger"]
for m in RAW:
    if not variant_eligible(m):
        continue
    base, on = split_trigger(f"{m}{SUFFIX}")
    check(f"{m}{SUFFIX} → {m}", on and base == m, f"got={(base, on)}")

print("\n7) 工具名格式与上游一致")
check("router_emit_<24hex>", re.fullmatch(r"router_emit_[0-9a-f]{24}", ns2["new_tool_name"]()) is not None)

print()
if fails:
    print(f"❌ {len(fails)} 项未通过：" + ", ".join(fails))
    sys.exit(1)
print("✅ 全部通过")
