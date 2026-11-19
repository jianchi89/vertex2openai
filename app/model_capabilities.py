"""
模型能力矩阵（按家族识别）

用于决定：
- 每个模型支持的思考方式（thinking_level / 无）与合法档位
- 可用的采样参数集合（避免把不支持的参数发给模型导致 400）
- 生图模型支持的宽高比与分辨率白名单
- 是否要求最后一条消息为 user（Gemini 3.x 强制，否则 400 → 影响预填充）

设计原则：**按家族模式识别，而非写死每个型号**，因此往 vertexModels.json 里新增模型 ID
基本即插即用；未知/未来型号保守回退到“最新代（gemini-3）”档案。

依据（2026-07 官方文档核实）：
- thinking: ai.google.dev/gemini-api/docs/thinking
- 3.6/3.5 开发指南（采样弃用 + 预填充 400 + 去 candidate_count）
- 生图: ai.google.dev/gemini-api/docs/image-generation ；Agent Platform 3-pro-image / 3-1-flash-image
"""

import re
from typing import Any, Dict, Optional, Tuple

from models import normalize_content_part

# 所有“采样类”参数键（用于按模型剥离）
SAMPLING_KEYS = {
    "temperature", "top_p", "top_k",
    "presence_penalty", "frequency_penalty",
    "candidate_count", "seed", "stop_sequences", "max_output_tokens",
}

# 惩罚参数：新版 Gemini 收到非零 presence_penalty / frequency_penalty 直接 400。
# 默认从所有模型档案的 allowed_sampling 剔除（由 sanitize_sampling 自动剥离）；
# 前端仍可传这两个字段，但反代不会把它们发给 Google。
PENALTY_KEYS = {
    "presence_penalty", "frequency_penalty",
}

# ---- 生图分辨率与比例白名单 ----
# 出处：Agent Platform gemini-3-pro-image / gemini-3-1-flash-image / gemini-nano-banana-2-1
#      模型文档（核对于 2026-10-13）：三者均为 15 种宽高比
#      「1:1, 3:2, 2:3, 3:4, 1:4, 4:1, 4:3, 4:5, 5:4, 1:8, 8:1, 9:16, 16:9, 21:9, 9:21」
_PRO_IMAGE_ARS = {"1:1", "3:2", "2:3", "3:4", "1:4", "4:1", "4:3", "4:5", "5:4",
                 "1:8", "8:1", "9:16", "16:9", "21:9", "9:21"}
_FLASH_IMAGE_ARS = _PRO_IMAGE_ARS
_NANO_BANANA_ARS = _PRO_IMAGE_ARS

# ---- 思考档位阶梯（由低到高）----
# 出处：Agent Platform → Get started with Gemini 3（核对于 2026-07-26）
#   MINIMAL 仅 Gemini 3 Flash / 3.1 Flash-Lite 有；
#   MEDIUM 覆盖 Gemini 3 Flash / 3.1 Pro / 3.1 Flash-Lite；
#   LOW / HIGH 所有 3.x 都有。
# 2026-11 新增：gemini-4-hydrogen / gemini-4-neon 自带更高的 EXTRA_HIGH 档
#   （线上取值默认 "extra_high"，可用 settings["extra_high_wire"] 改成 "xhigh"，
#     见下方 EXTRA_HIGH_WIRE_DEFAULT 的注释）。
#   阶梯里必须把它排在 high 之上，否则 _clamp_level 的"就近向下取"会把它夹成 high。
_LEVEL_ORDER = ["minimal", "low", "medium", "high", "extra_high"]

#: 提供 EXTRA_HIGH 档的模型（前缀匹配，``gemini-4-hydrogen-preview`` 也算）。
#: 只写 4h / 4n 系：4c、4-argon 没有这一档，给它们下发 extra_high 会 400。
#: 2026-11-19 新增 gemini-4-neon（上游模型清单里与 4h 同属"四档思考 / 64K"那一类）。
EXTRA_HIGH_MODELS = ("gemini-4-hydrogen", "gemini-4-neon")


def _supports_extra_high(name: str) -> bool:
    """该型号是否有 EXTRA_HIGH 档。"""
    n = (name or "").lower()
    return any(n == m or n.startswith(m + "-") for m in EXTRA_HIGH_MODELS)


#: EXTRA_HIGH 这一档下发给上游时的**线格式取值**。
#: 两个来源说法不一致，所以不写死：
#:   - 2026-11-13 用户确认 4h 的取值是 "extra_high"；
#:   - 上游模型清单（bad-woman/vertex2openai 的 vertexModels.json）把同一档写成 "xhigh"。
#: 谁对取决于端点/时期。控制台接口可用 settings["extra_high_wire"] 覆盖（"" = 用这里的默认）。
#: 若日志里出现 400 且提到未知枚举/档位，先把 extra_high_wire 改成 "xhigh" 再试：
#:   curl -X POST <反代>/api/settings -H "Authorization: Bearer <API_KEY>" \
#:        -H "Content-Type: application/json" -d '{"extra_high_wire":"xhigh"}'
EXTRA_HIGH_WIRE_DEFAULT = "extra_high"


def wire_level(level: str, settings: Optional[Dict[str, Any]] = None) -> str:
    """内部档位名 → 下发给上游的取值（目前只有 extra_high 需要转换）。"""
    if str(level or "").strip().lower() != "extra_high":
        return level
    override = str((settings or {}).get("extra_high_wire") or "").strip()
    return override or EXTRA_HIGH_WIRE_DEFAULT

# MINIMAL 的可用性上界（含）：3.6 及更早的非 Pro 机型实测/文档支持 MINIMAL
# （3.6-flash 已真机验证 MINIMAL 生效、思考归零）。
# 出处：intro_gemini_3_7_flash.ipynb（核对于 2026-08-14）——3.7 Flash 的档位表
#   只列出 LOW / MEDIUM / HIGH，未提及 MINIMAL，且默认档位为 medium
#   （"replaced the high setting used in older Gemini 3 models"）。
# 方向性：漏给 MINIMAL 只是少压一点思考（拿到 LOW，不报错）；多给不支持的枚举会 400。
#   因此 3.7 及更新型号一律不提供 MINIMAL，等真机验证后再逐个放开。
_MINIMAL_MAX_VERSION = (3, 6)


def _supports_minimal(major: int, minor: int, is_pro: bool) -> bool:
    """该型号是否提供 MINIMAL 思考档位。"""
    if is_pro:
        return False                                  # Pro 系最低 LOW（官方）
    return (major, minor) <= _MINIMAL_MAX_VERSION


def _clamp_level(level: str, levels: set) -> str:
    """把任意档位夹到该模型的合法档位集合内。

    规则：**优先向下就近取**，向下没有再向上。
    这条方向性很关键——用户把档位调低是为了减少思考，
    旧实现对非法档位一律兜底成 "high"，导致 Pro 模型上选 minimal 反而拿到 high，
    与用户意图完全相反（P0-1）。
    """
    if not levels:
        return "low"
    if level in levels:
        return level
    try:
        idx = _LEVEL_ORDER.index(level)
    except ValueError:
        idx = _LEVEL_ORDER.index("high")   # 未知词按最高处理，再向下夹
    for i in range(idx, -1, -1):           # 先向下找最接近的合法档
        if _LEVEL_ORDER[i] in levels:
            return _LEVEL_ORDER[i]
    for i in range(idx + 1, len(_LEVEL_ORDER)):   # 向下没有再向上
        if _LEVEL_ORDER[i] in levels:
            return _LEVEL_ORDER[i]
    return sorted(levels)[0]


def sort_levels(levels) -> list:
    """按强度而非字典序排列档位，供控制台下拉框使用。"""
    known = [lv for lv in _LEVEL_ORDER if lv in levels]
    extra = sorted(lv for lv in levels if lv not in _LEVEL_ORDER)
    return known + extra


def _strip_known_suffixes(name: str) -> str:
    """去掉别名后缀（-search / 分辨率 / 思考档位后缀），得到判断家族用的基础名。"""
    n = name.lower()
    for suf in ("-search", "-1k", "-2k", "-4k", "-512"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = re.sub(r"-think-(extra[-_]?high|minimal|low|medium|high|off|none)$", "", n)
    return n


# ---------------------------------------------------------------- 档位后缀（-think-<档位>）
# 为什么要有：很多客户端（酒馆手机版、TauriTavern 等）的模型是**下拉选择**、不能手输
# 模型名，也没法发 reasoning_effort。把档位写进模型名（``gemini-4-hydrogen-think-extra-high``）
# 就成了唯一能"选中某一档"的方式，而且不依赖落盘状态（冷启动不丢）。
#
# 语义：模型名里的档位是**最强意图**——优先于前端 reasoning_effort、控制台档位，
# 也优先于预填充压制/「关闭原生思考」的降档（见 resolve_thinking 里的注释）。

LEVEL_SUFFIX_PREFIX = "-think-"

#: 认得的档位词（含客户端常见的几种写法）
_LEVEL_TOKENS = {
    "minimal", "low", "medium", "high",
    "extra_high", "extra-high", "extrahigh", "xhigh", "x-high",
    "off", "none",
}


def _norm_level_token(token: str) -> Optional[str]:
    """把后缀里的档位词归一成档案里的档位名；不认识返回 None。"""
    t = (token or "").strip().lower()
    if t not in _LEVEL_TOKENS:
        return None
    t = t.replace("-", "_")
    if t in ("extrahigh", "xhigh"):
        # xhigh / x-high 是客户端对"最高档"的叫法：在支持 EXTRA_HIGH 的模型上就用它，
        # 其余模型由 _clamp_level 就近夹回 high
        t = "extra_high"
    if t in ("off", "none"):
        # 老写法 -think-off / -think-none：Gemini 3.x 起无法真正关闭思考，压到最低合法档
        t = "minimal"
    return t


def split_level_suffix(name: str) -> Tuple[str, Optional[str]]:
    """识别模型名末尾的思考档位后缀，返回 (去掉后缀的基础名, 档位或 None)。

    只认**末尾**那一段：``gemini-4-hydrogen-think-extra-high`` → ("gemini-4-hydrogen",
    "extra_high")；``gemini-4-hydrogen-think-foo`` 原样返回（不猜）。
    """
    n = (name or "").strip()
    low = n.lower()
    idx = low.rfind(LEVEL_SUFFIX_PREFIX)
    if idx <= 0:
        return n, None
    level = _norm_level_token(low[idx + len(LEVEL_SUFFIX_PREFIX):])
    if level is None:
        return n, None
    base = n[:idx].strip().strip("-_ ")
    if not base:
        return n, None
    return base, level


def level_suffix(level: str) -> str:
    """档位 → 模型名后缀（extra_high → "-think-extra-high"）。"""
    return LEVEL_SUFFIX_PREFIX + str(level or "").strip().lower().replace("_", "-")


#: 默认在 /v1/models 里为「支持该档位」的模型生成 ``<模型>-think-<档位>`` 变体。
#: 只放"新增/高价值"的档位，不给 minimal/low/medium/high 都生成——那会让模型列表
#: 从 12 条膨胀到 40+ 条，酒馆下拉反而更难选。想加别的档位直接往这里加即可。
LEVEL_VARIANT_LEVELS = ("extra_high",)


def _temp_deprecated(name: str) -> bool:
    """temperature/top_p/top_k 是否已废弃。

    据官方 latest-model 文档：自 Gemini 3.6 Flash 与 Gemini 3.5 Flash-Lite 起，
    以及**所有更新/未来的 Gemini 模型**，这三个采样参数被废弃（现忽略、未来 400），
    需从请求中移除。更早的 3.x（3.0–3.5 非 lite）仍接受（但建议保持默认）。
    """
    n = name.lower()
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
    if not m:
        return True  # 未知/未来型号 → 前向安全，按已废弃处理
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    if major >= 4:
        return True
    if major == 3 and minor >= 6:
        return True
    if major == 3 and minor == 5 and "flash-lite" in n:
        return True
    return False


def get_profile(model_name: str) -> Dict[str, Any]:
    """返回给定模型的能力档案。"""
    raw = (model_name or "").lower()
    name = _strip_known_suffixes(raw)

    # ---- 生图模型 ----
    # 识别规则："image" in name 覆盖 3.x 生图；"nano-banana" in name 覆盖 Nano Banana 家族
    # （gemini-nano-banana-2.1 名字里不带 image，不加这条会被误判为文本模型）。
    if "image" in name or "nano-banana" in name:
        is_nano = "nano-banana" in name
        is_lite = "flash-lite" in name
        is_flash = "flash" in name
        # 官方合法 thinking 档位（Agent Platform，核对于 2026-10-13）：
        #   nano-banana-2.1: MINIMAL / MEDIUM / HIGH
        #   3.1-flash-image: MINIMAL / HIGH
        #   3-pro-image:     仅 HIGH
        # 默认档位统一 HIGH（用户决策 2026-10-13）。
        if is_nano:
            ars, sizes = _NANO_BANANA_ARS, {"1K", "2K", "4K"}
            thinking_levels = {"minimal", "medium", "high"}
        elif is_lite:
            ars, sizes = _PRO_IMAGE_ARS, {"1K"}
            thinking_levels = {"minimal", "high"}
        elif is_flash:
            ars, sizes = _FLASH_IMAGE_ARS, {"512", "1K", "2K", "4K"}
            thinking_levels = {"minimal", "high"}
        else:  # pro-image 及未知生图
            ars, sizes = _PRO_IMAGE_ARS, {"1K", "2K", "4K"}
            thinking_levels = {"high"}
        return {
            "family": "image",
            "is_image": True,
            "thinking_kind": "level",
            "thinking_levels": thinking_levels,
            "default_level": "high",
            "allowed_sampling": set(),          # 生图剥离所有采样参数
            "image_aspect_ratios": ars,
            "image_sizes": sizes,
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    # ---- 文本 / 多模态 ----
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    major = int(m.group(1)) if m else 3          # 未知 → 视作最新代
    minor = int(m.group(2)) if (m and m.group(2)) else 0
    is_pro = "pro" in name

    if major >= 3:
        # Gemini 3.x：思考用 thinking_level（取代 thinking_budget），且不支持 candidate_count。
        # temperature/top_p/top_k：自 3.6 Flash / 3.5 Flash-Lite 起（及所有更新/未来模型）已废弃，
        # 官方要求从请求移除（现忽略、未来 400）；更早的 3.x 仍可用但建议保持默认。
        levels = {"low", "medium", "high"}
        if _supports_minimal(major, minor, is_pro):
            levels = levels | {"minimal"}
        if _supports_extra_high(name):
            # gemini-4-hydrogen：比 high 更高的 EXTRA_HIGH（线上取值 extra_high）
            levels = levels | {"extra_high"}
        if "flash-lite" in name and "minimal" in levels:
            default_level = "minimal"
        elif is_pro:
            default_level = "high"
        else:
            # 非 Pro 的 3.x 默认 medium。3.7 Flash 官方明确默认 medium
            # （取代旧 3.x 的 high），与这里一致。
            default_level = "medium"
        allowed = set(SAMPLING_KEYS)
        allowed.discard("candidate_count")  # 所有 Gemini 3.x 不支持 candidate_count
        allowed -= PENALTY_KEYS  # 惩罚参数默认剥离：新版 Gemini 收到非零值直接 400
        temp_dep = _temp_deprecated(name)
        if temp_dep:
            allowed -= {"temperature", "top_p", "top_k"}
        return {
            "family": "g3",
            "is_image": False,
            "thinking_kind": "level",
            "thinking_levels": levels,
            "default_level": default_level,
            "allowed_sampling": allowed,
            "sampling_advice": "deprecated" if temp_dep else "recommend_default",
            "supports_search": True,
            "requires_user_last_turn": True,
        }

    # ---- 更早（2.0 / 1.5 等）：无思考、全采样、宽松 ----
    return {
        "family": "legacy",
        "is_image": False,
        "thinking_kind": None,
        "allowed_sampling": set(SAMPLING_KEYS) - PENALTY_KEYS,
        "supports_search": True,
        "requires_user_last_turn": False,
    }


# 客户端 reasoning_effort 归一化（SillyTavern 等会发 min/minimal/max/xhigh/auto 等）
_EFFORT_ALIASES = {
    "min": "minimal", "minimal": "minimal",
    "low": "low", "medium": "medium", "med": "medium",
    "high": "high", "max": "high", "xhigh": "high", "x-high": "high", "very_high": "high",
    "none": "off", "off": "off",
    # EXTRA_HIGH（gemini-4-hydrogen 起才有）：只有明确写出来才用，不把前端惯发的
    # xhigh 自动升上来——xhigh 一直映射到 high（见上），否则酒馆默认请求会在 4h 上
    # 直接跑最高档，正是 README 里"思考跑飞"的成因。想用就在模型名里选
    # gemini-4-hydrogen-think-extra-high。
    "extra_high": "extra_high", "extra-high": "extra_high", "extrahigh": "extra_high",
    "extra high": "extra_high",
}


def _norm_effort(e: Optional[str]) -> Optional[str]:
    if not isinstance(e, str):
        return None
    s = e.strip().lower()
    if not s or s == "auto":
        return None  # auto/空 → 交给控制台/模型默认
    return _EFFORT_ALIASES.get(s, s)


def _effort(request: Any) -> Optional[str]:
    e = getattr(request, "reasoning_effort", None)
    if not e and getattr(request, "model_extra", None):
        e = request.model_extra.get("reasoning_effort")
    return _norm_effort(e)


def _extra(request: Any, key: str) -> Any:
    v = getattr(request, key, None)
    if v is None and getattr(request, "model_extra", None):
        v = request.model_extra.get(key)
    return v


def forced_level_of(request: Any, model_name: str = "") -> Optional[str]:
    """本次请求被模型名后缀钉住的思考档位（``-think-extra-high`` → "extra_high"）。

    两处都看：``request.model`` 是客户端**原样发来的**模型名（通道内部可能已经把后缀
    剥掉才去调上游，所以不能只看 model_name），``model_name`` 是调用方手里的基础名。
    """
    for candidate in (getattr(request, "model", "") or "", model_name or ""):
        try:
            level = split_level_suffix(candidate)[1]
        except Exception:
            level = None
        if level:
            return level
    return None


def resolve_thinking(model_name: str, request: Any, settings: Dict[str, Any],
                     prefill_active: bool = False,
                     forced_level: Optional[str] = None) -> Dict[str, Any]:
    """
    计算思考配置（中立结构，各通道再转成自己的线格式）。
    返回 {"mode": None} 或 {"mode":"level","level":..}

    优先级（默认）：单次请求 reasoning_effort > 控制台/该模型专属 > 家族默认。
    **例外：模型名里的档位后缀（-think-<档位>）压过以上全部**——见下。

    控制台 native_thinking_mode（原生思考控制，向后兼容旧 hide_thoughts/thinking_force_console 布尔）：
    - "request"（默认）：跟随请求 effort（无则控制台/模型默认），返回思考。
    - "off"（关闭原生思考，酒馆预设推荐）：把思考压到该模型最低合法档
      （3.x/4.x=minimal/low，生图=minimal/high），忽略前端 effort，并 include_thoughts=False。**重要：batchGraphql(Studio) 会忽略
      includeThoughts，故 Cookie 通道还需在响应侧剥离思考块（见 chat_completions）；把档位压到 minimal
      才是 Studio 下真正减少原生思考、避免重预设思考阶段被截断的关键（已真机验证）。**
    - "console"（强制控制台档位）：忽略前端 effort，用控制台/该模型专属档位（未设则模型默认），返回思考。

    另外 prefill_suppress_thinking（默认开）：检测到预填充时等效走 "off" 压制路径。

    模型名档位后缀（2026-11 新增）：用户在客户端下拉里选的 ``xxx-think-extra-high`` 是
    **一次性的、最具体的意图**，所以它优先于前端 effort、控制台档位、以及上面那两条
    降档路径（off / 预填充压制），并且 include_thoughts=True（既然明确要这一档，就把
    思考一起回传，否则"选了 extra high 却看不到思考"更让人困惑）。返回值里带
    ``"forced": True``，通道可据此打一行日志说明压过了哪个设置。
    """
    prof = get_profile(model_name)
    if prof["thinking_kind"] is None:
        return {"mode": None}

    settings = settings or {}
    forced = forced_level or forced_level_of(request, model_name)

    # 统一的“原生思考控制”模式：request（跟随请求）| off（关闭原生思考）| console（强制控制台档位）
    # 向后兼容旧布尔开关（上一版的 hide_thoughts / thinking_force_console）。
    mode = settings.get("native_thinking_mode")
    if mode not in ("request", "off", "console"):
        if settings.get("hide_thoughts"):
            mode = "off"
        elif settings.get("thinking_force_console"):
            mode = "console"
        else:
            mode = "request"

    # suppress = 关闭原生思考（压到最低 + 隐藏 + 忽略前端）。预填充压制也走这条路。
    suppress = bool(mode == "off" or (prefill_active and settings.get("prefill_suppress_thinking", True)))
    ignore_client = bool(suppress or mode == "console")

    req_effort = None if ignore_client else _effort(request)
    include_thoughts = not suppress

    if prof["thinking_kind"] == "level":
        levels = prof["thinking_levels"]
        if forced:
            # 模型名后缀优先：仍按该模型的合法档位夹一次（别的模型没有 extra_high）
            return {"mode": "level", "level": _clamp_level(forced, levels),
                    "include_thoughts": True, "forced": True,
                    "forced_over": ("off" if suppress else
                                    "console" if mode == "console" else
                                    "request" if _effort(request) else None)}
        if suppress:
            # 压制：3.x 无法完全关闭思考 → 压到该模型最低合法档并隐藏
            return {"mode": "level", "level": _clamp_level("minimal", levels),
                    "include_thoughts": False}
        level = req_effort or settings.get("thinking_g3_level") or prof.get("default_level", "high")
        level = str(level).lower()
        if level in ("off", "none"):
            level = "minimal"
        # 统一就近向下夹取：Pro 上选 minimal 得到 low，而不是被抬成 high
        level = _clamp_level(level, levels)
        return {"mode": "level", "level": level, "include_thoughts": include_thoughts}

    return {"mode": None}


def apply_sampling_policy(profile: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """用控制台的「采样参数处理」覆盖自动判定，返回修正后的档案副本。

    自动判定按版本号（3.6+ / 3.5-flash-lite / 4.x 起废弃），但版本号表达不了
    "号更小却发布更晚"的情况——比如日后出一个 gemini-3.5-pro，官方按"更新模型"
    废弃采样，版本判据却会放行。有了这个开关，加模型就不必改代码。

      auto       —— 用内置版本判定（默认）
      deprecated —— 强制剥离 temperature / top_p / top_k
      allowed    —— 强制保留（官方澄清某模型仍可调时用）
    """
    policy = str(settings.get("sampling_policy", "auto") or "auto").lower()
    if policy not in ("deprecated", "allowed") or profile.get("is_image"):
        return profile          # 生图本来就剥离全部采样，不受该开关影响
    prof = dict(profile)
    allowed = set(prof.get("allowed_sampling", set()))
    if policy == "deprecated":
        allowed -= {"temperature", "top_p", "top_k"}
        prof["sampling_advice"] = "deprecated"
    else:
        allowed |= {"temperature", "top_p", "top_k"}
        # candidate_count 是 3.x 的硬限制，不归这个开关管
        if prof.get("family") == "g3":
            allowed.discard("candidate_count")
        prof["sampling_advice"] = "recommend_default"
    prof["allowed_sampling"] = allowed
    return prof


def sanitize_sampling(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """按档案剥离不支持的采样参数（防止未来 3.x 传弃用参数直接 400）。"""
    allowed = profile.get("allowed_sampling", set())
    for key in list(config.keys()):
        if key in SAMPLING_KEYS and key not in allowed:
            config.pop(key, None)
    return config


def resolve_image_size(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    """确定生图分辨率（大写 1K/2K/4K/512），按模型白名单校验，缺省回退。"""
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    sizes = prof["image_sizes"]
    settings = settings or {}
    raw = _extra(request, "image_size") or settings.get("image_size") or "1K"
    size = str(raw).upper().replace("0.5K", "512").replace("512PX", "512")
    if size in sizes:
        return size
    # 回退：就近向下取档（4K→2K→1K→512），向下没有再向上。
    # 旧实现写成 `for cand in (...): if cand in sizes: return "1K" if "1K" in sizes else cand`，
    # 因为 1K 在所有白名单里，任何非法值都返回 1K，注释里的“优先给相近的高档”从未生效。
    ladder = ["512", "1K", "2K", "4K"]
    idx = ladder.index(size) if size in ladder else len(ladder) - 1
    for i in range(idx, -1, -1):
        if ladder[i] in sizes:
            return ladder[i]
    for i in range(idx + 1, len(ladder)):
        if ladder[i] in sizes:
            return ladder[i]
    return "1K"


def _prompt_aspect_ratio(request: Any) -> Optional[str]:
    """从最后一条 user 消息文本里解析宽高比（--ar 优先，其次独立比例）。"""
    try:
        for msg in reversed(request.messages):
            if getattr(msg, "role", None) != "user":
                continue
            c = msg.content
            content = ""
            if isinstance(c, str):
                content = c
            elif isinstance(c, list):
                # F-1：先归一，否则 pydantic 生成的 ContentPartText 会被 isinstance(p, dict) 漏掉，
                # 导致列表形式内容里的 `--ar 16:9` 永远检测不到。
                parts = [normalize_content_part(p) for p in c]
                content = " ".join(
                    p.get("text", "") for p in parts
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            m = re.search(r"(?i)--ar\s*(\d+[:：]\d+)", content) or re.search(r"\b(\d+[:：]\d+)\b", content)
            return m.group(1) if m else None
    except Exception:
        return None
    return None


def resolve_aspect_ratio(model_name: str, request: Any, settings: Dict[str, Any]) -> Optional[str]:
    """
    生图宽高比解析（两通道共用）。优先级：
    请求额外字段 aspect_ratio/ar > OpenAI size 映射 > 提示词解析 > 控制台默认，
    最后按该生图模型白名单校验；不合法则返回 None（交给模型自动决定）。
    """
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    settings = settings or {}
    raw = _extra(request, "aspect_ratio") or _extra(request, "ar")
    if not raw:
        size_param = _extra(request, "size")
        if isinstance(size_param, str):
            if size_param == "1024x1024":
                raw = "1:1"
            elif size_param == "1024x768":
                raw = "4:3"
            elif size_param == "768x1024":
                raw = "3:4"
            elif ":" in size_param:
                raw = size_param
    if not raw:
        raw = _prompt_aspect_ratio(request)
    if not raw:
        raw = settings.get("image_aspect_ratio") or None
    return validate_aspect_ratio(model_name, raw)


def validate_aspect_ratio(model_name: str, ar: Optional[str]) -> Optional[str]:
    """校验宽高比是否在该生图模型白名单内；不在则返回 None（交给模型自动决定）。"""
    if not ar:
        return None
    prof = get_profile(model_name)
    if not prof["is_image"]:
        return None
    norm = str(ar).replace("：", ":").strip()
    return norm if norm in prof["image_aspect_ratios"] else None


def capabilities_summary(model_name: str, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """给控制台前端用的精简能力描述（决定显示/禁用哪些控件）。

    传入该模型生效的 settings 时，会把「采样参数处理」开关算进去，
    否则控制台显示的能力会和实际下发的参数对不上。
    """
    prof = get_profile(model_name)
    if settings:
        prof = apply_sampling_policy(prof, settings)
    thinking: Dict[str, Any] = {"kind": prof["thinking_kind"]}
    if prof["thinking_kind"] == "level":
        # 按强度排序（minimal→extra_high），不要用字典序（会排成 high, low, medium, minimal）
        thinking["levels"] = sort_levels(prof["thinking_levels"])
        thinking["can_off"] = False
        thinking["default_level"] = prof.get("default_level")
        # 这些档位在 /v1/models 里各有对应的 "<模型>-think-<档位>" 变体，客户端下拉可直接选
        thinking["variant_levels"] = [lv for lv in LEVEL_VARIANT_LEVELS
                                      if lv in prof["thinking_levels"]]
    return {
        "family": prof["family"],
        "is_image": prof["is_image"],
        "thinking": thinking,
        "sampling": sorted(prof["allowed_sampling"]),
        "sampling_advice": prof.get("sampling_advice"),
        "image_aspect_ratios": sorted(prof.get("image_aspect_ratios", set())),
        "image_sizes": sorted(prof.get("image_sizes", set())),
        "supports_search": prof["supports_search"],
    }
