import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_express_models, refresh_models_config_cache
from runtime_state import app_state
import anti_truncation
import model_capabilities as mc

router = APIRouter()
_last_model_fetch_time = 0


def _level_suffixes(base_id: str) -> List[str]:
    """该模型要在模型列表里额外暴露的思考档位变体后缀（默认只给 extra_high）。

    为什么必须并进列表：酒馆手机版等客户端的模型是下拉选择，既不能手输模型名，
    也没有 reasoning_effort 输入框。把档位做成 ``<模型>-think-<档位>`` 后，刷新模型
    列表就能直接选到 gemini-4-hydrogen-think-extra-high。
    """
    try:
        prof = mc.get_profile(base_id)
    except Exception:
        return []
    if prof.get("thinking_kind") != "level" or prof.get("is_image"):
        return []
    levels = prof.get("thinking_levels") or set()
    return [mc.level_suffix(lv) for lv in mc.LEVEL_VARIANT_LEVELS if lv in levels]


@router.get("/v1/models")
async def list_models(fastapi_request: Request, api_key: str = Depends(get_api_key)):
    global _last_model_fetch_time

    current_time = time.time()
    if current_time - _last_model_fetch_time > 3600:
        await refresh_models_config_cache()
        _last_model_fetch_time = current_time

    express_key_manager_instance = fastapi_request.app.state.express_key_manager
    
    # 动态放行：开启网页反代或者配置有 Express API Key 均可安全获取模型列表
    has_web_proxy = app_state.is_web_proxy_enabled()
    has_express_key = express_key_manager_instance.get_total_keys() > 0
    
    raw_models = await get_express_models() if (has_express_key or has_web_proxy) else []

    # 抗截断版本：客户端只能从模型列表里选，所以变体必须并进列表（默认只给
    # gemini-4-argon / 3.8-flash / 3.7-flash，见 anti_truncation.variant_base_models）。
    try:
        variant_targets = set(anti_truncation.variant_base_models(raw_models))
    except Exception as exc:
        print(f"⚠️ [抗截断] 生成模型变体失败（不影响模型列表）：{exc}")
        variant_targets = set()

    final_model_list: List[Dict[str, Any]] = []
    processed_ids: Set[str] = set()

    def add_model(base_id: str):
        suffixes = [""]
        if "gemini" in base_id.lower() and "image" not in base_id.lower():
            suffixes.append("-search")
        is_variant = base_id in variant_targets
        if is_variant:
            suffixes.append(anti_truncation.VARIANT_SUFFIX)
        # 思考档位变体（默认只有 extra_high，见 mc.LEVEL_VARIANT_LEVELS）：
        # 有档位的模型直接多一条 "<模型>-think-<档位>"；若该模型同时有抗截断版本，
        # 再给一条两者叠加的，省得用户在酒馆里二选一。
        for lv_suf in _level_suffixes(base_id):
            suffixes.append(lv_suf)
            if is_variant:
                suffixes.append(lv_suf + anti_truncation.VARIANT_SUFFIX)

        for suffix in suffixes:
            final_id = f"{base_id}{suffix}"
            if final_id in processed_ids:
                continue
            final_model_list.append({
                "id": final_id,
                "object": "model",
                "created": int(current_time),
                "owned_by": "google",
                "permission": [],
                "root": base_id,
                "parent": None,
            })
            processed_ids.add(final_id)

    for model_id in raw_models:
        add_model(model_id)

    return {"object": "list", "data": sorted(final_model_list, key=lambda item: item["id"])}