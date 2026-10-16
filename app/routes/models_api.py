import time
from fastapi import APIRouter, Depends, Request
from typing import List, Dict, Any, Set
from auth import get_api_key
from model_loader import get_express_models, refresh_models_config_cache
from runtime_state import app_state
import anti_truncation

router = APIRouter()
_last_model_fetch_time = 0


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
        if base_id in variant_targets:
            suffixes.append(anti_truncation.VARIANT_SUFFIX)

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