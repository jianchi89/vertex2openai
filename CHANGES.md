# vertex2openai 三轮重做 — 变更说明

基于原版 `vertex2openai-main`（2026-09-01），只做加法，未改原有架构。

## 第一轮：Express Key 多账号管理

- `app/runtime_state.py`：Express Key 记录管理（增删改、启用/禁用、Project ID 与来源）；
  环境变量只做**一次性**初始导入——用户在控制台清空保存后，重启不会再把环境变量的 Key 复种回来。
- `app/express_key_manager.py`：多账号记录选择；`get_express_record()` 返回该 Key **自己**的 project_id。
- `app/express_key_probe.py`（新增）：经 Identity Toolkit 自动探测各 Key 的 Project ID；单个/批量；失败不阻断调用。
- `app/upstreams/express_sdk.py`：location 钉定优先用当前 Key 自己的项目；回退顺序：Key 自有 Project ID → 全局 Project ID → 裸模型名。
- `app/main.py`：lifespan 中一次性导入环境变量 Key、启动后台 Project ID 探测；控制台新增 Express Key 多账号卡片
  （明文显示 Key、增删改、启用/禁用、Project ID 与来源、单个探测/探测全部、保存空列表需确认）；
  新增 API：`GET /api/express-keys`、`POST /api/express-keys`、`POST /api/express-keys/probe`、`POST /api/express-keys/project`。

## 第二轮：弱网可靠交付（服务端缓冲 + 断线重放，不用 SSE 心跳）

- `app/resumable.py`（新增）：`ResumableBuffer` / `ResumableStore` / `DetachedRequest` / `compute_stream_id` /
  `pump_stream` / `sse_from_buffer`；TTL 清理、容量淘汰、并发锁。
- `app/routes/chat_api.py`：`/v1/chat/completions` 包一层可靠传输——
  - 流式：缓冲命中（未完整交付）则凭 `Last-Event-ID` / `?resume_from=` 重放/续传，不再打上游；
    否则建缓冲、后台泵抽帧（`DetachedRequest` 让上游读取与前端连接解绑），当前连接只做缓冲的读者；
    每帧带 `id: <seq>`，`X-Resume-Id` 响应头给出缓冲 ID。
  - 非流式：完整回复落缓冲；原样重发命中未交付缓冲直接重放。
  - 新增 `GET /v1/resumable/{stream_id}?from=<seq>&limit=&wait=` 分片轮询（短连接抗抖动，`wait` 为长轮询），
    返回带序号的事件分片、`next` 游标与 `done` 标志，客户端按 seq 去重、按 done 判定结束。
  - 请求 ID 优先取幂等请求头（`x-resume-id` > `x-request-id` > `x-idempotency-key`），否则用 API Key + 请求体指纹；
    幂等分支混入 API Key 与指纹，不同用户/不同请求体不会串缓冲。
- `app/config.py`：新增 `resumable_enabled`（默认开）、`resumable_ttl_seconds`（1800）、`resumable_max_streams`（200）。
- `app/main.py`：lifespan 启动缓冲清理器，停机时取消后台泵；控制台「开关」区可配三项续传参数。

## 第三轮：惩罚参数默认剥离

- `app/model_capabilities.py`：新增 `PENALTY_KEYS = {"presence_penalty", "frequency_penalty"}`，
  从 Gemini 3.x / 2.5 / legacy 全部分支的 `allowed_sampling` 剔除（`sanitize_sampling` 自动剥离兜底）。
- `app/api_helpers.py`：`create_generation_config()` 不再下发这两个字段；前端模型字段保留、仍可传；
  非零值被剥离时仅在 `debug_outbound` 开启时打一条提示。

## 验证

- 全部修改文件 `py_compile` 通过；`uvicorn main:app` 启动无报错。
- 功能测试通过：各模型档案剥离、sanitize 兜底、generationConfig 不含惩罚参数；
  缓冲落帧序号（`id == 下标`，注释帧不占序号）、`Last-Event-ID` 续传、请求 ID 优先级/隔离/稳定性、
  `DetachedRequest` 解绑、容量淘汰、TTL 配置。
- 与原包全量 md5 比对：修改 8 文件、新增 2 文件，无其他改动。

## 修复补丁（修复、补跑验证）
1. 并发去重竞态：`resumable.py` 新增 `ResumableStore.create_if_absent()`（持锁原子占位）；
   `routes/chat_api.py` 的 `_resumable_stream` / `_resumable_json` 改走原子占位，并发同体请求只打一次上游，
   上游失败/非流式响应时经 `_abort_placeholder` 撤占位（给跟读者补错误帧+终止帧）。
2. 轮询缺 `mark_delivered`：`resumable_poll` 在 `done && end >= next_seq` 时标记已交付，
   纯轮询读完后再发同一请求走新生成而非重放。

## 审计修复（2026-10-02，审计并修复，13 条断言验证通过）
1. 流式占位防僵尸：`_resumable_stream` 在 `await _call_upstream` 期间若任务被取消（客户端断开），
   原 `except Exception` 抓不到 `CancelledError`，占位缓冲会烂在 store 里无人认领——
   同指纹重试会跟读这个死缓冲直到被清扫（默认配置下最长 2 小时）。新增 `except asyncio.CancelledError`
   分支：先经 `_abort_placeholder` 撤占位（给跟读者补错误帧 + 终止帧）再重抛；无取消时行为不变。
2. 轮询坏游标误标记：`resumable_poll` 在 `from_` 越过末尾的坏游标下也会 `mark_delivered`，
   一次坏游标就挡住后续正常重放。改为仅当 `start <= buf.next_seq`（真正把本轮读到末尾）时才标记交付。

## 签名轮换修复（2026-10-04，用户抓包、换入）
- Cookie 直连通道突发全线空回复：上游 `cloudconsole-pa` 返回 `QUERY_SIGNATURE_NOT_FOUND`
  （code 5 "Requested entity was not found"），系 Google 作废了硬编码在
  `app/cookie_auth.py` 的 `STREAM_GENERATE_QUERY_SIGNATURE`（README 已预警此内部接口无兼容性承诺）。
- 新签名由用户从 Cloud Console 前端 `batchGraphql` 请求（`operationName=StreamGenerateContent`
  的 Payload）抓取：`2/Hc4FpJfYmM+gO5TB0hcjYY0Iwj0rqLIVDZOhurBbu/I=`，已替换旧值；
  旧签名在全代码中零残留。部署后需重启服务生效。

## 第四轮：Gemini 4c / 4h 适配（2026-11-13）

新增两个 4.x 模型：`gemini-4-carbon`（4c）与 `gemini-4-hydrogen`（4h）。两者原先**都没有抗截断变体**；
4h 还自带一个更高的思考档 `extra_high`，之前代理不认识、列表里也选不到。

- `vertexModels.json` / `app/config.py`：把 4c、4h 加入模型列表与 `anti_truncation_models` 默认值，
  于是 `/v1/models` 里直接出现 `gemini-4-carbon-抗截断`、`gemini-4-hydrogen-抗截断`。
- `app/config.py`（新增 `migrate_setting()`）：**旧默认值静默升级**——`web_state.json` 里存着旧默认列表
  （`["gemini-4-argon","gemini-3.8-flash","gemini-3.7-flash"]`）时会自动升级到含 4c/4h 的新默认值；
  用户自己改过的列表（包括故意清空的 `[]`）一律不动。`runtime_state` 的 `get_settings`/`get_setting` 都走它。
- `app/model_capabilities.py`：
  - 档位阶梯加入 `extra_high`（排在 `high` 之上），`EXTRA_HIGH_MODELS = ("gemini-4-hydrogen",)`
    命中时该模型档案多一档 `extra_high`；4c 及其它模型没有这一档，收到 `extra_high` 会被
    `_clamp_level` 就近夹成 `high`（不会拿不支持的枚举去冒 400）。
  - 新增 **模型名档位后缀** `-think-<档位>`（`split_level_suffix` / `level_suffix`），
    `resolve_thinking()` 新增 `forced_level`：模型名里的档位是**最强意图**，压过前端
    `reasoning_effort`、控制台档位、以及"关闭原生思考"/预填充压制的降档，并回传思考
    （返回值带 `forced` / `forced_over`，通道会打日志说明压过了谁）。
  - `reasoning_effort` 别名新增 `extra_high`/`extra-high`/`extrahigh`；**`xhigh` 仍映射到 `high`**
    （保持 README 里"酒馆恒发 xhigh"的既有行为，不会在 4h 上被自动顶到最高档）。
  - `capabilities_summary` 多返回 `default_level` 与 `variant_levels`，控制台据此提示。
- `app/routes/models_api.py`：`/v1/models` 为支持该档位的模型额外生成 `<模型>-think-<档位>` 变体
  （默认只有 `extra_high`，见 `LEVEL_VARIANT_LEVELS`），与抗截断叠加时再给一条
  `gemini-4-hydrogen-think-extra-high-抗截断`。生图模型不生成。
- `app/upstreams/express_sdk.py`：新增 `_peel_model_aliases()`，**顺序无关地**反复剥掉
  `-抗截断` / `-think-<档位>` / `-search` 三种后缀（最多四层），避免叠加变体把假模型名发给上游（404）；
  `_build_thinking_config()` 接收 `forced_level`。
- `app/upstreams/cookie_proxy.py`：同样剥离档位后缀；`thinkingLevel` 透传为 `EXTRA_HIGH`
  （Studio 的枚举未实测，若报未知枚举请改用 Express 通道或选 `high`）。
- `app/main.py`：控制台「思考档位」下拉新增 `extra_high`（标注"仅 4h"），能力提示里显示
  默认档位、最高档与对应的模型名变体。
- 自测：`py_compile` 全绿；另用独立脚本覆盖 45 项断言（档案档位、后缀解析、优先级、夹取、
  `/v1/models` 变体、抗截断候选、旧值升级），全部通过。

