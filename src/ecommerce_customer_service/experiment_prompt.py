"""Explicit opt-in prompt experiment; the legacy benchmark stays unchanged.

These are global business rules from policies.py, not task IDs, verifier intents,
expected answers, or task metadata. New outputs must record this prompt version.
"""
from .simulator import SYSTEM_PROMPT


PROMPT_VERSION = "business_boundary_v12"
_OLD_BOUNDARY = "用户对订单号表示不确定或要求先不要操作时，只能列出订单并请用户确认准确订单号，不能替用户选择目标订单。"
_NEW_BOUNDARY = """用户对目标订单号表示不确定时，调用 list_orders 列出真实订单并请用户确认准确订单号，不能替用户选择目标订单。
目标订单已知，但用户拒绝、暂缓或撤回操作授权时，立即停止该操作；最后明确说明本次未执行、已暂停。不要因此查询无关订单，也不要继续催促确认。身份确认不是取消、改地址或退款的授权。
取消订单与修改地址各自只允许订单状态 pending 或 paid；其他状态不可直接执行。该规则独立于商品退货政策，不能用 search_refund_policy 的结果推导地址或取消资格。
search_refund_policy 只判断指定商品的售后退货资格：必须已签收、商品可退且在工具返回的退货期限内；必须查询并依据真实返回结果，不能凭通常经验承诺可退。取消未履约订单的退款不是售后退货。
政策不允许的操作要清楚解释并结束，不能一边拒绝一边询问是否继续执行，也不能擅自承诺拦截、换货、补发或人工处理。
若用户原请求已经明确确认相应操作，完成必要查询后直接调用可用写工具，不重复询问。最终回复依据实际工具结果说明完成、未执行或失败；只说准备查询、将会办理不等于执行。"""
if SYSTEM_PROMPT.count(_OLD_BOUNDARY) != 1:
    raise RuntimeError("legacy_prompt_changed; re-review the boundary experiment")

EXPERIMENT_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_OLD_BOUNDARY, _NEW_BOUNDARY)

# TRL discovers a static public-method tool surface. Do not promise dynamic
# hiding in that distinct runtime. Backend checks still enforce every call.
ONLINE_PROMPT_VERSION = "business_boundary_v12_static_tools"
ONLINE_SYSTEM_PROMPT = EXPERIMENT_SYSTEM_PROMPT.replace(
    "写工具在确认前会暂时隐藏；这不表示系统缺少该工具。查询完成但写工具尚未出现时，应询问用户确认原请求，不能因此升级人工。",
    "工具名称可见不代表当前获准执行；后端仍检查身份、查询顺序和用户授权。缺少明确授权时先询问，不能把工具可见或 confirmed=true 当成用户授权。",
).replace(
    "只能调用当前轮实际提供的工具；未提供升级工具时不得请求升级。临时工具错误后应重试原工具，直到升级工具在后续轮次出现。",
    "临时工具错误后重试原工具；只有身份核验失败或同一业务工具反复失败时才请求升级，后端会检查前置条件。",
).replace("每轮只会提供当前状态允许使用的工具。", "每次调用均受后端业务和授权检查约束。")
