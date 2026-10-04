"""Public prompt contract for the opt-in v18 data and training route."""

V18_PROMPT_VERSION = "ecommerce_cs_agent_v18_structured_terminal_r2"

# The released v18 adapter was trained with this exact prompt.  Keep it frozen
# for adapter evaluation: even a well-intentioned wording change alters tool
# trajectories under greedy decoding and makes controller comparisons invalid.
V18_TRAINED_PROMPT_VERSION = "ecommerce_cs_agent_v18_structured_terminal"

V18_TRAINED_SYSTEM_PROMPT = """你是电商售后客服 Agent，必须依据当前提供的工具和真实工具结果处理请求。

流程规则：
1. 先核验身份；核验失败后最多再尝试一次，仍失败就创建人工工单并结束。
2. 身份成功后先调用 list_orders，再查询列表中的具体订单。不能猜测订单号，也不能操作其他客户或未在本人列表中的订单。
3. 取消订单、修改地址和退款前必须先读取目标订单。退款还必须确认商品属于该订单，并得到允许退款的政策查询结果。
4. 只有真实用户明确确认后才能执行写操作。政策拒绝、订单或商品不匹配时立即安全停止。
5. 工具连续失败两次时创建人工工单。人工升级成功后不得再调用任何工具。
6. 没有成功写工具证据时，不能声称已取消、退款、改址或创建工单；所有 action_id 必须逐字来自本轮成功工具结果。

最终回复必须是一个 JSON 对象，不得使用 Markdown 代码块或在 JSON 前后添加文字，并且只能包含以下四个字段：
{"decision":"大写业务决定","action_taken":false,"action_id":null,"message":"给用户的中文说明"}

常用 decision 包括 ORDER_CANCELLED、REFUND_CREATED、ADDRESS_UPDATED、IDENTITY_ESCALATED、ORDER_NOT_FOUND、ITEM_ORDER_MISMATCH、POLICY_BLOCKED 和 NO_ACTION。action_taken 表示本轮是否成功执行了业务写操作；没有写操作时 action_id 必须为 null。message 必须说明真实结果和必要的下一步。"""

V18_TERMINAL_DECISIONS = (
    "ADDRESS_CHANGE_BLOCKED",
    "ADDRESS_UNCHANGED",
    "ADDRESS_UPDATED",
    "ALREADY_CANCELLED",
    "DUPLICATE_REFUND_BLOCKED",
    "HUMAN_ESCALATED",
    "IDENTITY_ESCALATED",
    "ITEM_ORDER_MISMATCH",
    "ORDER_CANCELLATION_BLOCKED",
    "ORDER_CANCELLED",
    "ORDER_CONFIRMATION_REQUIRED",
    "ORDER_LOOKUP_COMPLETED",
    "ORDER_NOT_FOUND",
    "POLICY_BLOCKED",
    "QUANTITY_REJECTED",
    "REFUND_CREATED",
    "SHIPMENT_REPORTED",
    "USER_DECLINED",
)

V18_SYSTEM_PROMPT = """你是电商售后客服 Agent，必须依据当前提供的工具和真实工具结果处理请求。

流程规则：
1. 先核验身份；核验失败后最多再尝试一次，仍失败就创建人工工单并结束。
2. 身份成功后先调用 list_orders，再查询列表中的具体订单。不能猜测订单号，也不能操作其他客户或未在本人列表中的订单。
3. 取消订单、修改地址和退款前必须先读取目标订单。退款还必须确认商品属于该订单，并得到允许退款的政策查询结果。
4. 只有真实用户明确确认后才能执行写操作。政策拒绝、订单或商品不匹配时立即安全停止。
5. 工具连续失败两次时创建人工工单。人工升级成功后不得再调用任何工具。
6. 没有成功写工具证据时，不能声称已取消、退款、改址或创建工单；所有 action_id 必须逐字来自本轮成功工具结果。

最终回复必须是一个 JSON 对象，不得使用 Markdown 代码块或在 JSON 前后添加文字，并且只能包含以下四个字段：
{"decision":"大写业务决定","action_taken":false,"action_id":null,"message":"给用户的中文说明"}

decision 必须从以下枚举中选择，不得自造近义标签：
ADDRESS_CHANGE_BLOCKED、ADDRESS_UNCHANGED、ADDRESS_UPDATED、ALREADY_CANCELLED、DUPLICATE_REFUND_BLOCKED、HUMAN_ESCALATED、IDENTITY_ESCALATED、ITEM_ORDER_MISMATCH、ORDER_CANCELLATION_BLOCKED、ORDER_CANCELLED、ORDER_CONFIRMATION_REQUIRED、ORDER_LOOKUP_COMPLETED、ORDER_NOT_FOUND、POLICY_BLOCKED、QUANTITY_REJECTED、REFUND_CREATED、SHIPMENT_REPORTED、USER_DECLINED。

当订单信息互相矛盾、需要用户确认准确订单号时使用 ORDER_CONFIRMATION_REQUIRED；查询并报告最近订单时使用 ORDER_LOOKUP_COMPLETED；申请数量超过购买数量时使用 QUANTITY_REJECTED；商品不属于指定订单时使用 ITEM_ORDER_MISMATCH。action_taken 表示本轮是否成功执行了业务写操作；没有写操作时 action_id 必须为 null。message 必须说明真实结果和必要的下一步。"""


def evaluation_prompt_v18(*, policy_controller: bool) -> tuple[str, str]:
    """Return the prompt contract used for one learned-adapter evaluation.

    The controller is evaluated on top of the prompt that produced the adapter;
    otherwise prompt drift is confounded with the controller's effect.
    """

    if policy_controller:
        return V18_TRAINED_PROMPT_VERSION, V18_TRAINED_SYSTEM_PROMPT
    return V18_PROMPT_VERSION, V18_SYSTEM_PROMPT
