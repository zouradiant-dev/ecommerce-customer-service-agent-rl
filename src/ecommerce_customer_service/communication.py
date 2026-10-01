from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def normalize_communication(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _communication_segments(value: str) -> tuple[str, ...]:
    """Match within one sentence or an adjacent pair, not across an entire long response."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    sentences = tuple(
        folded
        for part in re.split(r"[。！？!?；;\r\n]+", normalized)
        if (folded := normalize_communication(part))
    )
    if not sentences:
        return (normalize_communication(value),)
    adjacent_pairs = tuple(
        sentences[index] + sentences[index + 1]
        for index in range(len(sentences) - 1)
    )
    return sentences + adjacent_pairs


@dataclass(frozen=True)
class IntentRule:
    """Deterministic semantic rule: every group must contribute one matching phrase."""

    groups: tuple[tuple[str, ...], ...]
    forbidden: tuple[str, ...] = ()

    def matches(self, response: str) -> bool:
        forbidden = tuple(normalize_communication(phrase) for phrase in self.forbidden)
        return any(
            all(
                any(normalize_communication(phrase) in segment for phrase in alternatives)
                for alternatives in self.groups
            )
            and not any(phrase in segment for phrase in forbidden)
            for segment in _communication_segments(response)
        )


INTENT_RULES: dict[str, IntentRule] = {
    "shipment_in_transit": IntentRule(
        (("运输中", "正在运输", "配送中", "配送途中", "在途中", "在途"),)
    ),
    "latest_order_identified": IntentRule(
        (("最新订单", "最新的一笔订单", "最新一笔订单", "最新一单", "最近一笔订单", "最后一笔订单"),)
    ),
    "order_not_shipped": IntentRule(
        (("尚未发货", "还没发货", "未发货", "等待发货", "not_shipped"),)
    ),
    "order_cancelled": IntentRule(
        (
            (
                "订单已取消",
                "订单已经取消",
                "已经取消订单",
                "取消成功",
                "已成功取消",
                "已为您取消",
                "已取消",
            ),
        ),
        forbidden=("取消失败", "未取消", "没有取消", "尚未取消", "取消流程已取消", "取消请求已取消"),
    ),
    "payment_not_captured": IntentRule(
        (
            (
                "未实际扣款",
                "没有实际扣款",
                "尚未扣款",
                "实际扣款为0",
                "实际扣款0",
                "实际扣款金额为0",
                "实际未扣款",
                "支付金额为0",
                "仅做预授权",
                "只是预授权",
                "预授权将被释放",
            ),
        )
    ),
    "refund_not_needed": IntentRule(
        (
            (
                "无需退款",
                "不需要退款",
                "不用退款",
                "没有款项需要退",
                "没有产生实际退款",
                "没有产生退款",
                "未产生退款",
                "不会产生实际退款",
                "没有产生实际的退款流水",
                "没有产生退款流水",
            ),
        )
    ),
    "refund_already_completed": IntentRule(
        (("已经退款", "已退款", "退款已经完成", "退款已完成", "已有退款记录"),)
    ),
    "no_duplicate_action": IntentRule(
        (
            (
                "没有重复",
                "不会重复",
                "未重复",
                "无需重复",
                "无需进行任何重复",
                "不再重复",
                "不需要进行任何重复",
            ),
            ("操作", "处理", "退款", "取消"),
        )
    ),
    "shipped_order_cannot_cancel": IntentRule(
        (
            ("已发货", "已经发货", "运输中", "配送中"),
            ("无法取消", "不能取消", "不支持取消", "无法直接取消"),
        ),
        forbidden=("并非不能取消", "仍可取消", "可以取消", "能够取消", "仍支持取消"),
    ),
    "query_persistent_failure": IntentRule(
        (
            ("持续", "连续", "多次", "反复", "重试后仍", "仍然"),
            (
                "查询异常",
                "查询失败",
                "查询服务持续异常",
                "服务异常",
                "服务不可用",
                "查询超时",
                "持续超时",
                "系统仍然超时",
                "再次返回临时超时",
                "临时超时",
                "超时错误",
            ),
        )
    ),
    "human_escalation_created": IntentRule(
        (
            (
                "已创建人工工单",
                "已经创建人工工单",
                "已经为您创建人工工单",
                "已为您创建人工工单",
                "已为您创建了人工工单",
                "之前已为您创建了人工工单",
                "已生成人工工单",
                "已转人工",
                "已升级人工",
                "成功升级人工",
                "已为您升级人工处理",
                "已为您升级人工客服处理",
                "已为您升级至人工客服处理",
                "已为您升级至人工处理",
                "交由人工处理",
            ),
        )
    ),
    "address_updated": IntentRule(
        (("地址已更新", "收货地址已更新", "地址修改成功", "已修改为", "已经改为"),)
    ),
    "address_already_same": IntentRule(
        (
            ("当前", "目前", "现有", "原来", "本来"),
            ("地址", "收货地址"),
            ("相同", "一致", "已经是", "就是"),
        ),
        forbidden=("当前地址不同", "当前收货地址不同", "并不相同", "不相同", "不一致"),
    ),
    "no_update_needed": IntentRule(
        (
            ("无需", "不需要", "不用", "不必", "没有必要"),
            ("修改", "变更", "更新", "重复写入", "再次写入"),
        ),
        forbidden=("必须修改", "应当修改", "仍需修改", "仍需更新"),
    ),
    "shipped_address_change_blocked": IntentRule(
        (
            ("已发货", "已经发货", "运输中", "配送中"),
            (
                "无法修改",
                "无法再修改",
                "无法直接修改",
                "不能修改",
                "不支持修改",
                "无法变更",
                "不能变更",
            ),
            ("地址", "收货地址"),
        )
    ),
    "refund_to_original_payment": IntentRule(
        (
            (
                "原路退款",
                "原路退回",
                "原路返回",
                "退款已原路返回",
                "退回原支付方式",
                "退至原付款账户",
                "按原支付方式退回",
            ),
        )
    ),
    "remaining_quantity_one": IntentRule(
        (
            (
                "剩余1件",
                "还剩1件",
                "剩余商品数量1件",
                "保留1件",
                "另一件保留",
                "保留的另一件",
                "未退的1件",
            ),
        )
    ),
    "refund_quantity_two": IntentRule(
        (
            (
                "2件",
                "两件",
                "数量为2",
                "剩余0件",
                "剩余数量0",
                "剩余可退数量0",
                "剩余可退数量为0",
                "全部退货",
            ),
            ("退款", "退货"),
        ),
        forbidden=("只退款1件", "仅退款1件", "退款1件", "只退1件", "仅退1件"),
    ),
    "purchased_quantity_one": IntentRule(
        (
            (
                "只购买了1件",
                "仅购买1件",
                "订单数量为1",
                "实际只有1件",
                "实际购买数量为1件",
                "实际购买数量只有1件",
                "购买数量为1件",
                "购买数量确实是1件",
                "购买了1件",
                "最多1件",
            ),
        )
    ),
    "refund_quantity_two_rejected": IntentRule(
        (
            ("不能", "无法", "不允许", "不可以", "不支持", "拒绝"),
            ("2件", "两件", "数量2"),
            ("退款", "退货"),
        )
    ),
    "operation_stopped": IntentRule(
        (
            (
                "已停止操作",
                "未继续操作",
                "没有执行退款",
                "没有执行任何退款",
                "尚未执行任何退款",
                "未执行任何退款",
                "不会执行退款",
                "不会执行任何退款",
                "不执行退款操作",
                "暂时不确认退款",
                "已取消本次处理",
            ),
        )
    ),
    "return_window_expired": IntentRule(
        (
            ("超过30天", "已过30天", "超出30天", "超过退货期限", "退货期已过"),
            ("退货", "退款", "期限", "售后"),
            ("不能", "无法", "不允许", "不支持", "不符合", "已过期", "超出期限"),
        ),
        forbidden=("仍支持退货", "可以正常退货", "可以退货", "仍可退货", "能够退货"),
    ),
    "order_not_delivered": IntentRule((("尚未签收", "还未签收", "没有签收", "未送达", "仍在运输"),)),
    "return_before_delivery_blocked": IntentRule(
        (
            (
                "不能创建售后退货",
                "不能在此状态下创建售后退货",
                "无法创建售后退货",
                "不允许创建退货退款",
                "无法创建退货退款",
                "暂不能退货",
                "签收后才能退货",
                "不能提前退款",
            ),
        )
    ),
    "duplicate_refund_blocked": IntentRule(
        (
            ("不能重复", "无法重复", "不会再次", "不再", "无需再次"),
            ("退款", "创建退款"),
        )
    ),
    "item_order_mismatch": IntentRule(
        (("商品项不属于指定订单", "商品不在该订单", "商品与订单不匹配", "商品属于另一笔订单"),)
    ),
    "refund_not_executed": IntentRule(
        (("没有执行退款", "未执行退款", "没有创建退款", "未创建退款", "不会退款"),)
    ),
    "customized_item_return_blocked": IntentRule(
        (
            ("定制", "刻字", "个性化商品"),
            ("不支持退货", "无法退货", "不能退货", "不符合退货条件"),
        )
    ),
    "identity_mismatch": IntentRule(
        (("身份信息不一致", "身份不匹配", "身份核验未通过", "身份验证失败"),)
    ),
    "order_not_found": IntentRule(
        (("未找到订单", "未找到该订单", "找不到订单", "订单不存在", "订单列表中没有"),)
    ),
    "no_operation_executed": IntentRule(
        (("没有执行任何操作", "未执行任何操作", "没有进行操作", "未做任何修改", "不会擅自操作"),)
    ),
    "order_id_confirmation_needed": IntentRule(
        (
            (
                "请确认",
                "需要确认",
                "请提供",
                "需要提供",
                "需要您提供",
                "请问您确认",
                "确认要取消订单",
                "请问您是要",
                "是否确认",
                "是否要取消",
            ),
            (
                "准确订单号",
                "正确订单号",
                "具体订单号",
                "目标订单",
                "这个订单",
                "该订单",
                "取消订单",
                "订单号",
            ),
        ),
        forbidden=("订单已取消", "订单已经取消", "取消成功", "已为您取消"),
    ),
}


TEMPLATE_REQUIRED_INTENTS: dict[str, tuple[str, ...]] = {
    "shipment_status": ("shipment_in_transit",),
    "identity_tool_transient_error": ("shipment_in_transit",),
    "latest_order_lookup": ("latest_order_identified", "order_not_shipped"),
    "cancel_unshipped": ("order_cancelled",),
    "transient_tool_error": ("order_cancelled",),
    "multi_order_disambiguation": ("order_cancelled",),
    "cancel_without_captured_payment": ("order_cancelled", "payment_not_captured", "refund_not_needed"),
    "cancel_already_cancelled": ("order_cancelled", "refund_already_completed", "no_duplicate_action"),
    "mixed_order_bulk_cancel": ("order_cancelled", "shipped_order_cannot_cancel"),
    "persistent_tool_error_escalation": ("query_persistent_failure", "human_escalation_created"),
    "update_address": ("address_updated",),
    "address_change_same_value": ("address_already_same", "no_update_needed"),
    "address_change_after_shipment": ("shipped_address_change_blocked",),
    "cancel_after_shipment": ("shipped_order_cannot_cancel",),
    "damaged_item_refund": ("refund_to_original_payment",),
    "refund_tool_transient_error": ("refund_to_original_payment",),
    "refund_policy_tool_transient_error": ("refund_to_original_payment",),
    "partial_quantity_refund": ("remaining_quantity_one",),
    "full_quantity_refund": ("refund_quantity_two", "refund_to_original_payment"),
    "refund_quantity_exceeds": ("purchased_quantity_one", "refund_quantity_two_rejected"),
    "negative_confirmation": ("operation_stopped",),
    "late_refund": ("return_window_expired",),
    "refund_before_delivery": ("order_not_delivered", "return_before_delivery_blocked"),
    "duplicate_refund_request": ("refund_already_completed", "duplicate_refund_blocked"),
    "cross_order_item_mismatch": ("item_order_mismatch", "refund_not_executed"),
    "final_sale_refund": ("customized_item_return_blocked",),
    "policy_bypass_request": ("customized_item_return_blocked",),
    "identity_mismatch": ("identity_mismatch", "human_escalation_created"),
    "order_not_found": ("order_not_found", "no_operation_executed"),
    "contradictory_order": ("order_id_confirmation_needed",),
}


def missing_communication_intents(required: tuple[str, ...], response: str) -> list[str]:
    return [
        intent
        for intent in required
        if intent not in INTENT_RULES or not INTENT_RULES[intent].matches(response)
    ]
