from __future__ import annotations

from dataclasses import dataclass, replace
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
    patterns: tuple[str, ...] = ()
    reject_patterns: tuple[str, ...] = ()

    def matches(self, response: str) -> bool:
        # New synonyms must not accept negated, future or contradictory claims.
        # These guards are opt-in per intent; unrelated v8 rules stay unchanged.
        folded_response = normalize_communication(response)
        if any(re.search(pattern, folded_response) for pattern in self.reject_patterns):
            return False
        forbidden = tuple(normalize_communication(phrase) for phrase in self.forbidden)
        return any(
            (
                all(
                    any(normalize_communication(phrase) in segment for phrase in alternatives)
                    for alternatives in self.groups
                )
                or any(re.search(pattern, segment) for pattern in self.patterns)
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
                "已经被取消",
                "已被取消",
            ),
        ),
        patterns=(r"订单[a-z0-9]{0,80}已(?:经)?于(?:\d{4}年\d{1,2}月\d{1,2}日|\d{8})取消",),
        forbidden=("取消失败", "未取消", "没有取消", "尚未取消", "取消流程已取消", "取消请求已取消"),
        reject_patterns=(
            r"(?:并非|不是|未|尚未)(?:已(?:经)?)?被取消",
            r"(?:并非|不是)已(?:经)?于[0-9年月日]{6,14}取消",
            r"已(?:经)?于[0-9年月日]{6,14}取消.{0,40}(?:实际上|其实)(?:该)?订单(?:尚未|未|没有)取消",
        ),
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
        (("已经退款", "已退款", "退款已经完成", "退款已完成", "已有退款记录", "已经有退款记录", "退款成功"),),
        patterns=(
            r"已(?:经)?(?:成功)?(?:取消并(?:且)?)?完成退款",
            r"已(?:经)?完成取消和退款(?:流程)?",
            r"退款已(?:经)?成功(?:完成|处理)",
            r"退款已(?:经)?处理完成",
            # Processing alone is not completion: require a refunded status
            # or an explicit statement that the funds were returned.
            r"退款已(?:经)?处理(?:(?!订单|物流|支付|商品|发货|签收|运输).){0,80}(?:状态(?:为|是)?refunded|已退回(?:到)?)",
            r"退款状态(?:为|是)?(?:completed|已完成)",
            # Associate a structured status with the refund record/amount, not
            # another object's completed shipping/payment status.
            r"退款(?:记录|金额)(?:(?!订单|物流|支付|商品|发货|签收|运输).){0,160}状态(?:为|是)?(?:completed|已完成)",
        ),
        reject_patterns=(
            r"(?:未|尚未|还未|没有|并未|并非|不是)(?:已(?:经)?)?(?:成功)?(?:完成退款|退款(?:已(?:经)?)?完成|已退款|退款成功|已(?:经)?有退款记录)",
            r"(?:如果|假如|若|待|等待|计划|将|即将|准备)(?:已(?:经)?)?(?:完成退款|退款(?:已(?:经)?)?完成|退款成功)",
            r"退款(?:状态)?(?:为|是)?(?:待处理|处理中|未完成|失败)",
            r"退款(?:记录|金额)(?:(?!订单|物流|支付|商品|发货|签收|运输).){0,160}状态(?:为|是)?(?:pending|processing|failed|待处理|处理中|未完成|失败)",
            r"(?:如果|假如|若|将|即将|准备)(?:退款状态(?:为|是)?(?:completed|已完成)|已(?:经)?完成取消和退款)",
            r"退款(?:尚未|还未|未|没有|并未)(?:处理完成|退回)",
            r"(?:并非|不是|如果|假如|假设|若)(?:退款)?已(?:经)?处理(?:完成|.{0,80}(?:状态(?:为|是)?refunded|已退回))",
            r"退款已(?:经)?处理(?:完成)?(?:(?!订单|物流|支付|商品|发货|签收|运输).){0,80}状态(?:为|是)?(?:pending|processing|failed|待处理|处理中|未完成|失败)",
        ),
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
                "不需要重复",
                "不必重复",
                "无需再次",
                "不用再次",
                "不需要再次",
            ),
            ("操作", "处理", "退款", "取消"),
        ),
        reject_patterns=(
            r"(?:并非|不是)(?:不需要|不必)(?:重复|再次)",
            r"(?:但|但是|不过|然而)(?:我(?:们)?)?(?:会|将|仍然|仍|继续|正在)(?:进行)?(?:重复|再次)(?:取消|退款|操作|处理)",
        ),
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
                "购买数量是1件",
                "购买了1件",
                "最多1件",
            ),
        ),
        patterns=(r"(?:实际)?购买数量(?:确实)?(?:是|为|只有)(?:1|一)件", r"(?:只|仅)(?:购买|买)了?(?:1|一)件"),
        reject_patterns=(
            r"(?:并非|不是|不只|不仅|没有|未)(?:实际)?(?:购买(?:数量)?|买).{0,8}(?:1|一)件",
            r"(?:订单数量为|最多)1\d+",
        ),
    ),
    "refund_quantity_two_rejected": IntentRule(
        (
            ("不能", "无法", "不允许", "不可以", "不支持", "拒绝"),
            ("2件", "两件", "数量2"),
            ("退款", "退货"),
        ),
        # A hard upper limit of ONE item rejects a request for TWO. Merely
        # suggesting one item, or asking why two are wanted, is not a refusal.
        patterns=(r"(?:只|仅)(?:能|允许)申请(?:1|一)件(?:商品)?(?:的)?(?:退款|退货)",),
        reject_patterns=(
            r"(?:并非|不是|不只|不仅)(?:只|仅)(?:能|允许)申请(?:1|一)件",
            r"(?<!不)(?<!无法)(?<!不能)(?:可以|允许|支持|能够)(?:按|申请|办理|退还)?(?:2|两|二)件(?:商品)?(?:数量)?(?:的)?(?:退款|退货)",
            r"(?:并非|不是)(?:不能|无法|不允许|不可以|不支持)(?:按|申请)?(?:2|两|二)件(?:数量)?退款",
            r"(?:我(?:们)?)(?:将|会|已(?:经)?|现在就)?(?:按|申请|办理|退还)(?:2|两|二)件(?:商品)?(?:数量)?(?:的)?(?:退款|退货)",
            r"(?<!如果)(?<!假如)(?<!若)您(?:现在)?(?:提到|表示)(?:只)?想按(?:1|一)件数量申请退款",
            r"(?:不存在|没有|并未)(?:超额退款|超额退货)",
        ),
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
                "未执行退款操作",
                "尚未执行退款操作",
                "暂时不确认退款",
                "已取消本次处理",
            ),
        ),
        patterns=(
            r"退款(?:操作)?不会(?:被)?执行",
            r"(?:无法|不能)执行(?:任何)?退款操作",
            r"您不需要我现在执行(?:任何)?(?:退款)?操作",
            r"我(?:们)?(?:将|会|已(?:经)?)?(?:暂时)?暂停退款(?:操作|处理|流程)",
            r"退款操作(?:将)?(?:保持|维持)暂停状态",
            r"我(?:们)?(?:将|会)?(?:保持|维持)当前状态(?:直到|直至)您(?:明确)?确认执行退款操作",
            r"需要用户明确确认后才能操作.{0,40}尚未执行退款",
        ),
        reject_patterns=(
            r"(?:并非|不是)(?:退款(?:操作)?不会(?:被)?执行|(?:无法|不能)执行(?:任何)?退款操作|您不需要我现在执行)",
            r"(?:但|但是|不过|然而|实际上)(?:我(?:们)?)?(?:将|会|已(?:经)?|正在|继续|立即|马上|现在就)?(?:为您)?(?:执行|办理|处理|创建)(?:退货|退款)",
            r"我(?:们)?(?:已(?:经)?|刚刚|刚才|正在|继续|将立即|现在就)(?:为您)?(?:执行|办理|处理|创建)(?:了)?(?:退货|退款)",
            r"(?:退货|退款)(?:操作)?已(?:经)?(?:被)?(?:执行|创建|办理|完成|成功)",
            r"我(?:们)?(?:并非|不是|不会|不|无法|不能|没有|并未|未)(?:将|会|已(?:经)?)?(?:暂时)?暂停退款",
            r"(?:并非|不是|如果|假如|假设|若)(?:我(?:们)?)?(?:将|会|已(?:经)?)?(?:暂时)?暂停退款",
            r"退款操作(?:未|没有|不会|不能)(?:保持|维持)?暂停",
            r"(?:并非|不是)(?:尚未|未)执行退款操作",
            r"(?:无需|不需要|不用|不必)(?:您|用户)?(?:明确)?确认.{0,12}(?:直接|立即|马上|就)(?:执行|办理|处理|创建)退款",
        ),
    ),
    "return_window_expired": IntentRule(
        (
            ("超过30天", "已过30天", "超出30天", "超过退货期限", "退货期已过"),
            ("退货", "退款", "期限", "售后"),
            ("不能", "无法", "不允许", "不支持", "不符合", "已过期", "超出期限", "被拒绝"),
        ),
        forbidden=("仍支持退货", "可以正常退货", "可以退货", "仍可退货", "能够退货"),
        reject_patterns=(
            r"(?:未|尚未|没有|并未|并非|不是)(?:超过|超出|已过)(?:30天|退货期限)",
            r"(?:如果|假如|假设|若)(?:商品|订单)?(?:已(?:经)?)?(?:超过|超出|已过)(?:30天|退货期限)",
            r"(?:仍|仍然)(?:可|可以|能够|支持)(?:正常)?(?:退货|退款)",
            r"(?:但|但是|不过|然而|实际上)(?:现在|当前|仍|仍然)?(?:可以|允许|能够|支持)(?:正常)?(?:退货|退款)",
        ),
    ),
    "order_not_delivered": IntentRule((("尚未签收", "还未签收", "没有签收", "未送达", "仍在运输"),)),
    "return_before_delivery_blocked": IntentRule(
        (
            (
                "不能创建售后退货",
                "不能在此状态下创建售后退货",
                "无法创建售后退货",
                "不允许创建退货退款",
                "不允许创建售后退货",
                "不能创建退货退款",
                "无法创建退货退款",
                "暂不能退货",
                "签收后才能退货",
                "不能提前退款",
            ),
        ),
        patterns=(
            r"不符合(?:直接|立即)?创建(?:售后)?退货退款的条件",
            r"(?:不允许|不能|无法)在(?:订单|包裹|商品)?(?:尚未|还未|未)(?:签收|送达)(?:时|前)?创建(?:售后)?退货(?:退款)?",
        ),
        reject_patterns=(
            r"(?:并非|不是)(?:不允许|不能|无法)创建(?:售后)?退货",
            r"(?:尚未签收|还未签收|未送达)(?:但|却|也|仍可|仍然)(?:可以|允许|能够|支持)?创建(?:售后)?退货",
            r"(?:当前|现在)(?:仍|仍然)?(?:可以|允许|能够|支持)创建(?:售后)?退货",
            r"(?:并非|不是)不符合(?:直接|立即)?创建(?:售后)?退货退款的条件",
            r"(?:但|但是|不过|实际上)(?:我(?:们)?)?(?:将|会|已(?:经)?|立即)?(?:为您)?创建(?:售后)?退货",
        ),
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
                "此订单",
                "取消订单",
                "订单号",
            ),
        ),
        patterns=(
            r"现在确认就是(?:这个|该|这笔|目标)订单吗",
            r"请(?:您)?确认(?:是否是|是不是)(?:这个|该|这笔|目标)订单",
            r"您(?:之前)?提到(?:要)?取消的是(?:这个|该|这笔|目标)订单吗",
            r"请(?:您)?明确告知(?:准确|正确|具体)?订单号",
            r"(?:请|需要)(?:您)?确认(?:一下)?(?:准确|正确|具体)(?:的)?订单号",
        ),
        forbidden=("订单已取消", "订单已经取消", "取消成功", "已为您取消"),
        reject_patterns=(
            r"(?:无需|不用|不必|不要)(?:现在)?确认(?:就是|是否是)(?:这个|该|这笔|目标)订单",
            r"(?:无需|不用|不必|不要)(?:再|再次)?(?:确认|提供|告知)(?:准确|正确|具体)?(?:的)?订单号",
            r"(?:无需|不用|不必)(?:您)?(?:再|再次)?确认.{0,16}(?:我)?(?:直接|立即|现在就)(?:取消|操作)",
            r"(?:但|但是|不过|实际上)(?:我(?:们)?)?(?:将|会)?(?:直接|立即|现在就)取消",
            r"(?:如果|假如|若)(?:我)?请(?:您)?明确告知(?:准确|正确|具体)?订单号",
        ),
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


# Keep the previous contract available for reproducing historical v11 rewards.
# New runs use the corrected rules; recorded reports are never relabelled in place.
INTENT_RULES_V11 = dict(INTENT_RULES)
INTENT_RULES["latest_order_identified"] = replace(
    INTENT_RULES_V11["latest_order_identified"],
    groups=(
        INTENT_RULES_V11["latest_order_identified"].groups[0]
        + ("最近订单",),
    ),
)
INTENT_RULES["refund_already_completed"] = replace(
    INTENT_RULES_V11["refund_already_completed"],
    patterns=(
        *INTENT_RULES_V11["refund_already_completed"].patterns,
        r"(?:该|这(?:个|件)|此)?(?:商品|订单)?已(?:经)?退过款",
        r"已(?:经)?退过(?:并)?退款",
    ),
    reject_patterns=(
        *INTENT_RULES_V11["refund_already_completed"].reject_patterns,
        r"(?:如果|假如|假设|若|未|尚未|没有|并未|并非|不是).{0,16}(?:退过款|退过并退款)",
    ),
)
INTENT_RULES["duplicate_refund_blocked"] = replace(
    INTENT_RULES_V11["duplicate_refund_blocked"],
    patterns=(
        r"(?:没有|未)(?:再|再次)创建(?:新的|重复)?退款",
        r"(?:没有|未)创建(?:新的|重复)退款",
    ),
    reject_patterns=(
        r"(?:并非|不是)(?:没有|未)(?:再|再次)?创建(?:新的|重复)?退款",
        r"(?:但|但是|不过|实际上)(?:我(?:们)?)?(?:会|将|已(?:经)?|正在|立即)?(?:再|再次)?创建(?:新的|重复)?退款",
    ),
)
INTENT_RULES["refund_to_original_payment"] = replace(
    INTENT_RULES_V11["refund_to_original_payment"],
    patterns=(
        r"(?:退款|款项|退回).{0,60}原(?:支付|付款)(?:渠道|路径)",
        r"原(?:支付|付款)(?:渠道|路径).{0,40}(?:退款|退回|返回)",
    ),
    reject_patterns=(
        r"(?:不|未|尚未|没有|并未|不能|无法|并非|不是)(?:按|向|通过|退回至|退至|已)?原(?:路|支付|付款)",
        r"(?:原路|原(?:支付|付款)(?:方式|渠道|路径|账户)).{0,16}(?:退款失败|退回失败|不能退款|无法退款)",
        r"(?:如果|假如|假设|若).{0,16}(?:原路|原(?:支付|付款)(?:方式|渠道|路径|账户))",
        r"(?:退款|款项).{0,12}(?:不|未|尚未|没有|并未)(?:会|将)?(?:按)?原(?:路|支付|付款)",
        r"(?:改为|改走|而是|实际退至|实际退到|实际上退至|实际上退到)(?:其他|另一个|另一|新)(?:账户|渠道|方式)",
    ),
)
INTENT_RULES["human_escalation_created"] = replace(
    INTENT_RULES_V11["human_escalation_created"],
    groups=(tuple(value for value in INTENT_RULES_V11["human_escalation_created"].groups[0]
                  if value != "交由人工处理"),),
    patterns=(r"已(?:经)?(?:为您)?(?:成功)?(?:升级|转交|转接|转|交由)(?:至|给|到)?人工(?:支持|客服|处理)",),
    reject_patterns=(
        r"(?:并非|不是|未|尚未|没有|并未)(?:已(?:经)?)?(?:为您)?(?:升级|转交|转接|转)(?:至|给|到)?人工",
        r"(?:计划|准备|将|会|需要|可以|如果|假如|若)(?:为您)?(?:升级|转交|转接|转)(?:至|给|到)?人工",
        r"(?:升级|转交|转接|转)(?:至|给|到)?人工(?:支持|客服|处理)?(?:失败|未成功)",
        r"(?:未|尚未|没有|并未|不是|并非)(?:已(?:经)?)?(?:为您)?(?:创建|生成)(?:了)?人工工单",
        r"人工工单(?:尚未|未|没有)(?:创建|生成)",
        r"(?:如果|假如|假设|若)(?:已(?:经)?)?(?:为您)?(?:升级|转交|转接|转|交由)(?:至|给|到)?人工",
        r"(?:如果|假如|假设|若)(?:已(?:经)?)?(?:为您)?(?:创建|生成)(?:了)?人工工单",
        r"(?:将|准备|计划)(?:交由人工处理|创建人工工单|生成人工工单)",
    ),
)


def missing_communication_intents(
    required: tuple[str, ...], response: str, *, legacy_v11: bool = False,
) -> list[str]:
    rules = INTENT_RULES_V11 if legacy_v11 else INTENT_RULES
    return [
        intent
        for intent in required
        if intent not in rules or not rules[intent].matches(response)
    ]
