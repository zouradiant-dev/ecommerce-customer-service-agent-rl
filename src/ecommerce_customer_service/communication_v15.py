"""Opt-in, explainable clause-level grading for eleven boundary intents.

This is a bounded deterministic semantic parser, not general language
understanding. Unknown wording fails closed. It never mutates older intent
rules and never takes an expected answer or task ID as matching input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Clause:
    index: int
    sentence: int
    text: str
    folded: str
    question: bool
    exclusions: tuple[str, ...]

    @property
    def asserted(self):
        return not self.question and not self.exclusions

    @property
    def contradiction_scope(self):
        return not self.question and not set(self.exclusions) & {"reported", "hypothetical", "foreign_target"}


QUOTES = re.compile(r'```[\s\S]*?```|`[^`]*`|"[^"]*"|\'[^\']*\'|“[^”]*”|「[^」]*」|『[^』]*』|‘[^’]*’')
ATOM = re.compile(r"(?:[a-z][a-z0-9_-]*|\d+(?:\.\d+)?(?:件|元|分)?|已取消|未取消|已发货|未发货|已签收|未签收|已送达|未送达)")
HYPOTHETICAL = re.compile(r"如果|假如|假设|假定|即使|例如|比如|示例|举例|比方|(?:^|[，,])若")
FUTURE = re.compile(r"计划|准备|打算|稍后|明天|将来|未来|一会儿|待会儿|接下来|之后才|以后才|将(?:停止|暂停|终止)|^(?:在|等|待)(?:您|用户|客户)?(?:确认|授权|同意)(?:之后|后)")
UNCERTAIN = re.compile(r"可能|也许|不一定|未必|不确定|不能保证|无法保证|不能确定|无法确定|大概|理论上|通常")
DENIED = re.compile(r"并非|不是|不代表|不意味着|并不表示|没有说|没说|并未说|没有表示|并未表示|不能说|不保证|不承诺")
REPORTER = re.compile(r"^(?:但|而|不过)?(?:用户|客户|您|你|他|她|别人|其他客服|银行|规则|提示|文档)(?:曾经|之前|现在)?(?:说|提到|写着|表示|声称|承诺|保证|要求)")
OTHER_ACTION = re.compile(r"^(?:用户|客户|您|你|他|她|别人|其他客服|银行)(?:自己|本人)?(?:不会|没有|并未|尚未|未|暂不|不|已|已经|会|将)(?:自动|继续|擅自|直接|现在|为我)?(?:执行|进行|办理|处理|创建|退款|暂停)")
FOREIGN = re.compile(r"另一(?:笔|个)?订单|其他订单|别人的订单|其他客户")
AUTHORIZATION = re.compile(r"(?:只有|等待|等|待|在)?(?:您|你|用户|客户)(?:最后|最终|明确)?(?:确认|授权|同意)(?:执行退款|退款|取消)?(?:之后|后|才能)")
REFUND_ACTION = r"(?:执行|进行|办理|处理|创建|发起)(?:了)?(?:任何)?(?:退货退款|退款|退货)(?!政策|信息|查询|核对|咨询|资格|条件|说明|操作说明|操作示例|操作查询)"
ADDRESS_ACTION = r"(?:修改|更改|变更|更新)(?:收货|订单|配送)?地址"
PERMISSION = r"(?:可以|允许|支持|能够|能|可)(?:直接|现在|继续|尝试|为您|帮您|通过系统|进行|执行|申请|创建|办理|按|到){0,4}"
BLOCK = r"(?:不能|无法|不允许|不可以|不支持|不符合|不可|禁止|拒绝|不再|不)(?:直接|现在|在此时|此时|为您|帮您|通过系统工具|通过系统|通过工具|进行|执行|申请|办理|创建|提前|继续|再){0,5}"


def fold(text):
    return re.sub(r"[\s*`\"'“”「」『』‘’()（）:：]", "", unicodedata.normalize("NFKC", text).casefold())


def _visible(text):
    def replace(match):
        value = match.group()
        if value.startswith("```"):
            return ""
        inside = value[1:-1].strip()
        return inside if ATOM.fullmatch(inside) else ""
    return QUOTES.sub(replace, unicodedata.normalize("NFKC", text).casefold())


def parse_clauses(response):
    visible = _visible(response)
    clauses = []
    for sentence_number, sentence_match in enumerate(re.finditer(r"[^。！？!?；;\r\n]+[。！？!?；;]?", visible)):
        sentence = sentence_match.group()
        question = sentence.rstrip().endswith(("?", "？"))
        inherited = set()
        for fragment in re.split(r"[，,:：]+", sentence):
            text = fragment.strip(" 。！？!?；;-•")
            if not text:
                continue
            reasons = set(inherited)
            if re.match(r"(?:但|然而|不过|而)(?:我|我们|订单|实际|目前|本次)", text):
                # Explicit contrast switches from a report to the assistant's
                # own statement. A hypothetical scope is deliberately retained.
                reasons.discard("reported")
                reasons.discard("denied_assertion")
            if re.match(r"请|我|我们|因此|所以|目前|本次", text):
                reasons.discard("denied_assertion")
            if re.match(r"请(?:问|确认|提供|告知|明确)", text) and any(
                    c.sentence == sentence_number and re.match(r"(?:但)?您(?:之前|现在)?(?:提到|表示).+", c.text)
                    for c in clauses):
                # A new direct request after acknowledging the customer's doubt
                # is the assistant's request, not a continuation of that doubt.
                reasons.discard("reported")
            if re.search(r'["\'“”「」『』‘’`]', text):
                reasons.add("reported")  # unmatched quotation: fail closed
            customer_history_question = question and re.match(r"您(?:之前)?提到.{0,60}(?:订单|ord_[a-z0-9]+).*吗$", fold(text))
            if REPORTER.search(text) and not customer_history_question:
                reasons.add("reported")
                inherited.add("reported")
            if re.search(r"(?:我理解|我听说|我了解到).{0,8}(?:您|用户|客户)|(?:您|用户|客户)(?:希望|期待|请求)|我(?:希望|期待)", text):
                reasons.add("reported")
            if OTHER_ACTION.search(text):
                reasons.add("reported")
            if HYPOTHETICAL.search(text):
                reasons.add("hypothetical")
                inherited.add("hypothetical")
            if FOREIGN.search(text):
                reasons.add("foreign_target")
            if FUTURE.search(text):
                reasons.add("future_only")
                inherited.add("future_only")
            if UNCERTAIN.search(text):
                reasons.add("uncertain")
            if DENIED.search(text):
                reasons.add("denied_assertion")
                inherited.add("denied_assertion")
            clauses.append(Clause(len(clauses), sentence_number, text, fold(text), question, tuple(sorted(reasons))))
    return tuple(clauses)


def _match(pattern, text):
    return re.search(pattern, text)


def _negative_prefix(text, start):
    return bool(re.search(r"(?:不会|没有|并未|尚未|未|不|不能|无法|并非|不是)(?:自动|继续|擅自|直接|为您|帮您|进行|执行)?$", text[:start]))


def _positive_permission(text, action):
    for match in re.finditer(PERMISSION + action, text):
        if not _negative_prefix(text, match.start()):
            return match
    return None


def _active_execution(clause, action):
    text = clause.folded
    pattern = r"(?:我|我们|系统|本次|这次)?(?:已(?:经)?|刚刚|刚才|正在|继续|会|将|马上|立即|现在就)(?:为您|帮您|直接|立即|现在就|继续){0,3}" + action
    for match in re.finditer(pattern, text):
        if _negative_prefix(text, match.start()):
            continue
        before = text[:match.start()]
        if AUTHORIZATION.search(before) and not re.search(r"已(?:经)?|刚刚|刚才|正在", match.group()):
            continue
        return match
    return None


@dataclass(frozen=True)
class IntentSpec:
    support: str
    opposite: str = r"(?!)"
    prerequisite: str | None = None
    request: bool = False


SPECS = {
    "operation_stopped": IntentSpec(
        r"(?:不会|没有|并未|尚未|未|暂不|不再|不)(?:自动|继续|擅自)?" + REFUND_ACTION
        + r"|(?:不会|没有|并未|尚未|未|暂不|不再|不)(?:执行|进行|办理|处理|创建)(?:任何|相关)?(?:写)?操作"
        + r"|(?:本次|这次)?(?:退款操作|退款流程|操作|处理|流程)(?:已(?:经)?)(?:停止|暂停|终止|取消)"
        + r"|(?:已(?:经)?)(?:停止|暂停|终止|取消)(?:本次|这次|当前)?(?:退款操作|退款流程|操作|处理|流程)"
        + r"|(?:本次|这次)?未执行退款(?!政策|信息|查询|说明|操作说明)|不会擅自退款"
        + r"|退款(?:操作|申请|处理)?(?:将)?(?:不会|未|没有)(?:被)?(?:执行|进行|办理|创建)"
        + r"|退款操作(?:将)?(?:保持|维持)暂停状态|您不需要我现在执行(?:任何)?(?:退款)?操作",
        r"(?:无需|不需要|不用|不必)(?:您|用户)?(?:明确)?确认.{0,12}(?:执行|进行|办理|处理|创建)退款"
        + r"|退款(?:操作|申请|处理)?已(?:经)?(?:被)?(?:执行|进行|办理|创建|完成|成功)",
    ),
    "shipped_address_change_blocked": IntentSpec(
        BLOCK + r"(?:修改|更改|变更|更新)(?:收货|订单|配送)?地址"
        + r"|(?:地址|地址修改|地址更改)(?:功能|操作)?(?:不可用|不允许|不能|无法|不可)(?:修改|更改|变更|更新)?",
        prerequisite=r"已发货|已经发货|运输中|配送中|状态.{0,8}shipped|处于shipped|订单.{0,24}shipped",
    ),
    "order_id_confirmation_needed": IntentSpec(
        r"(?:请|需要|是否|是不是|是|您想|您之前提到|想取消|确定|确认).{0,65}(?:订单|ord_[a-z0-9]+).{0,25}(?:吗|确认|提供)?"
        + r"|(?:请|需要).{0,8}(?:准确|正确|具体|目标)(?:的)?订单号"
        + r"|(?:订单|ord_[a-z0-9]+).{0,25}(?:吗|请确认|需要确认|请提供)",
        r"(?:无需|不用|不必|不要)(?:您)?(?:再|再次|现在)?(?:确认|提供|告知).{0,20}(?:订单|ord_[a-z0-9]+)",
        request=True,
    ),
    "purchased_quantity_one": IntentSpec(
        r"(?:实际)?(?:购买|买)(?:了|数量)?(?:确实)?(?:仅|只)?(?:为|是|有|只有)?(?:1|一)件"
        + r"|(?:订单|商品)(?:实际)?数量(?:确实)?(?:仅|只)?(?:为|是|有|只有)?(?:1|一)件"
        + r"|实际(?:仅|只)?(?:有|为|是|只有)(?:1|一)件",
        r"(?:实际)?(?:购买|买)(?:了|数量)?(?:为|是|有|只有)?(?:2|两|二|[3-9]|[1-9]\d+)件",
    ),
    "refund_quantity_two_rejected": IntentSpec(
        BLOCK + r".{0,12}(?:2|两|二)件.{0,12}(?:退款|退货)"
        + r"|(?:退款|退货).{0,12}" + BLOCK + r".{0,8}(?:2|两|二)件"
        + r"|(?:只|仅)(?:能|允许)(?:申请|办理|退还)?(?:1|一)件(?:商品)?(?:的)?(?:退款|退货)"
        + r"|(?:不能|不允许|不可以|不可)超额(?:到|按)?(?:2|两|二)件",
        r"(?:并非|不是)(?:不能|不允许|不可以|不支持).{0,12}(?:2|两|二)件"
        + r"|(?:不存在|没有|并未)(?:超额退款|超额退货)",
    ),
    "order_not_delivered": IntentSpec(
        r"(?:尚未|还未|还没有|没有|未|尚未完成|还未完成)(?:签收|送达)|仍在运输|正在运输中|状态.{0,8}shipped",
        r"(?:已(?:经)?)(?:完成)?(?:签收|送达)|状态.{0,8}delivered",
    ),
    "return_before_delivery_blocked": IntentSpec(
        BLOCK + r"(?:售后)?(?:退货退款|退货|退款)"
        + r"|(?:不符合|未达到|尚未达到).{0,8}(?:退货退款|退款退货|退货|退款)(?:的)?(?:条件|要求|状态)"
        + r"|(?:签收|送达)后(?:才|方)(?:能|可)(?:申请|办理|创建)?(?:退货|退款)",
    ),
    "return_window_expired": IntentSpec(
        r"(?:超过|超出|已过)(?:30天|退货期限|退货窗口)|退货(?:期|期限|窗口)(?:已|已经)?(?:过期|超过|已过)|超出期限|已过期",
        r"(?:仍可|仍然可以|可以正常)(?:退货|退款)|(?:未|尚未|没有)(?:超过|超出|已过)(?:30天|退货期限)",
        prerequisite=BLOCK + r"(?:正常)?(?:退货|退款)|不符合.{0,10}(?:退货|退款)|(?:退货|退款).{0,12}(?:被拒绝|不支持|不能|无法|不允许|不符合)",
    ),
    "order_cancelled": IntentSpec(
        r"订单.{0,75}(?:已(?:经)?(?:被)?取消|状态.{0,8}cancelled)|(?:订单)?已(?:经)?取消(?:订单)?|订单.{0,75}状态.{0,8}canceled",
        r"订单.{0,45}(?:尚未取消|未取消|没有取消|取消失败|状态.{0,8}(?:paid|pending))",
    ),
    "refund_already_completed": IntentSpec(
        r"(?:已(?:经)?(?:成功)?)(?:取消并|完成取消和)?(?:完成退款|退款)|退款已(?:经)?(?:成功|完成|处理完成)"
        + r"|已有退款记录|已经有退款记录|退款(?:记录|金额)?(?:的)?状态.{0,10}(?:completed|refunded|已完成)"
        + r"|(?:退款|款项|金额).{0,15}已(?:经)?(?:原路)?(?:退回|退还|返回)",
        r"(?:退款)(?:记录|金额|状态)?(?:为|是|显示为)?(?:尚未|未|没有|待|正在)?(?:完成|处理)?(?:失败|处理中|未完成|pending|processing|failed)"
        + r"|(?:没有|未|尚未)完成退款|尚未退款|没有退款记录",
    ),
    "no_duplicate_action": IntentSpec(
        r"(?:没有|未|不会|无需|不再|不需要|不必|不用)(?:进行任何|进行)?(?:重复|再次).{0,12}(?:操作|处理|退款|取消)"
        + r"|(?:操作|处理|退款|取消).{0,8}(?:没有|未|不会)(?:重复|再次)",
        r"(?:我|我们|系统)?(?:已(?:经)?|正在|会|将|继续)(?:进行)?(?:重复|再次)(?:取消|退款|操作|处理)",
    ),
}


def _evidence(clause, match, reason):
    return {"clause_index": clause.index, "text": clause.text,
            "matched": match.group() if match else None, "reason": reason}


def recognize_intent(intent, response, *, clauses=None):
    """Return supported/contradicted/unrecognized with auditable evidence."""
    clauses = parse_clauses(response) if clauses is None else clauses
    if intent not in SPECS:
        return {"intent": intent, "passed": False, "status": "unsupported_intent",
                "support": [], "contradictions": [], "excluded_clauses": []}
    spec = SPECS[intent]
    support, contradictions = [], []
    prerequisite = not spec.prerequisite or any(
        c.asserted and _match(spec.prerequisite, c.folded) for c in clauses)
    for clause in clauses:
        text = clause.folded
        sentence_prefix = "".join(c.folded for c in clauses if c.sentence == clause.sentence and c.index < clause.index)
        match = _match(spec.support, text)
        eligible = clause.asserted
        if spec.request:
            eligible = not clause.exclusions and (clause.question or bool(re.search(r"请|需要|是否|是不是|请问|能否", text)))
        if match and eligible and prerequisite:
            # A bare display of an order is not a clarification request.
            if intent != "order_id_confirmation_needed" or re.search(r"确认|提供|告知|吗|是否|是不是|请问|请明确", text):
                support.append(_evidence(clause, match, "asserted_intent"))
        if intent == "order_id_confirmation_needed" and eligible and re.fullmatch(r"请(?:问)?确认是否继续", text):
            # Resolve an adjacent explicit confirmation back to its proposed order,
            # not to an arbitrary earlier display of an order ID.
            previous = [c for c in clauses if c.sentence == clause.sentence - 1]
            proposal = next((c for c in previous if re.search(
                r"如果(?:您|用户|客户)确认.{0,16}(?:取消)?订单(?:ord_[a-z0-9]+)?", c.folded)), None)
            if proposal:
                support.append({**_evidence(clause, _match(r"确认是否继续", text), "adjacent_order_confirmation"),
                                "antecedent_clause_index": proposal.index, "antecedent_text": proposal.text})
        if not clause.contradiction_scope:
            continue
        opposite = _match(spec.opposite, text)
        if opposite and intent == "order_not_delivered":
            # "requires delivered items" is a rule, not a delivered-state claim.
            if (re.search(r"(?:需|需要|要求|只有|不符合).{0,14}$", text[:opposite.start()])
                    or re.match(r"(?:的)?(?:商品|条件|才能|才可|才可以)", text[opposite.end():])
                    or re.search(r"(?:退款|退货)政策(?:要求|条件)$", sentence_prefix)):
                opposite = None
        if opposite and not _negative_prefix(text, opposite.start()):
            contradictions.append(_evidence(clause, opposite, "opposite_claim"))
        if match and "denied_assertion" in clause.exclusions and not spec.request:
            contradictions.append(_evidence(clause, match, "denied_supporting_claim"))
        if intent == "operation_stopped":
            execution = _active_execution(clause, REFUND_ACTION + r"|(?:我|我们|系统)(?:已(?:经)?|正在|会|将)(?:为您|帮您)?退款(?!政策|信息|查询|资格)")
            if execution:
                contradictions.append(_evidence(clause, execution, "actual_or_unconditional_future_refund"))
        elif intent == "shipped_address_change_blocked":
            permission = _positive_permission(text, ADDRESS_ACTION)
            if permission and not re.search(r"(?:签收|送达|未发货|pending|paid)(?:之后|后|时)|仅允许|只允许|只有", text):
                contradictions.append(_evidence(clause, permission, "address_change_allowed"))
        elif intent == "refund_quantity_two_rejected":
            permission = _positive_permission(text, r"(?:2|两|二)件(?:商品)?(?:数量)?(?:的)?(?:申请|办理|创建)?(?:退款|退货)")
            permission = permission or _match(r"(?:退款|退货)" + PERMISSION + r"(?:2|两|二)件", text)
            if permission:
                contradictions.append(_evidence(clause, permission, "two_item_refund_allowed"))
        elif intent == "return_before_delivery_blocked":
            permission = _positive_permission(text, r"(?:售后)?(?:退货退款|退货|退款)")
            product_context = re.search(r"(?:商品|该商品|这个商品)(?:状态)?[^。!?]{0,55}$", sentence_prefix + text)
            if permission and product_context and not re.search(r"当前|现在|直接|创建|办理|申请|为您|帮您", text):
                permission = None  # product returnability is not current authorization
            if permission and not re.search(r"(?:签收|送达)(?:之后|后)|只有|才能", text):
                contradictions.append(_evidence(clause, permission, "refund_allowed_before_delivery"))
        elif intent == "order_id_confirmation_needed":
            execution = _active_execution(clause, r"(?:取消(?:这个|该|此|目标)?订单|(?:进行|执行)取消操作|(?:使用|按)ord_[a-z0-9]+(?:进行|执行)取消操作)")
            if execution:
                contradictions.append(_evidence(clause, execution, "cancellation_without_disambiguation"))
            confirmation = _match(r"(?:您|用户|客户)(?:已经|已|刚刚|现在又).{0,8}确认.{0,20}(?:订单|取消)", text)
            if confirmation:
                contradictions.append(_evidence(clause, confirmation, "unjustified_user_confirmation_claim"))
    passed = bool(support) and not contradictions
    return {"intent": intent, "passed": passed,
            "status": "contradicted" if contradictions else "supported" if passed else "unrecognized",
            "support": support, "contradictions": contradictions,
            "excluded_clauses": [asdict(c) for c in clauses if c.exclusions]}


def recognize_required(required, response):
    clauses = parse_clauses(response)
    return [recognize_intent(intent, response, clauses=clauses) for intent in required if intent in SPECS]
