"""Opt-in v17 strict, bounded extensions of frozen v15; no task-ID matching.

This is a deterministic business parser, not a learned/general semantic judge.
Only explicit statements, a locally resolved action, and a one-item refund
counteroffer count. Merely saying a two-item request is inconsistent does not.
The verifier grounds every newly resolved intent in owned tool evidence.
"""
from dataclasses import replace
import re

from . import communication_v15 as frozen

SPECS = frozen.SPECS
FULL_REFUND = r"(?:已(?:经)?)(?:取消并|完成取消和)?(?:全额|全部|足额)(?:退款|退还)"
REFUND_COMPLETED_PAST = r"(?:已(?:经)?退过款|已(?:经)?退过(?:并)?退款)"
QUANTITY_ONE = (r"(?:订单|商品)(?:中)?(?:实际)?数量(?:确实)?(?:仅|只)?(?:为|是|有|只有)(?:1|一)(?![\d.一二三四五六七八九十])(?:件)?"
                r"|订单(?:中)?(?:实际)?(?:只有|仅有|只购买了?)(?:1|一)件商品")
QUANTITY_OTHER = (r"(?:订单|商品)(?:中)?(?:实际)?数量(?:确实)?(?:为|是|有|只有)(?:2|两|二|[3-9]|[1-9]\d+)(?:件)?(?!\d)"
                  r"|订单(?:中)?(?:实际)?(?:只有|仅有|只购买了?)(?:2|两|二|[3-9]|[1-9]\d+)件商品")
STOP_EXPLICIT = r"(?:不能|无法|不允许|不可)(?:直接|现在|目前|继续|擅自|为您|帮您){0,3}" + frozen.REFUND_ACTION
OVERQUANTITY_BLOCK = r"(?:不能|无法|不允许|不可)(?:进行|申请|办理|创建)?超额(?:退款|退货)"
COUNTEROFFER = r"请(?:您)?确认是否(?:继续)?按(?:1|一)件(?:商品)?(?:申请|办理)?(?:退款|退货)"
ORDER_QUESTION = r"(?:您|你)?(?:确认要|想要|想|确定要)?取消(?:哪(?:个|一(?:个|笔))|吗)"
DRAFT = r"我(?:需要|应该|必须|应当)(?:明确)?(?:拒绝|解释|询问|告知|说明|停止)"


def parse_clauses(response):
    clauses = list(frozen.parse_clauses(response))
    for index, clause in enumerate(clauses):
        reasons = set(clause.exclusions)
        if re.search(DRAFT, clause.folded):
            reasons.add("future_only")  # intent to explain is not an explanation
        # A refusal after acknowledging a REQUEST is the assistant's refusal.
        # A quotation/report of what someone SAID must not get this reset.
        previous = [c for c in clauses[:index] if c.sentence == clause.sentence]
        if reasons == {"reported"} and re.match(r"(?:但|不过)?" + OVERQUANTITY_BLOCK, clause.folded):
            if not any(frozen.REPORTER.search(p.text) and re.search(r"说|声称|表示|提到|承诺|保证", p.folded)
                       for p in previous) and any(
                    re.match(r"(?:用户|客户|您|你)(?:要求|希望|想|请求).*(?:退款|退货)", p.folded)
                    and not re.search(r"说|声称|表示|提到", p.folded) for p in previous):
                reasons.discard("reported")
        clauses[index] = replace(clause, exclusions=tuple(sorted(reasons)))
    return tuple(clauses)


def _local_antecedents(clauses, clause, distance=1):
    # At most the preceding sentence; never search arbitrary earlier history.
    return [c for c in clauses if c.index < clause.index and c.sentence >= clause.sentence - distance
            and (c.asserted or (not c.exclusions and re.match(r"(?:我|我们)(?:已列出|看到|查到)", c.folded)
                               and not re.search(r"吗|哪|是否|是不是|能否", c.folded)))]


def _add(result, clause, match, reason, antecedent=None):
    evidence = frozen._evidence(clause, match, "v17_" + reason)
    if antecedent is not None:
        evidence.update(antecedent_clause_index=antecedent.index, antecedent_text=antecedent.text)
    result["support"].append(evidence)


def recognize_intent(intent, response, *, clauses=None):
    clauses = parse_clauses(response) if clauses is None else clauses
    result = frozen.recognize_intent(intent, response, clauses=clauses)
    if intent not in SPECS:
        return result
    for clause in clauses:
        text = clause.folded
        previous = _local_antecedents(clauses, clause)
        if clause.asserted:
            pattern = {"refund_already_completed": rf"(?:{FULL_REFUND}|{REFUND_COMPLETED_PAST})",
                       "purchased_quantity_one": QUANTITY_ONE,
                       "operation_stopped": STOP_EXPLICIT,
                       "refund_quantity_two_rejected": OVERQUANTITY_BLOCK}.get(intent)
            match = re.search(pattern, text) if pattern else None
            if match and not frozen._negative_prefix(text, match.start()):
                _add(result, clause, match, "explicit_expression")
            if intent == "operation_stopped" and re.fullmatch(r"(?:但|因此|所以)?(?:不能|无法|不允许)(?:直接|现在|继续)?执行", text):
                antecedent = next((c for c in reversed(previous) if re.search(r"退款(?:条件|操作|申请|资格|流程)?", c.folded)
                                   and not re.search(r"修改|地址|取消订单|发货|工单", c.folded)), None)
                if antecedent and not any(re.search(r"地址|工单|取消订单|取消操作|扣款操作", c.folded) for c in previous):
                    _add(result, clause, re.search(r"(?:不能|无法|不允许).*执行", text), "resolved_refund_stop", antecedent)
            if intent == "shipped_address_change_blocked":
                antecedent = next((c for c in reversed(previous) if c.sentence == clause.sentence
                                   and re.search(r"地址(?:修改|更改)?|修改地址", c.folded)), None)
                match = re.search(r"(?:当前|本次)?(?:状态(?:为|是)?)?shipped(?:不能|无法|不可)(?:修改|更改|改)", text)
                if antecedent and match:
                    _add(result, clause, match, "resolved_address_block", antecedent)
        if not clause.exclusions and clause.question:
            if intent == "order_id_confirmation_needed":
                match = re.search(ORDER_QUESTION, text)
                antecedent = next((c for c in reversed(previous) if re.search(r"订单|ord_[a-z0-9]+", c.folded)), None)
                if match and antecedent:
                    # A yes/no question needs a unique order; which-order
                    # requests may refer to a list. Tool grounding is separate.
                    unique = re.search(r"只有(?:1|一)(?:个|笔)订单|(?:这个|该|此)订单|ord_[a-z0-9]+", antecedent.folded)
                    if "哪" in match.group() or unique:
                        _add(result, clause, match, "resolved_order_question", antecedent)
            elif intent == "refund_quantity_two_rejected":
                match = re.search(COUNTEROFFER, text)
                # A quantity statement, acknowledgement of the customer's
                # request, then a counteroffer is a bounded three-sentence unit.
                nearby = _local_antecedents(clauses, clause, distance=2)
                antecedent = next((c for c in reversed(nearby) if re.search(QUANTITY_ONE, c.folded)
                                   or re.search(SPECS["purchased_quantity_one"].support, c.folded)), None)
                if match and antecedent:
                    _add(result, clause, match, "one_item_counteroffer_pending_confirmation", antecedent)
        if clause.contradiction_scope:
            pattern = None
            if intent == "purchased_quantity_one":
                pattern = QUANTITY_OTHER
            elif intent == "refund_quantity_two_rejected":
                pattern = r"(?:我|我们|系统)?(?:已(?:经)?|正在|会|将)(?:为您|帮您)?(?:按|为)?(?:2|两|二)件(?:商品)?(?:申请|办理|执行|创建)?(?:退款|退货)"
            match = re.search(pattern, text) if pattern else None
            if match and not frozen._negative_prefix(text, match.start()):
                result["contradictions"].append(frozen._evidence(clause, match, "v17_opposite_claim"))
    result["passed"] = bool(result["support"]) and not result["contradictions"]
    result["status"] = "contradicted" if result["contradictions"] else "supported" if result["passed"] else "unrecognized"
    return result


def recognize_required(required, response):
    clauses = parse_clauses(response)
    return [recognize_intent(intent, response, clauses=clauses) for intent in required if intent in SPECS]
