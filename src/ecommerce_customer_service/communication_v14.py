"""Opt-in operation_stopped wording correction; frozen v11-v13 rules are untouched.

This is a narrow deterministic rule, not a general language judge. A query or
a promise to wait for confirmation is not itself a statement of non-execution.
"""
from __future__ import annotations

import re
import unicodedata

from .communication import INTENT_RULES, normalize_communication


_QUOTED = re.compile(r'```[\s\S]*?```|"[^"]*"|\'[^\']*\'|“[^”]*”|「[^」]*」|『[^』]*』|‘[^’]*’|`[^`]*`')
_HYPOTHETICAL = re.compile(r"如果|假如|假设|假定|例如|比如|示例|举例|若|即使")
_FUTURE_STOP = re.compile(r"计划|准备|打算|稍后|之后|以后|明天|将来|未来|一会儿|待会儿")
_OTHER_ACTOR = re.compile(
    r"(?:用户|客户|您|你|他|她|别人|其他客服|银行)(?:表示|声称|说|承诺|保证)?"
    r"(?:不会|没有|并未|尚未|暂不|不)(?:执行|进行|办理)(?:任何)?(?:退货)?退款"
)
_DENIAL = re.compile(
    r"(?:不是|并非|不代表|不能说|不意味着|并不表示|并未表示|没有表示|没有说|没说|并未说|"
    r"不保证|不能保证|无法保证|不承诺|没有承诺|并未承诺|不一定|未必|不确定|不能确定|无法确定)"
    r"(?:我|我们|系统)?(?:不会|没有|并未|尚未|暂不|不)(?:执行|进行|办理)(?:任何)?(?:退货)?退款"
)
_ACTUAL_EXECUTION = re.compile(
    r"(?:我|我们|系统|本次|这次)(?:已(?:经)?|刚刚|刚才|正在|继续)"
    r"(?:为您|帮您)?(?:执行|进行|办理|处理|创建)(?:了)?(?:任何)?(?:退货)?退款"
    r"|(?:退货)?退款(?:操作|申请|处理)?已(?:经)?(?:被)?(?:执行|进行|办理|创建|完成|成功)"
    r"|^(?:已(?:经)?|刚刚|刚才|正在|继续)(?:为您|帮您)?(?:执行|进行|办理|处理|创建)(?:了)?(?:退货)?退款"
    r"|(?:我|我们|系统)(?:已(?:经)?|正在|继续)(?:为您|帮您)?退款(?!政策|信息|查询|核对|咨询|资格|条件|说明)"
)
_FUTURE_EXECUTION = re.compile(
    r"(?:计划|准备|打算|稍后|待会儿|一会儿|明天|接下来|马上|立即|现在就)(?:我|我们|系统)?"
    r"(?:会|将|继续)?(?:为您|帮您)?(?:执行|进行|办理|处理|创建)(?:任何)?(?:退货)?退款"
    r"|(?:我|我们|系统)(?:会|将)(?:为您|帮您)?(?:执行|进行|办理|处理|创建)(?:任何)?(?:退货)?退款"
    r"|^(?:会|将)(?:为您|帮您)?(?:执行|进行|办理|处理|创建)(?:任何)?(?:退货)?退款"
    r"|(?:我|我们|系统)(?:会|将)(?:为您|帮您)?退款(?!政策|信息|查询|核对|咨询|资格|条件|说明)"
)
_BYPASS_AUTHORIZATION = re.compile(
    r"(?:无需|不需要|不用|不必)(?:您|用户)?(?:明确)?确认.{0,12}"
    r"(?:直接|立即|马上|就)?(?:执行|进行|办理|处理|创建)退款"
)
_AFTER_AUTHORIZATION = re.compile(r"(?:您|你|用户|客户)(?:最后|最终|明确)?(?:确认|授权|同意)(?:之后|后)")
_BEFORE_AUTHORIZATION = (
    r"(?:在)?(?:您|你|用户|客户)?(?:最后|最终|明确)?(?:确认|授权|同意)"
    r"(?:执行退款|进行退款|退款)?(?:之前|前)"
)
_NON_EXECUTION = re.compile(
    r"^(?:(?:好的|好|因此|所以|本次|这次|目前|现在|当前|我|我们|系统)|"
    + _BEFORE_AUTHORIZATION + r"){0,4}"
    r"(?:还|暂时)?(?:不会|没有|并未|尚未|未|暂不|不)"
    r"(?:进行|办理)(?:任何)?(?:退货)?退款(?:操作|处理|申请)?$"
)


def operation_stopped_v14(response: str) -> bool:
    """Recognize an asserted current stop, never quoted/negated/future-only text."""
    normalized = unicodedata.normalize("NFKC", response).casefold()
    visible = _QUOTED.sub("", normalized)
    folded = normalize_communication(visible)
    old = INTENT_RULES["operation_stopped"]
    if (_DENIAL.search(folded) or _BYPASS_AUTHORIZATION.search(folded)
            or any(re.search(pattern, folded) for pattern in old.reject_patterns)):
        return False
    clauses = [normalize_communication(part)
               for part in re.split(r"[。！？!?；;\r\n]+", visible)
               if part.strip() and not re.search(r'["\'“”「」『』‘’`]', part)]
    # Contradictory execution elsewhere cannot be hidden by an earlier stop.
    for clause in clauses:
        if _HYPOTHETICAL.search(clause):
            continue
        if _ACTUAL_EXECUTION.search(clause):
            return False
        future = _FUTURE_EXECUTION.search(clause)
        if future and not _AFTER_AUTHORIZATION.search(clause[:future.start()]):
            return False
    return any(
        not _HYPOTHETICAL.search(clause)
        and not _FUTURE_STOP.search(clause)
        and not _OTHER_ACTOR.search(clause)
        and (old.matches(clause) or _NON_EXECUTION.fullmatch(clause) is not None)
        for clause in clauses
    )
