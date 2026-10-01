from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecommerce_customer_service.models import TaskSpec
from ecommerce_customer_service.simulator import OpenAICompatibleAgent, run_task
from ecommerce_customer_service.tasks import build_task_dataset, load_task_paths


@unittest.skipUnless(importlib.util.find_spec("langgraph"), 'install the optional runtime with pip install -e ".[agent]"')
class LangGraphAgentTest(unittest.TestCase):
    def test_multiturn_cancel_matches_native_rollout(self) -> None:
        from ecommerce_customer_service.langgraph_agent import LangGraphCustomerServiceAgent

        with tempfile.TemporaryDirectory(prefix="ecommerce-cs-graph-test-") as temporary:
            root = Path(temporary)
            build_task_dataset(root / "dataset", replicas=1, seed=7)
            task = next(
                TaskSpec.from_file(path)
                for path in load_task_paths(root / "dataset" / "tasks.jsonl")
                if TaskSpec.from_file(path).scenario == "cancel_unshipped"
            )
            identity = task.metadata["identity"]
            order_id = task.metadata["order_id"]

            def tool_call(name: str, arguments: dict[str, object], index: int) -> dict[str, object]:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                        }
                    ],
                }

            responses = [
                {"role": "assistant", "content": "请提供姓名和手机号后四位。"},
                tool_call("verify_customer_identity", identity, 0),
                tool_call("get_order", {"customer_id": task.user_id, "order_id": order_id}, 1),
                {"role": "assistant", "content": "退款金额为 219.00 元，请确认是否取消订单。"},
                tool_call("cancel_order", {"customer_id": task.user_id, "order_id": order_id, "confirmed": True}, 2),
                {"role": "assistant", "content": "订单已取消，219.00 元将原路退款。"},
            ]

            def make_agent(agent_class: type[OpenAICompatibleAgent]) -> OpenAICompatibleAgent:
                agent = agent_class("stub-model", max_steps=10, correction_retries=0)
                pending = iter(responses)

                def complete(messages: list[dict], tool_schemas: list[dict]) -> dict:
                    response = next(pending)
                    available = {item["function"]["name"] for item in tool_schemas}
                    for call in response.get("tool_calls", []):
                        self.assertIn(call["function"]["name"], available)
                    return response

                agent._complete = complete  # type: ignore[method-assign]
                return agent

            native = run_task(task, make_agent(OpenAICompatibleAgent))
            graphed = run_task(task, make_agent(LangGraphCustomerServiceAgent))
            self.assertEqual(native.reward, 1.0)
            self.assertEqual(graphed.reward, native.reward)
            self.assertEqual(graphed.verifier.to_dict(), native.verifier.to_dict())
            self.assertEqual(graphed.tool_executions, native.tool_executions)
            self.assertEqual(graphed.messages, native.messages)
            self.assertEqual(graphed.stopped_reason, native.stopped_reason)

