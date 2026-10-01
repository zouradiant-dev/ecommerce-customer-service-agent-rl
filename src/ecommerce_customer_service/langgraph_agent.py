from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .models import TaskSpec
from .simulator import AgentRun, OpenAICompatibleAgent, SYSTEM_PROMPT
from .tools import CustomerServiceTools


class _ConversationState(TypedDict):
    messages: list[dict[str, Any]]
    scripted_replies: list[dict[str, Any]]
    corrections_used: list[str]
    assistant: dict[str, Any]
    final_response: str
    stopped_reason: str
    model_steps: int
    continue_run: bool


class LangGraphCustomerServiceAgent(OpenAICompatibleAgent):
    """Run the existing business protocol as explicit LangGraph state transitions.

    The graph owns the model/tool/user-turn loop. Business authorization and
    verification remain in CustomerServiceTools and the shared rollout runner.
    A graph is compiled per rollout because its nodes close over that rollout's
    isolated tool instance. No checkpointer is used for these side-effecting
    evaluation runs.
    """

    def run(self, task: TaskSpec, tools: CustomerServiceTools) -> AgentRun:
        initial_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.user_prompt},
        ]
        if self.max_steps <= 0:
            return AgentRun(tuple(initial_messages), "", "max_steps")

        def model_turn(state: _ConversationState) -> dict[str, Any]:
            assistant = dict(self._complete(state["messages"], tools.available_schemas()))
            assistant["role"] = "assistant"
            return {
                "assistant": assistant,
                "messages": [*state["messages"], assistant],
                "model_steps": state["model_steps"] + 1,
            }

        def route_model(state: _ConversationState) -> str:
            return "execute_tools" if state["assistant"].get("tool_calls") else "handle_response"

        def execute_tools(state: _ConversationState) -> dict[str, Any]:
            messages = list(state["messages"])
            for index, call in enumerate(state["assistant"].get("tool_calls") or []):
                function = call.get("function") or {}
                name = str(function.get("name") or call.get("name") or "")
                arguments = function.get("arguments") or call.get("arguments") or "{}"
                execution = tools.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"call_{state['model_steps'] - 1}_{index}"),
                        "name": name,
                        "content": json.dumps(execution.output, ensure_ascii=False),
                    }
                )
            return {
                "messages": messages,
                "stopped_reason": "max_steps" if state["model_steps"] >= self.max_steps else "model_finished",
            }

        def handle_response(state: _ConversationState) -> dict[str, Any]:
            final = str(state["assistant"].get("content") or "")
            messages = list(state["messages"])
            replies = list(state["scripted_replies"])
            corrections = list(state["corrections_used"])
            continue_run = False

            reply_index = next(
                (
                    index
                    for index, reply in enumerate(replies)
                    if self._assistant_requests_reply(str(reply.get("kind") or ""), final)
                ),
                None,
            )
            if reply_index is not None:
                reply = replies.pop(reply_index)
                content = str(reply.get("content") or "").strip()
                messages.append({"role": "user", "content": content})
                tools.observe_user_message(content)
                continue_run = True
            else:
                correction = self._correction_for_state(tools, replies)
                if correction and correction[0] not in corrections and len(corrections) < self.correction_retries:
                    corrections.append(correction[0])
                    messages.append({"role": "system", "content": correction[1]})
                    continue_run = True

            return {
                "messages": messages,
                "scripted_replies": replies,
                "corrections_used": corrections,
                "final_response": final,
                "continue_run": continue_run,
                "stopped_reason": "max_steps" if continue_run and state["model_steps"] >= self.max_steps else "model_finished",
            }

        def route_next(state: _ConversationState) -> str:
            if state["model_steps"] >= self.max_steps:
                return END
            return "model_turn"

        def route_response(state: _ConversationState) -> str:
            return "model_turn" if state["continue_run"] and state["model_steps"] < self.max_steps else END

        graph = StateGraph(_ConversationState)
        graph.add_node("model_turn", model_turn)
        graph.add_node("execute_tools", execute_tools)
        graph.add_node("handle_response", handle_response)
        graph.add_edge(START, "model_turn")
        graph.add_conditional_edges("model_turn", route_model)
        graph.add_conditional_edges("execute_tools", route_next)
        graph.add_conditional_edges("handle_response", route_response)
        result = graph.compile().invoke(
            {
                "messages": initial_messages,
                "scripted_replies": [dict(item) for item in task.metadata.get("scripted_user_replies") or []],
                "corrections_used": [],
                "assistant": {},
                "final_response": "",
                "stopped_reason": "max_steps",
                "model_steps": 0,
                "continue_run": False,
            },
            config={"recursion_limit": 4 * self.max_steps + 10},
        )
        return AgentRun(tuple(result["messages"]), result["final_response"], result["stopped_reason"])
