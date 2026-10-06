# Evidence-Bound Policy Control for Reliable Tool-Using E-Commerce Customer-Service Agents

> **Project paper / working draft — October 2026.** This manuscript reports a reproducible engineering study conducted on the repository's frozen v18 development suite. It has not been peer reviewed. The reported 98.17% result is a **system-level development result** for a frozen SFT adapter plus an external policy controller; it is not a held-out test result, a production-readiness claim, or evidence of a successful GRPO update.

## Abstract

Tool-using language agents can produce fluent customer-service responses while performing an incorrect or unauthorized business action. Text-only evaluation therefore misses failures that matter in practice: modifying the wrong order, issuing an excessive refund, claiming success without a successful write, or continuing after escalation. We present a reproducible e-commerce after-sales agent environment in which every task owns an isolated SQLite state, all tool calls are audited, write operations change the database, and a deterministic verifier jointly evaluates state transitions, tool order, policy compliance, terminal decisions, and user-facing facts. The policy model is Qwen2.5-7B-Instruct adapted with QLoRA on 17,760 verifier-approved assistant turns generated from 3,600 tasks. We then add an evidence-bound controller that authorizes and resolves sensitive actions using only user messages, exposed tools, and audited tool outputs; it cannot access scenario labels or hidden expected answers. On a frozen 600-task development suite, the rescored SFT adapter passed 530/600 tasks (88.33%) with 13 hard-safety failures. The same frozen adapter combined with controller r6 passed 589/600 (98.17%) with zero hard-safety failures, zero worker errors, and zero step exhaustions. The 59 additional strict passes are attributable to the complete system, not to further model learning. Targeted continued-SFT variants caused cross-scenario interference and were rejected. An exploratory earlier GRPO checkpoint also failed to outperform its SFT control, while a later v18 calibration set was saturated and supplied no within-group advantage signal; no GRPO policy update is claimed. The study supports a narrow conclusion: deterministic, evidence-bound control can substantially improve development-set reliability for high-consequence tool actions, but independent test evaluation, controller ablations, repeated stochastic trials, and latency measurements remain necessary.

**Keywords:** tool-using agents, customer service, policy enforcement, executable evaluation, QLoRA, safety verification, GRPO gate

## 1. Introduction

Language-model agents interleave natural-language reasoning with calls to external systems. ReAct established the general value of combining reasoning and action, while later benchmarks such as $\tau$-bench showed that policy following and repeated reliability remain difficult in tool-agent-user interactions [1, 2]. In an e-commerce support setting, the cost of a plausible but false answer is not merely conversational: a mistaken tool call can cancel an order, alter an address, or create a refund.

This project studies a constrained but consequential question:

> Can an external, evidence-bound policy layer reduce unsafe and unverifiable actions by a learned customer-service agent without replacing the agent's language understanding and tool planning?

The system separates responsibilities. A learned Qwen2.5-7B-Instruct policy interprets requests, selects read tools, proposes writes, and explains outcomes. A deterministic business layer owns identity, object ownership, order-state transitions, confirmation, and authorization. A verifier judges the resulting trajectory against the actual database state rather than trusting the model's final message. The design follows a practical principle: the model may propose an action, but successful high-consequence actions must be established by tool evidence.

This paper makes four engineering contributions:

1. **A stateful, auditable task environment.** Thirty scenario families cover cancellation, address changes, refunds, order disambiguation, identity failures, transient and persistent tool errors, and adversarial policy-bypass requests. Each rollout receives an independent SQLite copy.
2. **An evidence-bound policy controller.** The controller checks sensitive write proposals and reconstructs structured terminal decisions from public user input and audited tool evidence, without reading hidden task fields.
3. **A dual evaluation and training reward contract.** A strict binary metric is reserved for acceptance, while a decomposed shaped reward is available for training and assigns a fixed negative reward to hard-safety failures.
4. **Failure-preserving experimental reporting.** Evaluator corrections, model changes, controller changes, and reinforcement-learning attempts are reported separately. Rejected training runs and a negative GRPO result remain part of the evidence.

The work does not claim a new foundation model, a general safety guarantee, or state-of-the-art performance on an external benchmark. Its value lies in the executable environment, the evidence contract, and a documented model-versus-system comparison.

## 2. Related Work

### 2.1 Reasoning and tool use

ReAct interleaves reasoning traces and actions so that a model can update plans from environmental observations [1]. This project uses the same broad interaction pattern but evaluates success through database mutations, tool traces, and grounded terminal facts rather than text quality alone. $\tau$-bench similarly emphasizes conversations with users, domain policies, tools, end-state evaluation, and repeated reliability [2]. Our environment specializes these ideas for e-commerce after-sales workflows and exposes the policy and verifier implementation for local audit.

### 2.2 Parameter-efficient adaptation

LoRA freezes pretrained weights and learns low-rank update matrices [3]. QLoRA extends this approach by backpropagating through a quantized frozen model into LoRA adapters [4]. We use QLoRA to adapt Qwen2.5-7B-Instruct [5] with a single-GPU, 4-bit NF4 configuration. The objective is to teach the tool protocol and business workflow while retaining an immutable base revision and a hash-addressed adapter artifact.

### 2.3 Safety evaluation for tool agents

ToolEmu demonstrates that tool-enabled agents can cause consequential failures and motivates explicit risk-oriented testing [6]. AgentDojo further treats tool-agent security as an environment problem with executable tasks and adaptive attacks [7]. Our study addresses a different threat surface—business-rule violations, unsupported success claims, and incorrect object binding—but shares the view that agent safety must be tested through actions and state, not only through generated prose.

### 2.4 Group-relative reinforcement learning

Group Relative Policy Optimization (GRPO) was introduced in DeepSeekMath as a memory-efficient policy-optimization method using relative outcomes within sampled groups [8]. A prerequisite is useful within-group reward variation. This repository implements a guarded GRPO path, but the evidence in this paper is deliberately centered on SFT and deterministic control because the formal v18 RL gate has not passed.

## 3. Task and System Formulation

### 3.1 Stateful customer-service tasks

A task consists of an initial database state $s_0$, a user conversation $u$, a set of tool schemas $\mathcal{A}$, policy constraints $P$, and a goal predicate $G$. An agent trajectory is

$$
\tau = (u_0, a_0, o_0, \ldots, u_T, a_T, o_T, y),
$$

where $a_t$ is a tool call, $o_t$ is the audited tool result, and $y$ is a structured terminal response. Read tools retrieve identity, order, shipment, payment, or policy evidence. Critical write tools are `cancel_order`, `update_shipping_address`, and `create_refund`; successful calls mutate the task's SQLite copy.

The environment covers 30 scenario families. Examples include cancellation before and after shipment, already-cancelled orders, address changes, full and partial refunds, cross-order item mismatch, duplicate refund requests, quantity overflow, multiple-order disambiguation, identity mismatch, transient tool errors, persistent error escalation, and explicit policy-bypass requests.

### 3.2 Strict success and hard safety

The strict verifier returns

$$
R_{\mathrm{strict}}(\tau) \in \{0,1\}.
$$

A trajectory passes only when the final database state is correct, required tools succeed in the required order, policies are satisfied, the terminal JSON is supported by tool evidence, and required user-facing facts are present and accurate. A correct-looking final sentence cannot compensate for a missing database write.

Hard-safety failures include unsafe writes, cross-order or unauthorized operations, and terminal claims of completed actions without trustworthy write evidence. These failures are reported separately from ordinary strict failures. This separation prevents a high average task score from hiding a small number of consequential violations.

### 3.3 Training-shaped reward

The optional `train_shaped_v18` reward combines five components:

$$
R_{\mathrm{shape}} = 0.35R_{state} + 0.25R_{tools} + 0.20R_{policy}
+ 0.10R_{terminal} + 0.10R_{communication}.
$$

Caps limit the score when state, required writes, tool sequence, terminal consistency, or trajectory completeness fails. Any hard-safety failure sets the reward to $-1$. This reward is intended to provide denser training feedback; `strict_binary_v18` remains the acceptance metric.

## 4. Method

### 4.1 Data construction and isolation

The v18 generator creates 3,600 tasks: 3,000 train tasks and 600 development tasks across all 30 scenario families. Task IDs, database states, and prompt families are separated across splits. A deterministic reference controller executes each task, and only verifier-approved assistant turns are retained. The resulting assistant-only SFT corpus contains 14,800 train rows and 2,960 development rows, for 17,760 rows in total. These reference trajectories are data-generation artifacts, not learned-model evaluation results.

Every rollout operates on a copied SQLite database. Parallel or repeated experiments therefore cannot modify one another's state. The evaluator records the complete message sequence, tool arguments, outputs, policy violations, database effects, structured terminal object, reward components, and source hashes.

### 4.2 QLoRA supervised fine-tuning

The frozen base is `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`. Training uses 4-bit NF4 double quantization, BF16 computation, LoRA rank 16, LoRA alpha 32, dropout 0.05, assistant-only loss, a maximum sequence length of 4,096, micro-batch size 1, gradient accumulation 16, and learning rate $5\times10^{-5}$. A 20-step smoke phase precedes the 925-step full run. The retained adapter has SHA-256 `f41c98a583a6ca66e18f1d6f8743029cb3a363cf96f43d80dbf0d29b74876b6c`.

Three later continued-SFT strategies—focused supplementation, scenario rebalancing, and full replay—improved narrow target slices but damaged neighboring scenarios. None replaced the original adapter.

### 4.3 Evidence-bound controller

Controller r6 wraps critical tools and the final response. It has two roles.

**Pre-execution authorization.** Before a critical write, the controller checks the currently exposed tool surface, whether the target order was successfully inspected, order ownership and state, item-to-order binding, refund-policy evidence, remaining refundable quantity, and explicit user confirmation. An invalid proposal returns an audited `blocked_by_policy_controller` result without executing the write.

**Post-execution grounding.** After tool interaction, the controller resolves a terminal decision from successful writes, inspected order state, user-requested IDs and quantities, and escalation evidence. It can repair a terminal label or required fact only when the visible evidence determines the answer. It does not read `TaskSpec.scenario`, expected tools, expected database state, or hidden reference text.

This division keeps semantic interpretation and ordinary planning in the language model while placing authorization and success claims behind deterministic evidence.

### 4.4 Experimental protocol

The principal comparison uses the same frozen SFT adapter and the frozen 600-task v18 development manifest. Evaluation uses one greedy rollout per task, temperature 0, at most 20 steps, at most 512 generated tokens per step, no correction retry, and an 8,192-token input limit. The controller is evaluated in three stages: a 70-task risk screen, a 150-task broad screen, and the complete 600-task development suite. Infrastructure failures stop the run rather than being silently retried.

The repository distinguishes four kinds of change:

- **rescoring:** recomputing derived verifier fields without regenerating a trajectory;
- **model training:** changing adapter parameters;
- **prompt/protocol change:** changing the input or terminal contract;
- **controller change:** changing external execution or response control.

Results from these interventions are not presented as interchangeable model improvements.

## 5. Results

### 5.1 Overall results

| System or intervention | Scope | Strict pass | Hard-safety failures | Interpretation |
|---|---:|---:|---:|---|
| Original SFT evaluation | 600 dev tasks | 508/600 (84.67%) | 13 | Original verifier output |
| Same SFT rollouts, corrected verifier | Same 600 stored rollouts | 530/600 (88.33%) | 13 | Rescoring only; no new inference |
| SFT + controller r4 | 150-task broad dev screen | 111/150 (74.00%) | 6 | Rejected due to regression |
| SFT + controller r6 | 70-task risk screen | 70/70 (100%) | 0 | Development gate |
| SFT + controller r6 | 150-task broad dev screen | 149/150 (99.33%) | 0 | Development gate |
| **SFT + controller r6** | **600-task frozen dev** | **589/600 (98.17%)** | **0** | Final reported development system |

On the comparable 600-task protocol, the controller-equipped system adds 59 strict passes over the rescored SFT adapter, an absolute increase of 9.84 percentage points. Hard-safety failures decrease from 13 to 0. The r6 run also records `safety_gate_pass_rate=1.0`, zero worker errors, zero maximum-step exhaustions, and mean shaped reward 0.9907.

These numbers measure the full system. Because adapter weights were frozen, they do not show that the model itself learned from 88.33% to 98.17%. The controller was developed using development failures, and the 70- and 150-task screens are not independent test sets.

### 5.2 Residual failures

Twenty-four of 30 scenario families achieved 20/20 strict passes in the final development run. The remaining 11 failures were concentrated as follows:

| Scenario | Strict passes | Main residual issue |
|---|---:|---|
| `duplicate_refund_request` | 18/20 | Terminal decision mismatch |
| `mixed_order_bulk_cancel` | 16/20 | Missing cancellation intent/write on an eligible order |
| `multi_order_disambiguation` | 19/20 | Missing required order-specific value |
| `order_not_found` | 19/20 | Incorrect or omitted order identifier |
| `partial_quantity_refund` | 18/20 | Missing exact amount or remaining quantity |
| `refund_policy_tool_transient_error` | 19/20 | Incomplete fact recovery after retry |

None of these 11 rollouts was classified as a hard-safety failure. The pattern suggests that the controller is strongest at authorization and grounded action claims, while exact communicative completeness and multi-object intent realization remain partially model-dependent.

### 5.3 Negative results and rejected variants

The development history contains two instructive failures. First, controller r4 passed narrow targeted checks but fell to 111/150 with six hard failures on the broader screen. Analysis traced much of the regression to prompt-contract drift and an incomplete fallback policy. Freezing the original training prompt and broadening the screen were necessary before r6 could be accepted.

Second, focused continued-SFT runs solved selected cases but interfered with already learned behavior in cancellation and refund scenarios. A low training loss did not predict broad development reliability. The project therefore retained the original adapter and moved deterministic invariants into the external controller.

### 5.4 Reinforcement-learning status

The repository includes an online GRPO interface with isolated state environments, reward receipts, a 10-step qualification stage, and a 100-step guarded experiment path. The empirical record does not support a GRPO improvement claim:

- on the frozen v17 final test, the SFT control and exploratory GRPO checkpoint-25 both scored 48/160 (30.0%); the paired difference was 0 with a bootstrap 95% interval of $[-0.05, 0.05]$;
- a later v18 train-only calibration produced 80/80 strict passes and a 0% mixed-group rate, so the sampled groups contained no relative advantage signal;
- `formal_rl_gate_passed` remains false, and no v18 policy update is reported.

This is a negative but useful result: implementing a GRPO interface is not equivalent to demonstrating learnable group-relative signal or a better policy.

## 6. Discussion

### 6.1 Why the controller helped

The SFT model was generally capable of reading orders and selecting plausible workflows, but it was brittle when several identifiers, policy constraints, or tool failures interacted. These cases are naturally expressed as finite-state invariants. The controller converted those invariants into executable checks and required proof from successful tool calls. It therefore removed several failure paths that are difficult to eliminate through small targeted fine-tuning without disturbing adjacent behavior.

### 6.2 What remains a model problem

Deterministic authorization does not solve every task. The residual errors involve deciding which eligible operation the user intended, preserving exact values through a multi-turn exchange, and producing complete natural-language explanations. Expanding the controller to dictate every response would inflate the score while reducing the scientific value of evaluating the learned policy. Future revisions should keep the boundary explicit: code owns invariant authorization; the model owns interpretation under ambiguity and natural-language realization.

### 6.3 Evaluation integrity

Three reporting rules were important. First, correcting a verifier changes measurement, not model ability. Second, a targeted slice cannot select a safety component without a broad regression screen. Third, safety and task completion must be reported together. Controller r4 illustrates why: a narrow success could have been misreported as progress even though the broader system became less reliable.

## 7. Limitations and Threats to Validity

1. **Development-set adaptation.** Controller r6 was iterated using development failures and then evaluated on the same frozen development suite. The 98.17% result may overestimate performance on unseen task families.
2. **No independent v18 test result.** The repository intentionally avoids opening a future test set during development. This protects the protocol but prevents a generalization claim in the present paper.
3. **Single deterministic rollout.** The principal result uses one temperature-zero rollout per task. It does not measure stochastic consistency or pass$^k$.
4. **Incomplete baselines and ablations.** A full paper should compare base Qwen, prompt-only control, SFT alone, pre-execution guard only, terminal grounding only, and the complete controller under the same manifest.
5. **One model and one domain.** Results may not transfer to other model sizes, languages, business databases, or tool APIs.
6. **No latency or cost analysis.** The current evidence does not quantify controller overhead, extra reads, GPU time, or operational cost.
7. **Synthetic environment.** SQLite tasks make exact verification possible, but production systems have concurrency, delayed effects, partial outages, and policy changes absent from this environment.
8. **No production safety guarantee.** Zero hard failures in 600 development tasks is evidence about this protocol, not proof of universal safety.

## 8. Reproducibility

The repository pins the base revision, records the retained adapter hash, freezes manifests and prompt versions, stores protocol and source hashes, fails closed on worker errors, and keeps large artifacts outside Git with SHA-256 receipts. Key implementation and evidence files are:

- `configs/sft-v18-48gb.json` — model, data, and QLoRA configuration;
- `configs/v18-reward.json` — reward weights, caps, and calibration gates;
- `src/ecommerce_customer_service/tools_v18.py` — evidence-bound business state machine;
- `src/ecommerce_customer_service/verifier_v18.py` — strict and shaped rewards;
- `src/ecommerce_customer_service/policy_controller_v18.py` — controller r6;
- `scripts/evaluate_sft_v18.py` — frozen evaluation runner;
- `docs/v18-sft-full-dev-analysis-2026-10-04.md` — SFT evaluation and rescoring audit;
- `docs/v18-project-issues-solutions-and-handoff.md` — full experimental record and artifact hashes.

Model weights, generated SQLite files, and raw rollouts are intentionally excluded from GitHub. Reproduction therefore requires the recorded base revision, adapter archive, task package, and their published hashes.

## 9. Conclusion

This study presents a stateful and verifiable e-commerce customer-service agent in which high-consequence actions are bound to explicit tool evidence. A frozen QLoRA SFT policy reached 530/600 strict passes after verifier correction, while the complete SFT-plus-controller system reached 589/600 with no hard-safety failures on the frozen v18 development suite. The result demonstrates the practical value of separating learned interpretation from deterministic authorization and evidence-grounded completion. It does not yet establish held-out generalization or reinforcement-learning gains. The next decisive experiments are a locked independent test, matched controller ablations, repeated stochastic trials, and cross-domain evaluation.

## References

[1] S. Yao et al. “ReAct: Synergizing Reasoning and Acting in Language Models.” ICLR 2023. <https://arxiv.org/abs/2210.03629>

[2] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan. “$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.” 2024. <https://arxiv.org/abs/2406.12045>

[3] E. J. Hu et al. “LoRA: Low-Rank Adaptation of Large Language Models.” ICLR 2022. <https://arxiv.org/abs/2106.09685>

[4] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. “QLoRA: Efficient Finetuning of Quantized LLMs.” NeurIPS 2023. <https://arxiv.org/abs/2305.14314>

[5] A. Yang et al. “Qwen2.5 Technical Report.” 2024. <https://arxiv.org/abs/2412.15115>

[6] Y. Ruan et al. “Identifying the Risks of LM Agents with an LM-Emulated Sandbox.” ICLR 2024. <https://arxiv.org/abs/2309.15817>

[7] E. Debenedetti et al. “AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.” NeurIPS 2024. <https://arxiv.org/abs/2406.13352>

[8] Z. Shao et al. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” 2024. <https://arxiv.org/abs/2402.03300>
