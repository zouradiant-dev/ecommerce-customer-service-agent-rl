# Explicit opt-in v13 reward correction

This is a small deterministic correction, not a general semantic judge. Generation, GPU training and training entry points are unchanged. `VERIFIER_VERSION` and existing default evaluation/training paths remain `deterministic_semantic_v12`; use `V13_VERIFIER_VERSION` explicitly for new scoring.

## Contract

- v12 intent and grounded-fact implementations remain available and unchanged. The version router also retains v11 behavior.
- v13 accepts a literal `refund_id=null` only when a successful owned target-order read establishes zero captured payment and an empty refund list, and an actual successful cancellation explicitly returns `refund_id=None`. Missing fields, failed/foreign tools, omitted cancellation, known non-null refunds or captured/refunded positive payments do not establish absence.
- The exemption removes only `fact_mismatch:refund_id:null`. Incorrect refund, return and ticket identifiers remain failures. Required tools, policy, state and retained runtime failures cannot be waived.
- Explicit labelled order totals use the owned `get_order.order.total_paid_cents` field. Explicit actual-capture labels use owned captured/refunded payment amounts; authorization is not capture. Unknown facts, unlabelled or conditional prose remain outside this scoped rule, not independently verified.
- No task-specific answers, IDs, amounts or aliases are added to training. Regression fixtures are synthetic controls, not model samples or readiness evidence.

## Local CPU result on the saved 20 train trajectories

v12 replay reproduces all original tool outputs, state, policy, required tools, verifier and rewards. v13 changes **15/20 to 17/20**, with only two binary changes: correctly reported absent refund IDs in cancellation trials 0 and 2. Cancellation trial 1 remains zero and gains `fact_mismatch:order_total:0元`; trial 3 still lacks execution. The incorrect return ID in policy-retry trial 1 stays zero. All four policy-retry rollouts actually retry the failed query.

The three positive write families are covered; 2/5 groups are mixed (40%). Engineering sampling checks pass. This is not model improvement, held-out evidence, a calibrated reward claim or a formal RL gate. Reviews are not silently promoted to `confirmed`; the existing smoke validator still rejects unsigned review evidence. Native dynamic-tool sampling does not validate TRL static-tool training.

## Deploy the code-only patch on the server

The installer checks exact known v12 code hashes before writing, checks frozen business dependencies, refuses unknown existing changes and active project jobs, saves exact replaced originals under `outputs/code-backups`, and is idempotent. The backup folder retains the old `reward-fix-v12-` prefix because the saved originals are v12. No original reports, databases, weights or training scripts are payload targets.

From Windows PowerShell, upload the archive and checksum. Replace only the server port if the instance endpoint has changed:

```powershell
scp -P 55801 "D:\桌面\llm-learning\reward-fix-v13-20261002-r1\reward-fix-v13-source-20261002-r1.tar.gz" "D:\桌面\llm-learning\reward-fix-v13-20261002-r1\reward-fix-v13-source-20261002-r1.tar.gz.sha256" featurize@workspace.featurize.cn:/home/featurize/work/
```

Then, in the connected **server Linux terminal**, deploy and run CPU-only tests:

```bash
(
  cd /home/featurize/work || exit 1
  sha256sum -c reward-fix-v13-source-20261002-r1.tar.gz.sha256 || exit 1
  task_v13_dir=$(mktemp -d /home/featurize/work/reward-v13-deploy.XXXXXX) || exit 1
  tar -xzf reward-fix-v13-source-20261002-r1.tar.gz -C "$task_v13_dir" || exit 1
  /home/featurize/grpo-env-v12/bin/python "$task_v13_dir/reward-fix-v13-source-20261002-r1/install_reward_fix_v13.py" \
    --repo /home/featurize/work/ecommerce-customer-service-agent-rl || exit 1
  cd /home/featurize/work/ecommerce-customer-service-agent-rl || exit 1
  /home/featurize/grpo-env-v12/bin/python -m unittest discover -s tests -p 'test_reward_v1*.py' -v || exit 1
  /home/featurize/grpo-env-v12/bin/python -m unittest discover -s tests -p 'test_rescore_reward_v13.py' -v || exit 1
)
```

The checksum is written as LF bytes for Linux; do not reconstruct it with a Windows text-output command. Stop on any unexpected hash or existing-file warning; do not force overwrite.

Deployment only makes explicit v13 scoring available. It does not change the online training default, approve reviews, calibrate reward or start GRPO. Before a bounded v13 GRPO smoke, explicitly align the online reward version with the reviewed scoring snapshot and record its runtime source hashes. Do not run the old launcher unchanged or silently relabel v12 generation evidence.

## CPU replay invocation

`scripts/rescore_reward_v13.py --pilot SOURCE_PILOT --manifest TASK_MANIFEST --output NEW_OUTPUT` accepts a complete frozen-v12 inline train pilot, replays all tools in temporary database copies and first verifies every old reward. It refuses stale reports, unknown generation snapshots, modified dependencies/databases, existing or nested outputs, and invalid/incomplete execution groups. Generation and scoring source hashes are stored separately. It creates a new report without model loading or generation. Pending review statuses and formal-gate flags remain false.
