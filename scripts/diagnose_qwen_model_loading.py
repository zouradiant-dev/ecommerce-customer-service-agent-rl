"""Audit the entire base model against disk before and after one GPU forward.

Run each loading mode in a separate process on the rented server. No adapters,
optimizer, training, downloads, or checkpoint writes are involved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("direct", "cpu"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path, default=Path(
        "/home/featurize/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0")
    sys.path.insert(0, str(args.project.resolve() / "src"))

    import torch
    import torch.nn.functional as F
    import transformers
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ecommerce_customer_service.training import render_with_assistant_mask

    def emit(**values):
        print(json.dumps({"mode": args.mode, **values}, ensure_ascii=False,
                         allow_nan=False), flush=True)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1807)
    free, total = torch.cuda.mem_get_info()
    emit(stage="environment", torch=torch.__version__, cuda=torch.version.cuda,
         transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
         free_gib=free / 1024**3, total_gib=total / 1024**3)
    target = "model.layers.26.mlp.up_proj.weight"
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    shards = defaultdict(list)
    for name, shard in index["weight_map"].items():
        shards[shard].append(name)

    def audit(model, stage):
        state = model.state_dict()
        bad, gpu_disagreements, elements = [], [], 0
        for shard, names in sorted(shards.items()):
            emit(stage=stage, checking_shard=shard)
            with safe_open(str(args.snapshot / shard), framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    tensor = state[name].detach()
                    gpu_finite = bool(torch.isfinite(tensor).all().item()) if tensor.is_cuda else None
                    actual = tensor.cpu().contiguous()
                    expected = handle.get_tensor(name).to(actual.dtype).contiguous()
                    if actual.shape != expected.shape:
                        raise RuntimeError(f"shape mismatch: {name}")
                    if actual.dtype != torch.float32:
                        raise RuntimeError(f"unexpected dtype: {name}: {actual.dtype}")
                    cpu_finite = bool(torch.isfinite(actual).all().item())
                    expected_finite = bool(torch.isfinite(expected).all().item())
                    if not expected_finite:
                        raise RuntimeError(f"CPU disk reference non-finite: {name}")
                    a, b = actual.view(torch.int32).flatten(), expected.view(torch.int32).flatten()
                    mismatch_count, first = 0, []
                    for offset in range(0, a.numel(), 1048576):
                        different = a[offset:offset + 1048576].ne(b[offset:offset + 1048576])
                        count = int(different.sum().item())
                        mismatch_count += count
                        if count and len(first) < 5:
                            first.extend((different.nonzero().flatten()[:5 - len(first)] + offset).tolist())
                    elements += actual.numel()
                    result = dict(name=name, mismatch_count=mismatch_count,
                                  cpu_roundtrip_finite=cpu_finite, gpu_finite=gpu_finite,
                                  gpu_pointer=hex(tensor.data_ptr()) if tensor.is_cuda else None)
                    if mismatch_count:
                        result.update(first_indices=first,
                                      expected_values=expected.flatten()[first].tolist(),
                                      actual_values=[str(value) for value in actual.flatten()[first].tolist()])
                    if mismatch_count or not cpu_finite:
                        bad.append(result)
                        emit(stage=stage, bad_weight=result)
                    if gpu_finite is not None and gpu_finite != cpu_finite:
                        gpu_disagreements.append(name)
                        emit(stage=stage, gpu_finite_check_disagrees_with_cpu=name)
                    if name == target:
                        emit(stage=stage, target_weight=result)
                    del actual, expected, a, b, different
        result = dict(stage=stage, checked_elements=elements,
                      all_weights_bitwise_equal=not bad, bad_weight_count=len(bad),
                      bad_weights=[item["name"] for item in bad],
                      gpu_finite_check_disagreements=gpu_disagreements)
        emit(**result)
        return result

    emit(stage="loading", path="GPU device_map" if args.mode == "direct" else "ordinary CPU load then GPU upload")
    kwargs = dict(torch_dtype=torch.float32, attn_implementation="eager",
                  local_files_only=True, use_safetensors=True, output_loading_info=True)
    if args.mode == "direct":
        kwargs.update(low_cpu_mem_usage=True, device_map={"": 0})
    else:
        kwargs.update(low_cpu_mem_usage=False)
    model, info = AutoModelForCausalLM.from_pretrained(str(args.snapshot), **kwargs)
    emit(stage="loaded", missing_keys=info.get("missing_keys", []),
         unexpected_keys=info.get("unexpected_keys", []), error_msgs=info.get("error_msgs", []))
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("error_msgs"):
        raise RuntimeError("loading info is not clean")
    if args.mode == "cpu":
        if not audit(model, "cpu_before_upload")["all_weights_bitwise_equal"]:
            emit(stage="stopped", reason="CPU model already differs from disk")
            raise SystemExit(2)
        model.to("cuda:0")
    model.eval()
    model.config.use_cache = False
    emit(stage="attention", actual=model.config._attn_implementation,
         attention_class=type(model.model.layers[0].self_attn).__name__)
    before = audit(model, "gpu_before_forward")
    if not before["all_weights_bitwise_equal"]:
        emit(stage="stopped", reason="GPU weights differ from disk before any forward")
        raise SystemExit(2)

    row_id = "sft_v18_v18_dev_address_change_after_shipment_0008_turn0"
    row = None
    with (args.project / "data/v18-sft-reference-r1/dev.jsonl").open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item.get("id") == row_id:
                row = item
                break
    if row is None:
        raise RuntimeError(f"diagnostic row not found: {row_id}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
    ids, mask = render_with_assistant_mask(tokenizer, row["messages"],
        tools=list(row.get("tools") or []),
        loss_mask=[bool(value) for value in row.get("loss_mask") or []] or None)
    if len(ids) > 4096:
        raise RuntimeError("diagnostic row exceeds 4096 tokens")
    labels = torch.tensor([[token if active else -100 for token, active in zip(ids, mask)]])
    active = labels[:, 1:].ne(-100)
    if not active.any():
        raise RuntimeError("diagnostic row has no supervised tokens")
    emit(stage="forward", row_id=row_id, sequence_tokens=len(ids), supervised_tokens=int(active.sum()))
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([ids], device="cuda:0"),
                       attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda:0"),
                       output_hidden_states=True, use_cache=False)
    torch.cuda.synchronize()
    first_bad, maxima = None, []
    for i, hidden in enumerate(output.hidden_states):
        cpu = hidden.float().cpu()
        finite = bool(torch.isfinite(cpu).all().item())
        if not finite and first_bad is None:
            first_bad = i
        maxima.append(float(cpu.abs().max()) if finite else None)
        del cpu
    selected = output.logits[:, :-1][active.to("cuda:0")].float().cpu()
    logits_finite = bool(torch.isfinite(output.logits.float().cpu()).all().item())
    selected_finite = bool(torch.isfinite(selected).all().item())
    loss = float(F.cross_entropy(selected, labels[:, 1:][active])) if selected_finite else None
    del output, selected
    emit(stage="forward_result", first_bad_hidden_state=first_bad, hidden_state_max_abs=maxima,
         logits_finite_on_cpu=logits_finite, target_logits_finite_on_cpu=selected_finite,
         selected_ce=loss, peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3)
    audit(model, "gpu_after_forward")
    emit(stage="finished")


if __name__ == "__main__":
    main()
