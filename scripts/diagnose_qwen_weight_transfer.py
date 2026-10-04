"""Compare CPU/GPU copies and dtype conversion of the observed bad base weight."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    revision = "a09a35458c702b33eeacc393d103063234e8bc28"
    snapshot = (
        Path("/home/featurize/hf-cache/hub")
        / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / revision
    )
    target = "model.layers.26.mlp.up_proj.weight"
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    with safe_open(str(snapshot / index["weight_map"][target]), framework="pt", device="cpu") as handle:
        original = handle.get_tensor(target).clone()
    reference = original.float()
    print(json.dumps({
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "tensor": target,
        "cpu_bf16_finite": bool(torch.isfinite(original).all()),
        "cpu_fp32_finite": bool(torch.isfinite(reference).all()),
    }), flush=True)
    if not bool(torch.isfinite(reference).all()):
        raise RuntimeError("CPU reference is non-finite; no GPU test performed")

    def compare(label, actual, expected, iteration):
        actual = actual.cpu().contiguous()
        expected = expected.contiguous()
        integer_dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
        mismatches = actual.view(integer_dtype).ne(expected.view(integer_dtype))
        mismatch_count = int(mismatches.sum())
        result = {
            "round": iteration, "test": label,
            "bitwise_equal": mismatch_count == 0,
            "mismatch_count": mismatch_count,
            "nan_count_on_cpu": int(torch.isnan(actual).sum()),
            "inf_count_on_cpu": int(torch.isinf(actual).sum()),
        }
        if mismatch_count:
            indices = mismatches.flatten().nonzero().flatten()[:5]
            result["first_mismatch_indices"] = indices.tolist()
            result["expected_values"] = expected.flatten()[indices].float().tolist()
            result["actual_values"] = actual.flatten()[indices].float().tolist()
        print(json.dumps(result), flush=True)

    x = torch.linspace(-0.1, 0.1, reference.shape[1]).repeat(4, 1)
    y_reference = F.linear(x, reference)
    with torch.no_grad():
        for iteration in range(1, 4):
            gpu_fp32 = reference.to("cuda:0")
            torch.cuda.synchronize()
            compare("CPU FP32 -> GPU -> CPU", gpu_fp32, reference, iteration)
            y_actual = F.linear(x.to("cuda:0"), gpu_fp32).cpu()
            print(json.dumps({
                "round": iteration, "test": "FP32 linear",
                "result_finite_on_cpu": bool(torch.isfinite(y_actual).all()),
                "matches_cpu_with_tolerance": bool(torch.allclose(
                    y_actual, y_reference, rtol=1e-4, atol=1e-4,
                )),
                "max_abs_error": float((y_actual - y_reference).abs().max()),
            }), flush=True)
            del gpu_fp32, y_actual

            gpu_bf16 = original.to("cuda:0")
            torch.cuda.synchronize()
            compare("CPU BF16 -> GPU -> CPU", gpu_bf16, original, iteration)
            converted = gpu_bf16.float()
            torch.cuda.synchronize()
            compare("BF16 -> FP32 conversion on GPU", converted, reference, iteration)
            del gpu_bf16, converted
            torch.cuda.empty_cache()
    print("GPU weight transfer diagnostic finished", flush=True)


if __name__ == "__main__":
    main()
