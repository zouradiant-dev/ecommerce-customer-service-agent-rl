from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ecommerce_customer_service.serving.artifacts import MODEL_ID, verify_adapter, verify_local_base
from ecommerce_customer_service.serving.inference import BoundedGenerator, InferenceServer


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the verified v18 SFT LoRA; local files only, no training")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "models" / "v18-sft-verified")
    parser.add_argument("--base-model-dir", type=Path, required=True, help="Pinned HF snapshot containing Qwen base weights")
    parser.add_argument("--precision", choices=["bf16", "nf4"], default="bf16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-budget", type=int, default=8192)
    parser.add_argument("--check-only", action="store_true", help="Verify files without importing torch or loading GPU")
    args = parser.parse_args()
    if not 512 <= args.context_budget <= 32768:
        parser.error("--context-budget must be 512..32768")
    api_key = os.environ.get("MODEL_API_KEY", "")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and len(api_key) < 24:
        parser.error("Remote binding requires MODEL_API_KEY of at least 24 characters; prefer SSH forwarding")
    verify_adapter(args.adapter_dir)
    verify_local_base(args.base_model_dir)
    if args.check_only:
        print("Pinned adapter/base files verified. GPU inference has not been tested.")
        return 0
    # No network fetch: both the base and adapter are explicit, verified local directories.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    from peft import PeftModel
    from ecommerce_customer_service.local_hf import TransformersCompleter
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU fallback is disabled")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options = dict(local_files_only=True, trust_remote_code=False, torch_dtype=dtype, device_map={"": 0})
    if args.precision == "nf4":
        options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(args.base_model_dir, **options)
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False, local_files_only=True)
    model.eval()
    generation = GenerationConfig(do_sample=False, max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    completer = TransformersCompleter(model, tokenizer, torch, generation,
                                     device="cuda:0", max_input_tokens=args.context_budget)
    server = InferenceServer((args.host, args.port), BoundedGenerator(completer),
                             api_key=api_key, precision=args.precision)
    print(f"Loaded {MODEL_ID}; inference URL http://{args.host}:{server.server_port}/v1", flush=True)
    print("Online task quality is unverified. No business operations run in this process.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
