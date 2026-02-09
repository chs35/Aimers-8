"""
EXAONE-4.0-1.2B FP8 W8A8 + KV Cache FP8 통합 스크립트

1. llmcompressor로 FP8 W8A8 Static 양자화 (MANTA-1M 캘리브레이션)
2. 원본 BF16 모델에서 KV scale 캘리브레이션 (MANTA-1M)
3. 양자화된 모델에 KV scale 주입
4. config.json을 modelopt 형식으로 변경 (vLLM KV FP8 auto-resolve 트리거)

결과: FP8 Weight + FP8 KV Cache → 최대 속도 개선

사용법:
  python experiments/quantize_fp8_full.py
  python experiments/quantize_fp8_full.py --num-samples 1024 --max-length 2048
  python experiments/quantize_fp8_full.py --skip-weight-quant  # KV scale만 재캘리브레이션
"""

import argparse
import json
import os
import random
import shutil
import sys
import logging
from datetime import datetime
from pathlib import Path

import torch
import safetensors.torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
SCRIPT_DIR = Path(__file__).parent
LOG_DIR = SCRIPT_DIR / "logs"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_quantize_fp8_full.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    print(f"Log: {log_path}")
    return log_path


def prepare_calibration_dataset(tokenizer, num_samples=1024, max_length=2048, seed=42):
    """MANTA-1M에서 캘리브레이션 데이터 준비."""
    random.seed(seed)

    print(f"Loading MANTA-1M ({num_samples} samples, max_length={max_length})...")
    dataset = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")

    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)

    texts = []
    for sample in dataset:
        conversations = sample.get("conversations", [])
        if conversations:
            text = "\n".join([turn.get("content", "") for turn in conversations])
        else:
            text = sample.get("text", sample.get("content", ""))
        if text:
            texts.append(text)

    ds = Dataset.from_dict({"text": texts})

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    print(f"Calibration samples: {len(ds)}")
    return ds, texts


def step1_weight_quantization(model_path, output_path, num_samples, max_length, seed):
    """FP8 W8A8 Static 양자화 (llmcompressor)."""
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError:
        raise ImportError("llmcompressor 필요: pip install llmcompressor")

    print("\n" + "=" * 60)
    print("[Step 1] FP8 W8A8 Weight Quantization")
    print("=" * 60)

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Preparing calibration dataset...")
    calibration_ds, _ = prepare_calibration_dataset(
        tokenizer, num_samples=num_samples, max_length=max_length, seed=seed,
    )

    print("\nQuantization recipe: FP8 Static W8A8")
    print("  targets: Linear")
    print("  ignore: ['lm_head']")

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8",
        ignore=["lm_head"],
    )

    print("Applying quantization...")
    oneshot(
        model=model,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    print(f"Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    # torch_dtype을 float16으로 설정
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["torch_dtype"] = "float16"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("Weight quantization complete.")
    return tokenizer


def step2_kv_calibration(model_path, num_samples, max_length, seed):
    """원본 BF16 모델에서 KV scale 캘리브레이션."""
    print("\n" + "=" * 60)
    print("[Step 2] KV Cache Scale Calibration")
    print("=" * 60)

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading BF16 model for KV calibration: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    print(f"Layers: {num_layers}")

    # MANTA-1M 로드
    _, texts = prepare_calibration_dataset(
        tokenizer, num_samples=num_samples, max_length=max_length, seed=seed,
    )

    k_maxes = [0.0] * num_layers
    v_maxes = [0.0] * num_layers

    print(f"Running KV calibration on {len(texts)} samples...")
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(model.device)

            outputs = model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values

            for layer_idx, (k, v) in enumerate(past_kv):
                k_abs = k.float().abs().max().item()
                v_abs = v.float().abs().max().item()
                k_maxes[layer_idx] = max(k_maxes[layer_idx], k_abs)
                v_maxes[layer_idx] = max(v_maxes[layer_idx], v_abs)

            if (i + 1) % 64 == 0:
                print(f"  Processed {i + 1}/{len(texts)}")

    k_scales = [k_max / FP8_E4M3_MAX for k_max in k_maxes]
    v_scales = [v_max / FP8_E4M3_MAX for v_max in v_maxes]

    print(f"\nKV Scale results:")
    print(f"{'Layer':>6} {'K absmax':>10} {'K scale':>10} {'V absmax':>10} {'V scale':>10}")
    print("-" * 50)
    for i in range(num_layers):
        print(f"  {i:>3}    {k_maxes[i]:>8.4f}   {k_scales[i]:>8.6f}   {v_maxes[i]:>8.4f}   {v_scales[i]:>8.6f}")

    # 메모리 해제
    del model
    torch.cuda.empty_cache()

    return k_scales, v_scales


def step3_inject_and_configure(output_path, k_scales, v_scales):
    """KV scale 주입 + modelopt config 설정."""
    print("\n" + "=" * 60)
    print("[Step 3] Inject KV Scales + ModelOpt Config")
    print("=" * 60)

    output_path = Path(output_path)
    num_layers = len(k_scales)

    # safetensors에 k_scale/v_scale 추가
    safetensors_path = output_path / "model.safetensors"
    if not safetensors_path.exists():
        # sharded model의 경우 첫 번째 shard를 찾음
        shards = sorted(output_path.glob("model-*.safetensors"))
        if shards:
            safetensors_path = shards[0]
        else:
            raise FileNotFoundError(f"No safetensors file found in {output_path}")

    print(f"Loading {safetensors_path.name} for scale injection...")
    tensors = safetensors.torch.load_file(str(safetensors_path))

    for i in range(num_layers):
        tensors[f"model.layers.{i}.self_attn.k_scale"] = torch.tensor(
            k_scales[i], dtype=torch.float32
        )
        tensors[f"model.layers.{i}.self_attn.v_scale"] = torch.tensor(
            v_scales[i], dtype=torch.float32
        )

    print(f"Saving safetensors with {num_layers * 2} scale tensors...")
    safetensors.torch.save_file(tensors, str(safetensors_path))
    del tensors

    # config.json을 modelopt 형식으로 변경
    config_path = output_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # hf_quant_config.json
    hf_quant_path = output_path / "hf_quant_config.json"
    with open(hf_quant_path, "w") as f:
        json.dump(config["quantization_config"], f, indent=2, ensure_ascii=False)

    print("Config updated: modelopt + KV FP8")

    # 파일 목록
    print(f"\nOutput files:")
    total_size = 0
    for f in sorted(output_path.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  Total: {total_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B FP8 W8A8 + KV Cache FP8 (Full Pipeline)"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="원본 모델 경로 (HF ID 또는 로컬)",
    )
    parser.add_argument(
        "--source-path", type=str,
        default="models/EXAONE-4.0-1.2B",
        help="KV 캘리브레이션용 로컬 모델 경로",
    )
    parser.add_argument(
        "--output-path", type=str,
        default="models/EXAONE-4.0-1.2B-fp8-full",
        help="출력 모델 경로",
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-weight-quant", action="store_true",
        help="Weight 양자화 스킵 (KV scale만 재캘리브레이션)",
    )
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("EXAONE-4.0-1.2B FP8 Full Pipeline")
    print("  W8A8 FP8 Weight + FP8 KV Cache + Calibrated Scales")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Source (KV calib): {args.source_path}")
    print(f"Output: {args.output_path}")
    print(f"Samples: {args.num_samples}, MaxLen: {args.max_length}")

    # Step 1: Weight Quantization
    if not args.skip_weight_quant:
        step1_weight_quantization(
            model_path=args.model_path,
            output_path=args.output_path,
            num_samples=args.num_samples,
            max_length=args.max_length,
            seed=args.seed,
        )
    else:
        print("\n[Step 1] Skipped (--skip-weight-quant)")

    # Step 2: KV Scale Calibration (from BF16 model)
    k_scales, v_scales = step2_kv_calibration(
        model_path=args.source_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Step 3: Inject scales + modelopt config
    step3_inject_and_configure(
        output_path=args.output_path,
        k_scales=k_scales,
        v_scales=v_scales,
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"벤치마크: python benchmark.py gsm8k -m {args.output_path}")


if __name__ == "__main__":
    main()
