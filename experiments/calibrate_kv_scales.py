"""
EXAONE-4.0-1.2B KV Cache FP8 캘리브레이션 + 스케일 주입

1. 캘리브레이션 데이터(MANTA-1M)로 forward pass
2. 레이어별 K/V 텐서의 absmax 수집
3. FP8 E4M3 스케일 계산 (scale = absmax / fp8_e4m3_max)
4. model.safetensors에 k_scale/v_scale 주입
5. config.json에 modelopt quantization_config 설정

결과: Weight BF16 + 캘리브레이션된 FP8 KV cache

사용법:
  python experiments/calibrate_kv_scales.py
  python experiments/calibrate_kv_scales.py --num-samples 64 --max-length 1024
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import torch
import safetensors.torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def collect_kv_scales(
    model_path: str,
    num_samples: int = 32,
    max_length: int = 512,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """캘리브레이션 데이터로 레이어별 K/V absmax를 수집하고 스케일 계산."""
    random.seed(seed)
    torch.manual_seed(seed)

    print("Loading model for calibration...")
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

    # 캘리브레이션 데이터 로드
    print(f"Loading MANTA-1M ({num_samples} samples)...")
    dataset = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
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

    # 레이어별 K/V absmax 추적
    k_maxes = [0.0] * num_layers
    v_maxes = [0.0] * num_layers

    print(f"Running calibration on {len(texts)} samples...")
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
                # k, v shape: (batch, num_heads, seq_len, head_dim)
                k_abs = k.float().abs().max().item()
                v_abs = v.float().abs().max().item()
                k_maxes[layer_idx] = max(k_maxes[layer_idx], k_abs)
                v_maxes[layer_idx] = max(v_maxes[layer_idx], v_abs)

            if (i + 1) % 8 == 0:
                print(f"  Processed {i + 1}/{len(texts)}")

    # 스케일 계산: scale = absmax / FP8_E4M3_MAX
    k_scales = [k_max / FP8_E4M3_MAX for k_max in k_maxes]
    v_scales = [v_max / FP8_E4M3_MAX for v_max in v_maxes]

    print(f"\nCalibration results:")
    print(f"{'Layer':>6} {'K absmax':>10} {'K scale':>10} {'V absmax':>10} {'V scale':>10}")
    print("-" * 50)
    for i in range(num_layers):
        print(f"  {i:>3}    {k_maxes[i]:>8.4f}   {k_scales[i]:>8.6f}   {v_maxes[i]:>8.4f}   {v_scales[i]:>8.6f}")

    return k_scales, v_scales


def inject_scales(
    model_path: str,
    output_path: str,
    k_scales: list[float],
    v_scales: list[float],
):
    """모델 복사 + safetensors에 k/v scale 주입 + modelopt config 설정."""
    model_path = Path(model_path)
    output_path = Path(output_path)

    if output_path.exists():
        print(f"기존 출력 디렉토리 삭제: {output_path}")
        shutil.rmtree(output_path)

    print(f"모델 파일 복사: {model_path} → {output_path}")
    shutil.copytree(model_path, output_path)

    # safetensors에 k_scale/v_scale 추가
    safetensors_path = output_path / "model.safetensors"
    print(f"Loading safetensors for scale injection...")
    tensors = safetensors.torch.load_file(str(safetensors_path))

    num_layers = len(k_scales)
    for i in range(num_layers):
        tensors[f"model.layers.{i}.self_attn.k_scale"] = torch.tensor(
            k_scales[i], dtype=torch.float32
        )
        tensors[f"model.layers.{i}.self_attn.v_scale"] = torch.tensor(
            v_scales[i], dtype=torch.float32
        )

    print(f"Saving safetensors with {num_layers * 2} scale tensors...")
    safetensors.torch.save_file(tensors, str(safetensors_path))

    # config.json에 modelopt quantization_config 설정
    config_path = output_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "FP8",
            "kv_cache_quant_algo": "FP8",
            "exclude_modules": ["*"],
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # hf_quant_config.json 생성
    hf_quant_path = output_path / "hf_quant_config.json"
    with open(hf_quant_path, "w") as f:
        json.dump(config["quantization_config"], f, indent=2, ensure_ascii=False)

    print(f"\n설정 완료:")
    print(f"  config.json: modelopt quantization_config 추가")
    print(f"  hf_quant_config.json: 생성")
    print(f"  model.safetensors: k_scale/v_scale {num_layers}개 레이어 주입")

    # 파일 목록
    print(f"\n생성된 파일:")
    total_size = 0
    for f in sorted(output_path.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  Total: {total_size:.2f} MB")

    print(f"\n벤치마크: python benchmark.py gsm8k -m {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B KV Cache FP8 Calibration + Scale Injection"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="캘리브레이션용 모델 경로 (HF ID 또는 로컬)",
    )
    parser.add_argument(
        "--source-path", type=str,
        default="models/EXAONE-4.0-1.2B",
        help="복사할 로컬 모델 디렉토리",
    )
    parser.add_argument(
        "--output-path", type=str,
        default="models/EXAONE-4.0-1.2B-kv-fp8-calibrated",
        help="출력 모델 경로",
    )
    parser.add_argument(
        "--num-samples", type=int, default=32,
        help="캘리브레이션 샘플 수",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="최대 시퀀스 길이",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    k_scales, v_scales = collect_kv_scales(
        model_path=args.model_path,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    inject_scales(
        model_path=args.source_path,
        output_path=args.output_path,
        k_scales=k_scales,
        v_scales=v_scales,
    )


if __name__ == "__main__":
    main()
