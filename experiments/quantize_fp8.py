"""
FP8 (E4M3) 양자화 스크립트 - L4 GPU용
llmcompressor + compressed-tensors 사용 (대회 환경 호환)
Static Scaling으로 Activation 스케일링 팩터 고정
Manta 1M 데이터셋으로 캘리브레이션
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from safetensors import safe_open
from safetensors.torch import save_file


# FP8 양자화 레시피 (Static Scaling)
FP8_RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""

# FP8 + KV Cache FP8 레시피
FP8_KV_CACHE_RECIPE = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""


def prepare_calibration_dataset(tokenizer, num_samples=512, max_length=2048):
    """Manta 1M 데이터셋에서 캘리브레이션 데이터 준비"""
    # Manta 1M 데이터셋 로드 (EXAONE-3.5-32B-Instruct로 생성된 합성 데이터)
    # https://huggingface.co/datasets/LGAI-EXAONE/MANTA-1M
    ds = load_dataset(
        "LGAI-EXAONE/MANTA-1M",
        split=f"train[:{num_samples}]"
    )
    ds = ds.shuffle(seed=42)

    def process_and_tokenize(example):
        # Manta 데이터셋 구조에 맞게 텍스트 추출
        text = example.get("text") or example.get("content") or str(example)
        return tokenizer(
            text,
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False
        )

    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)
    print(f"Prepared {len(ds)} calibration samples from MANTA-1M dataset")
    return ds


def fix_tied_weights_after_save(output_path: str):
    """llmcompressor의 save_compressed가 tie_word_embeddings를 무시하고
    lm_head.weight를 별도 저장하는 문제를 후처리로 수정.

    embed_tokens.weight와 lm_head.weight가 동일하면 lm_head.weight를 제거하여
    모델 크기를 ~0.4GB 절약. (98816×2048×2bytes = 386MB)
    """
    output_dir = Path(output_path)

    config_path = output_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    if not config.get("tie_word_embeddings", False):
        print("  tie_word_embeddings=False, skipping tied weight fix")
        return

    # 단일 safetensors 파일 처리
    single_file = output_dir / "model.safetensors"
    index_file = output_dir / "model.safetensors.index.json"

    if single_file.exists():
        _fix_single_safetensors(single_file)
    elif index_file.exists():
        _fix_sharded_safetensors(output_dir, index_file)
    else:
        print("  WARNING: No safetensors files found, skipping tied weight fix")


def _fix_single_safetensors(filepath: Path):
    """단일 safetensors 파일에서 중복 lm_head.weight 제거."""
    with safe_open(str(filepath), framework="pt") as f:
        keys = list(f.keys())
        metadata = f.metadata()

    if "lm_head.weight" not in keys:
        print("  lm_head.weight not found in safetensors, already tied")
        return

    # embed_tokens 키 찾기
    embed_key = None
    for k in keys:
        if "embed_tokens.weight" in k:
            embed_key = k
            break

    if embed_key is None:
        print("  WARNING: embed_tokens.weight not found, cannot verify tying")
        return

    # 텐서 로드 (lm_head 제외)
    tensors = {}
    original_size = filepath.stat().st_size
    with safe_open(str(filepath), framework="pt") as f:
        for key in keys:
            if key != "lm_head.weight":
                tensors[key] = f.get_tensor(key)

    # 메타데이터 보존하면서 재저장
    save_file(tensors, str(filepath), metadata=metadata or {})
    new_size = filepath.stat().st_size
    saved = (original_size - new_size) / 1e6
    print(f"  Removed duplicate lm_head.weight ({saved:.1f} MB saved)")
    print(f"  Safetensors: {original_size / 1e9:.3f} GB -> {new_size / 1e9:.3f} GB")


def _fix_sharded_safetensors(output_dir: Path, index_path: Path):
    """샤딩된 safetensors에서 중복 lm_head.weight 제거."""
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    if "lm_head.weight" not in weight_map:
        print("  lm_head.weight not in weight_map, already tied")
        return

    lm_head_shard = weight_map.pop("lm_head.weight")
    shard_path = output_dir / lm_head_shard

    # 해당 샤드에서 lm_head.weight 제거
    with safe_open(str(shard_path), framework="pt") as f:
        keys = list(f.keys())
        metadata = f.metadata()

    if "lm_head.weight" in keys:
        tensors = {}
        with safe_open(str(shard_path), framework="pt") as f:
            for key in keys:
                if key != "lm_head.weight":
                    tensors[key] = f.get_tensor(key)

        if tensors:
            original_size = shard_path.stat().st_size
            save_file(tensors, str(shard_path), metadata=metadata or {})
            new_size = shard_path.stat().st_size
            saved = (original_size - new_size) / 1e6
            print(f"  Removed lm_head.weight from {lm_head_shard} ({saved:.1f} MB saved)")
        else:
            # 샤드에 lm_head.weight만 있었으면 파일 삭제
            shard_path.unlink()
            print(f"  Removed empty shard: {lm_head_shard}")

    # 인덱스 업데이트
    # total_size 재계산
    total_size = 0
    for shard_file in set(weight_map.values()):
        sp = output_dir / shard_file
        if sp.exists():
            with safe_open(str(sp), framework="pt") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    total_size += t.numel() * t.element_size()

    index["metadata"]["total_size"] = total_size
    index["weight_map"] = weight_map

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Updated index: {index_path.name}")


def quantize_fp8(
    model_path: str,
    output_path: str,
    num_samples: int = 512,
    max_length: int = 2048,
    kv_cache_fp8: bool = False,
):
    """FP8 양자화 실행 (llmcompressor 사용)"""

    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max length: {max_length}")
    print(f"KV Cache FP8: {kv_cache_fp8}")
    print("=" * 60)

    # 토크나이저 로드
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 모델 로드
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # 캘리브레이션 데이터 준비
    print("Preparing calibration data from MANTA-1M...")
    ds = prepare_calibration_dataset(
        tokenizer,
        num_samples=num_samples,
        max_length=max_length
    )

    # 레시피 선택
    recipe = FP8_KV_CACHE_RECIPE if kv_cache_fp8 else FP8_RECIPE

    # 양자화 실행 (oneshot calibration)
    print("Running FP8 quantization with static calibration...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        save_compressed=True,
        output_dir=output_path,
        max_seq_length=max_length,
        num_calibration_samples=num_samples,
    )

    # 토크나이저 저장
    tokenizer.save_pretrained(output_path)

    # tie_word_embeddings 후처리: llmcompressor가 lm_head.weight를 별도 저장한 경우 제거
    print("\nChecking weight tying...")
    fix_tied_weights_after_save(output_path)

    print("=" * 60)
    print(f"FP8 quantization complete: {output_path}")

    # 모델 크기 비교
    original_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors"))
    quantized_size = sum(f.stat().st_size for f in Path(output_path).glob("*.safetensors"))

    if original_size > 0 and quantized_size > 0:
        print(f"Original size: {original_size / 1e9:.2f} GB")
        print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
        print(f"Compression ratio: {original_size / quantized_size:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="FP8 (E4M3) Quantization with llmcompressor (compressed-tensors)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/EXAONE-4.0-1.2B",
        help="Input model path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path (default: {model}-FP8)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=512,
        help="Number of calibration samples (default: 512)"
    )
    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=2048,
        help="Max sequence length for calibration (default: 2048)"
    )
    parser.add_argument(
        "--kv-cache-fp8",
        action="store_true",
        help="Enable FP8 KV cache quantization"
    )

    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"{args.model}-FP8"

    quantize_fp8(
        model_path=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        max_length=args.max_length,
        kv_cache_fp8=args.kv_cache_fp8,
    )


if __name__ == "__main__":
    main()
