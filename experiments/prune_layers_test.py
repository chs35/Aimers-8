"""
EXAONE-4.0-1.2B 레이어 프루닝 TPOT 검증 스크립트

레이어를 통째로 제거한 모델을 생성하여 TPOT 변화만 측정.
정확도는 떨어지지만, 레이어 수 vs TPOT 관계를 검증하는 것이 목적.

사용법:
  # 레이어 25,26,27 제거
  python experiments/prune_layers_test.py --remove 25,26,27

  # 30→24 레이어 (자동 선택)
  python experiments/prune_layers_test.py --keep-layers 24
"""

import argparse
import json
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def select_layers(total_layers: int, keep: int) -> list[int]:
    """
    유지할 레이어 인덱스 선택.
    앞쪽/뒤쪽 레이어를 우선 보존하고, 중간 레이어를 균등 간격으로 선택.
    - 초기 레이어: 일반적 특징 (syntax, surface) 보존
    - 최종 레이어: task-specific (output formatting) 보존
    - 중간 레이어: 균등 샘플링
    """
    if keep >= total_layers:
        return list(range(total_layers))

    # 앞 25%, 뒤 25% 보존, 나머지 중간에서 균등 선택
    front = max(2, keep // 4)
    back = max(2, keep // 4)
    middle_count = keep - front - back

    front_layers = list(range(front))
    back_layers = list(range(total_layers - back, total_layers))

    # 중간 영역에서 균등 간격 선택
    middle_start = front
    middle_end = total_layers - back
    if middle_count > 0 and middle_end > middle_start:
        step = (middle_end - middle_start) / (middle_count + 1)
        middle_layers = [int(middle_start + step * (i + 1)) for i in range(middle_count)]
    else:
        middle_layers = []

    selected = sorted(set(front_layers + middle_layers + back_layers))
    return selected[:keep]


def prune_model(
    model_path: str,
    output_path: str,
    layer_indices: list[int],
):
    print("=" * 60)
    print("EXAONE-4.0-1.2B Layer Pruning (TPOT Verification)")
    print("=" * 60)

    print(f"\nModel: {model_path}")
    print(f"Output: {output_path}")
    print(f"Keeping {len(layer_indices)} layers: {layer_indices}")

    # 토크나이저 로드
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 모델 로드
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    original_layers = len(model.model.layers)
    print(f"Original layers: {original_layers}")

    # 레이어 선택
    new_layers = torch.nn.ModuleList(
        [model.model.layers[i] for i in layer_indices]
    )
    model.model.layers = new_layers

    # config 업데이트
    model.config.num_hidden_layers = len(layer_indices)

    # layer_types 업데이트 (EXAONE-4.0 특유 필드)
    if hasattr(model.config, "layer_types") and model.config.layer_types:
        model.config.layer_types = [
            model.config.layer_types[i] for i in layer_indices
        ]

    print(f"Pruned layers: {len(layer_indices)}")

    # 저장
    print(f"\nSaving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # 파일 크기 출력
    print("\nGenerated files:")
    total_size = 0
    for f in sorted(Path(output_path).glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.2f} MB")
    print(f"  Total: {total_size:.2f} MB")

    print(f"\n레이어 프루닝 완료: {original_layers} → {len(layer_indices)}")
    print(f"벤치마크: python benchmark.py gsm8k -m {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="EXAONE-4.0-1.2B Layer Pruning for TPOT verification"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="원본 모델 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="출력 디렉토리 (모델명은 자동 생성)",
    )
    parser.add_argument(
        "--keep-layers",
        type=int,
        default=None,
        help="유지할 레이어 수 (자동 선택)",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default=None,
        help="유지할 레이어 인덱스 (쉼표 구분, 예: 0,1,2,5,10,15,28,29)",
    )

    args = parser.parse_args()

    total_layers = 30  # EXAONE-4.0-1.2B

    if args.layer_indices:
        layer_indices = [int(x) for x in args.layer_indices.split(",")]
    elif args.keep_layers:
        layer_indices = select_layers(total_layers, args.keep_layers)
    else:
        parser.error("--keep-layers 또는 --layer-indices 중 하나를 지정하세요")

    # 유효성 검사
    for idx in layer_indices:
        if idx < 0 or idx >= total_layers:
            parser.error(f"레이어 인덱스 {idx}가 범위 밖입니다 (0-{total_layers-1})")

    layer_indices = sorted(set(layer_indices))
    n = len(layer_indices)
    output_path = os.path.join(args.output_dir, f"EXAONE-4.0-1.2B-pruned-{n}L")

    prune_model(
        model_path=args.model_path,
        output_path=output_path,
        layer_indices=layer_indices,
    )


if __name__ == "__main__":
    main()
