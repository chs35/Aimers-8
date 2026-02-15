"""
run_pipeline: 경량화 파이프라인 오케스트레이터

EXAONE-4.0-1.2B 경량화 해커톤용 4단계 파이프라인을 순차 실행한다.
각 단계는 subprocess.run()으로 실행되며, 실패 시 즉시 중단된다.

파이프라인 흐름:
  Step 0: prepare_dataset.py   → dataset/dataset_N/{wp_calib, kd_train, fp8_calib}
           7개 HF 데이터셋에서 disjoint 3-split 생성
  Step 1: prune_vocab.py       → models/<name>-VocabPruned
           불필요한 언어의 토큰 제거 (임베딩 축소)
  Step 2: WidthPruning.py      → models/<name>-WP<size>
           FFN intermediate 채널 프루닝 (wanda 중요도 기반)
  Step 3: KD_Recovery.py       → models/<name>-KD
           Teacher(vocab-pruned)로 Knowledge Distillation 복구
  Step 4: quantize_fp8.py      → models/<name>-FP8
           FP8 (E4M3) Static Quantization (llmcompressor)

데이터 분리:
  각 단계가 서로 겹치지 않는(disjoint) 데이터셋을 사용한다.
  Step 0에서 글로벌 풀을 셔플 후 순서대로 3등분:
    wp_calib  (500)  → Step 2 Width Pruning 캘리브레이션
    kd_train  (2000) → Step 3 KD Recovery 학습
    fp8_calib (512)  → Step 4 FP8 양자화 캘리브레이션

사용법:
  # 1) 전체 파이프라인 (HF에서 모델 다운로드 + 데이터셋 생성 + 모든 단계)
  python generate/run_pipeline.py --base-model LGAI-EXAONE/EXAONE-4.0-1.2B

  # 2) 이미 로컬에 모델이 있는 경우
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B

  # 3) 데이터셋이 이미 생성되어 있으면 건너뛰기
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B --skip-dataset-prep
  #    → dataset/ 하위에서 가장 최근 dataset_N을 자동 탐지하여 사용

  # 4) 특정 데이터셋 디렉토리를 직접 지정
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B \
      --dataset-dir dataset/dataset_3

  # 5) 일부 단계만 건너뛰기 (예: Vocab Pruning 이미 완료)
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B \
      --skip-dataset-prep --skip-vocab

  # 6) KD용 Teacher를 별도 지정 (기본값: vocab-pruned 모델)
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B \
      --teacher models/EXAONE-4.0-1.2B-VocabPruned

  # 7) 데이터셋 샘플 수 조절
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B \
      --wp-samples 256 --kd-samples 3000 --fp8-samples 256

  # 8) FP8만 실행 (나머지 모두 건너뛰기)
  python generate/run_pipeline.py --base-model models/EXAONE-4.0-1.2B-WP5152-KD \
      --skip-dataset-prep --skip-vocab --skip-width --skip-kd

주요 인수:
  --base-model       시작 모델 (HF ID 또는 로컬 경로, 기본: LGAI-EXAONE/EXAONE-4.0-1.2B)
  --teacher          KD teacher 모델 경로 (기본: Step 1 출력인 vocab-pruned 모델)
  --dataset-dir      데이터셋 디렉토리 경로 (기본: dataset/ 하위 최신 dataset_N 자동 탐지)
  --output-base      모델 출력 base 디렉토리 (기본: models)
  --skip-*           해당 단계 건너뛰기 (dataset-prep, vocab, width, kd, fp8)
  --wp-samples       Width Pruning 캘리브레이션 샘플 수 (기본: 500)
  --kd-samples       KD 학습 샘플 수 (기본: 2000)
  --fp8-samples      FP8 캘리브레이션 샘플 수 (기본: 512)
  --seed             랜덤 시드 (기본: 42)
"""

import os
import sys
import json
import argparse
import subprocess
import re


def find_latest_dataset_dir(base_dir):
    """dataset/dataset_N 중 가장 최근(N이 가장 큰) 디렉토리를 찾는다."""
    if not os.path.isdir(base_dir):
        return None
    pattern = re.compile(r"^dataset_(\d+)$")
    max_n = -1
    best = None
    for entry in os.listdir(base_dir):
        m = pattern.match(entry)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
                best = os.path.join(base_dir, entry)
    return best


def run_step(step_name, cmd):
    """서브프로세스로 단계를 실행. 실패 시 즉시 중단."""
    print()
    print("=" * 60)
    print(f"  {step_name}")
    print("=" * 60)
    print(f"  CMD: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] {step_name} 실패 (exit code: {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  {step_name} 완료!")


def main():
    parser = argparse.ArgumentParser(description="경량화 파이프라인 오케스트레이터")
    parser.add_argument("--base-model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="시작 모델 (HF ID 또는 로컬 경로)")
    parser.add_argument("--teacher", type=str, default=None,
                        help="KD teacher 모델 (기본값: vocab-pruned 모델)")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="데이터셋 디렉토리 (기본: dataset → 자동으로 최신 dataset_N 탐지)")
    parser.add_argument("--output-base", type=str, default="models",
                        help="모델 출력 base 디렉토리 (기본값: models)")

    # 단계 건너뛰기
    parser.add_argument("--skip-dataset-prep", action="store_true",
                        help="데이터셋 준비 단계 건너뛰기")
    parser.add_argument("--skip-vocab", action="store_true",
                        help="Vocab Pruning 단계 건너뛰기")
    parser.add_argument("--skip-width", action="store_true",
                        help="Width Pruning 단계 건너뛰기")
    parser.add_argument("--skip-kd", action="store_true",
                        help="KD Recovery 단계 건너뛰기")
    parser.add_argument("--skip-fp8", action="store_true",
                        help="FP8 Quantization 단계 건너뛰기")

    # 데이터셋 샘플 수
    parser.add_argument("--wp-samples", type=int, default=500,
                        help="Width Pruning 캘리브레이션 샘플 수")
    parser.add_argument("--kd-samples", type=int, default=2000,
                        help="KD 학습 샘플 수")
    parser.add_argument("--fp8-samples", type=int, default=512,
                        help="FP8 캘리브레이션 샘플 수")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")

    args = parser.parse_args()

    base_model = args.base_model
    base_name = os.path.basename(base_model.rstrip("/\\"))

    print("=" * 60)
    print("경량화 파이프라인")
    print("=" * 60)
    print(f"  Base model: {base_model}")
    print(f"  Output base: {args.output_base}")
    print(f"  Skip: dataset={args.skip_dataset_prep}, vocab={args.skip_vocab}, "
          f"width={args.skip_width}, kd={args.skip_kd}, fp8={args.skip_fp8}")
    print()

    # ── Step 0: 데이터셋 준비 ──
    dataset_dir = args.dataset_dir

    if not args.skip_dataset_prep:
        run_step("Step 0: 데이터셋 준비", [
            sys.executable, "generate/prepare_dataset.py",
            "--output", "dataset",
            "--wp-samples", str(args.wp_samples),
            "--kd-samples", str(args.kd_samples),
            "--fp8-samples", str(args.fp8_samples),
            "--seed", str(args.seed),
        ])

    # dataset_dir 결정
    if dataset_dir is None:
        dataset_dir = find_latest_dataset_dir("dataset")
        if dataset_dir is None:
            print("[ERROR] dataset/ 하위에 dataset_N 디렉토리가 없습니다.")
            print("  --skip-dataset-prep 없이 실행하거나 --dataset-dir를 지정하세요.")
            sys.exit(1)

    print(f"\n  사용할 데이터셋: {dataset_dir}")

    # 데이터셋 경로
    wp_calib_path = os.path.join(dataset_dir, "wp_calib")
    kd_train_path = os.path.join(dataset_dir, "kd_train")
    fp8_calib_path = os.path.join(dataset_dir, "fp8_calib")

    # 경로 검증
    for path, name in [(wp_calib_path, "wp_calib"), (kd_train_path, "kd_train"), (fp8_calib_path, "fp8_calib")]:
        if not os.path.isdir(path):
            print(f"[WARNING] {name} 경로가 없습니다: {path}")

    # ── Step 1: Vocab Pruning ──
    current_model = base_model
    if not args.skip_vocab:
        vocab_output = os.path.join(args.output_base, f"{base_name}-VocabPruned")
        run_step("Step 1: Vocab Pruning", [
            sys.executable, "experiments/prune_vocab.py",
            "--model", current_model,
            "--output", vocab_output,
        ])
        current_model = vocab_output
    else:
        print("\n  Step 1: Vocab Pruning 건너뜀")
        # skip vocab이면 base_model이 이미 vocab pruned일 수 있음
        vocab_pruned_path = os.path.join(args.output_base, f"{base_name}-VocabPruned")
        if os.path.isdir(vocab_pruned_path):
            current_model = vocab_pruned_path
            print(f"  기존 vocab-pruned 모델 사용: {current_model}")

    # Teacher 결정
    teacher_model = args.teacher or current_model

    # ── Step 2: Width Pruning ──
    if not args.skip_width:
        wp_cmd = [
            sys.executable, "generate/WidthPruning.py",
            "--model", current_model,
            "--dataset", wp_calib_path,
        ]
        run_step("Step 2: Width Pruning", wp_cmd)
        # WidthPruning.py의 기본 출력 패턴: models/<name>-WP<intermediate_size>
        # 출력 경로를 찾기 위해 models/ 하위에서 탐색
        wp_candidates = []
        model_base = os.path.basename(current_model.rstrip("/\\"))
        if os.path.isdir(args.output_base):
            for entry in os.listdir(args.output_base):
                if entry.startswith(model_base) and "-WP" in entry:
                    wp_candidates.append(os.path.join(args.output_base, entry))
        if wp_candidates:
            # 가장 최근 수정된 것
            wp_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            current_model = wp_candidates[0]
        print(f"  Width Pruning 출력: {current_model}")
    else:
        print("\n  Step 2: Width Pruning 건너뜀")

    # ── Step 3: KD Recovery ──
    if not args.skip_kd:
        kd_cmd = [
            sys.executable, "generate/KD_Recovery.py",
            "--model", current_model,
            "--teacher", teacher_model,
            "--dataset", kd_train_path,
        ]
        run_step("Step 3: KD Recovery", kd_cmd)
        # 기본 출력: models/<name>-KD
        kd_output = os.path.join(args.output_base,
                                 f"{os.path.basename(current_model.rstrip('/'))}-KD")
        if os.path.isdir(kd_output):
            current_model = kd_output
        print(f"  KD Recovery 출력: {current_model}")
    else:
        print("\n  Step 3: KD Recovery 건너뜀")

    # ── Step 4: FP8 Quantization ──
    if not args.skip_fp8:
        fp8_output = f"{current_model}-FP8"
        fp8_cmd = [
            sys.executable, "experiments/quantize_fp8.py",
            "--model", current_model,
            "--output", fp8_output,
            "--dataset", fp8_calib_path,
        ]
        run_step("Step 4: FP8 Quantization", fp8_cmd)
        current_model = fp8_output
        print(f"  FP8 Quantization 출력: {current_model}")
    else:
        print("\n  Step 4: FP8 Quantization 건너뜀")

    # ── 최종 요약 ──
    print()
    print("=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)
    print(f"  최종 모델: {current_model}")
    print(f"  데이터셋:  {dataset_dir}")


if __name__ == "__main__":
    main()
