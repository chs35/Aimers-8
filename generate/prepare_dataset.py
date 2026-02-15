"""
prepare_dataset: 8개 HuggingFace 데이터셋에서 Disjoint 3-Split 생성

파이프라인의 각 단계(Width Pruning, KD Recovery, FP8 Quantization)가
서로 겹치지 않는(disjoint) 데이터를 사용하도록 보장한다.

통합 포맷: {"conversations": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "source": "dataset_name"}

데이터셋:
  SFT 메인:
    1. microsoft/orca-math-word-problems-200k (MIT) - 영어 수학
    2. Open-Orca/SlimOrca (MIT) - 영어 일반
    3. kuotient/orca-math-word-problems-193k-korean (CC-BY-SA-4.0) - 한글 수학
    4. nlpai-lab/kullm-v2 (Apache-2.0) - 한글 일반
  Calibration / 보강:
    5. openai/gsm8k (MIT) - 영어 수학 Calibration
    6. LGAI-EXAONE/MANTA-1M (CC-BY-NC-4.0) - STEM 강화
    7. kuotient/orca-math-korean-preference (CC-BY-SA-4.0) - DPO/Preference
    8. TIGER-Lab/MMLU-Pro (MIT) - 영어 STEM Calibration

사용법:
  python generate/prepare_dataset.py --output dataset --wp-samples 500 --kd-samples 2000 --fp8-samples 512 --seed 42
  # → dataset/dataset_1/{wp_calib, kd_train, fp8_calib, metadata.json}
"""

import os
import json
import argparse
import random
from datetime import datetime

from datasets import load_dataset, Dataset


# ╔══════════════════════════════════════════════════════════════╗
# ║  데이터셋별 정규화 함수                                       ║
# ╚══════════════════════════════════════════════════════════════╝

DATASET_CONFIGS = [
    # ── SFT 메인 ──
    {
        "id": "microsoft/orca-math-word-problems-200k",
        "name": "orca-math",
        "split": "train",
        "default_samples": 600,
    },
    {
        "id": "Open-Orca/SlimOrca",
        "name": "SlimOrca",
        "split": "train",
        "default_samples": 600,
    },
    {
        "id": "kuotient/orca-math-word-problems-193k-korean",
        "name": "orca-math-193k-korean",
        "split": "train",
        "default_samples": 500,
    },
    {
        "id": "nlpai-lab/kullm-v2",
        "name": "kullm-v2",
        "split": "train",
        "default_samples": 500,
    },
    # ── Calibration / 보강 ──
    {
        "id": "openai/gsm8k",
        "name": "gsm8k",
        "config": "main",
        "split": "train",
        "default_samples": 400,
    },
    {
        "id": "LGAI-EXAONE/MANTA-1M",
        "name": "MANTA-1M",
        "split": "train",
        "default_samples": 500,
    },
    {
        "id": "kuotient/orca-math-korean-preference",
        "name": "orca-math-korean-pref",
        "split": "train",
        "default_samples": 300,
    },
    {
        "id": "TIGER-Lab/MMLU-Pro",
        "name": "MMLU-Pro",
        "split": "test",
        "default_samples": 300,
    },
]


def normalize_orca_math(example):
    """orca-math-word-problems-200k: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_slimorca(example):
    """SlimOrca: from/value → role/content 변환 (human→user, gpt→assistant)"""
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    conversations = []
    for turn in example["conversations"]:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        conversations.append({"role": role, "content": turn["value"]})
    return conversations


def normalize_orca_math_193k_korean(example):
    """orca-math-word-problems-193k-korean: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_kullm_v2(example):
    """kullm-v2: instruction/input/output → user/assistant"""
    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n\n" + example["input"]
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]


def normalize_gsm8k(example):
    """gsm8k: question/answer → user/assistant (CoT reasoning 포함)"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_manta_1m(example):
    """MANTA-1M: conversations 필드가 이미 role/content 포맷"""
    conversations = example["conversations"]
    # 이미 {"role": ..., "content": ...} 형태인 경우
    if conversations and isinstance(conversations[0], dict) and "role" in conversations[0]:
        return conversations
    # from/value 형태인 경우 (혹시 모를 대비)
    ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {"role": ROLE_MAP.get(t.get("from", ""), t.get("from", "")),
         "content": t.get("value", t.get("content", ""))}
        for t in conversations
    ]


def normalize_orca_math_korean_pref(example):
    """orca-math-korean-preference: question/answer → user/assistant"""
    return [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]


def normalize_mmlu_pro(example):
    """MMLU-Pro: MCQ 포맷팅 (question + options → user, answer → assistant)"""
    options = example["options"]
    labels = [chr(ord("A") + i) for i in range(len(options))]
    choices_text = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(options))
    user_content = f"{example['question']}\n\n{choices_text}"

    answer = example.get("answer", "")
    # answer가 문자열(A, B, ...)인 경우 해당 선지 텍스트도 포함
    if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
        idx = ord(answer.upper()) - ord("A")
        if 0 <= idx < len(options):
            answer_text = f"{answer.upper()}. {options[idx]}"
        else:
            answer_text = answer
    elif isinstance(answer, int) and 0 <= answer < len(options):
        answer_text = f"{labels[answer]}. {options[answer]}"
    else:
        answer_text = str(answer)

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer_text},
    ]


NORMALIZERS = {
    "orca-math": normalize_orca_math,
    "SlimOrca": normalize_slimorca,
    "orca-math-193k-korean": normalize_orca_math_193k_korean,
    "kullm-v2": normalize_kullm_v2,
    "gsm8k": normalize_gsm8k,
    "MANTA-1M": normalize_manta_1m,
    "orca-math-korean-pref": normalize_orca_math_korean_pref,
    "MMLU-Pro": normalize_mmlu_pro,
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  핵심 로직                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def load_and_normalize(config, num_samples, seed=42):
    """단일 데이터셋 로드 → 정규화 → conversations 형식으로 변환"""
    ds_id = config["id"]
    ds_name = config["name"]
    ds_split = config["split"]
    ds_config = config.get("config")
    normalizer = NORMALIZERS[ds_name]

    print(f"  [{ds_name}] {ds_id} (split={ds_split}) 에서 {num_samples}개 로드 중...")

    try:
        # 스트리밍으로 필요한 만큼만 로드
        raw_ds = load_dataset(ds_id, ds_config, split=ds_split, streaming=True)
        raw_ds = raw_ds.shuffle(seed=seed, buffer_size=5000)
        samples = list(raw_ds.take(num_samples))
    except Exception as e:
        print(f"  [{ds_name}] 스트리밍 로드 실패, 일반 로드 시도: {e}")
        try:
            raw_ds = load_dataset(ds_id, ds_config, split=ds_split)
            raw_ds = raw_ds.shuffle(seed=seed)
            samples = list(raw_ds.select(range(min(num_samples, len(raw_ds)))))
        except Exception as e2:
            print(f"  [{ds_name}] 로드 실패, 건너뜀: {e2}")
            return []

    normalized = []
    skipped = 0
    for sample in samples:
        try:
            conversations = normalizer(sample)
            # 빈 대화 필터링
            if conversations and all(turn.get("content") for turn in conversations):
                normalized.append({
                    "conversations": conversations,
                    "source": ds_name,
                })
            else:
                skipped += 1
        except Exception:
            skipped += 1

    print(f"  [{ds_name}] {len(normalized)}개 정규화 완료 (건너뜀: {skipped})")
    return normalized


def find_next_dataset_dir(base_dir):
    """dataset/dataset_N에서 N을 자동 증가"""
    n = 1
    while os.path.exists(os.path.join(base_dir, f"dataset_{n}")):
        n += 1
    return n, os.path.join(base_dir, f"dataset_{n}")


def main():
    parser = argparse.ArgumentParser(description="Disjoint 3-Split 데이터셋 생성")
    parser.add_argument("--output", type=str, default="dataset",
                        help="출력 base 디렉토리 (기본값: dataset)")
    parser.add_argument("--wp-samples", type=int, default=500,
                        help="Width Pruning 캘리브레이션 샘플 수 (기본값: 500)")
    parser.add_argument("--kd-samples", type=int, default=2000,
                        help="KD 학습 샘플 수 (기본값: 2000)")
    parser.add_argument("--fp8-samples", type=int, default=512,
                        help="FP8 캘리브레이션 샘플 수 (기본값: 512)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본값: 42)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="각 데이터셋의 기본 샘플 수에 곱할 배율 (기본값: 1.0)")
    args = parser.parse_args()

    total_needed = args.wp_samples + args.kd_samples + args.fp8_samples

    print("=" * 60)
    print("Disjoint 3-Split 데이터셋 생성")
    print("=" * 60)
    print(f"  WP 캘리브레이션: {args.wp_samples}")
    print(f"  KD 학습:        {args.kd_samples}")
    print(f"  FP8 캘리브레이션: {args.fp8_samples}")
    print(f"  총 필요:         {total_needed}")
    print(f"  시드:            {args.seed}")
    print()

    # ── Step 1: 각 데이터셋 로드 + 정규화 ──
    print("=" * 60)
    print("Step 1: 데이터셋 로드 + 정규화")
    print("=" * 60)

    global_pool = []
    source_counts = {}

    # 필터링 손실을 고려해 기본 샘플 수에 20% 버퍼 추가
    default_total = sum(c["default_samples"] for c in DATASET_CONFIGS)
    if default_total * args.scale < total_needed * 1.2:
        auto_scale = (total_needed * 1.2) / default_total
        print(f"  [자동 스케일] 기본 풀({default_total})이 필요량({total_needed})에 부족할 수 있어 "
              f"scale {args.scale:.1f} → {auto_scale:.2f}로 조정")
        effective_scale = max(args.scale, auto_scale)
    else:
        effective_scale = args.scale

    for config in DATASET_CONFIGS:
        num = int(config["default_samples"] * effective_scale)
        normalized = load_and_normalize(config, num, seed=args.seed)
        global_pool.extend(normalized)
        source_counts[config["name"]] = len(normalized)

    print()
    print(f"  글로벌 풀 크기: {len(global_pool)}")
    for name, count in source_counts.items():
        print(f"    {name}: {count}")

    if len(global_pool) < total_needed:
        print(f"\n  [경고] 풀 크기({len(global_pool)}) < 필요 샘플 수({total_needed})")
        print(f"  --scale 값을 높이거나 샘플 수를 줄여주세요.")
        print(f"  가용한 샘플로 비율 기반 분할합니다.")
    print()

    # ── Step 2: 셔플 + 3-Split 분할 ──
    print("=" * 60)
    print("Step 2: 셔플 + Disjoint 3-Split 분할")
    print("=" * 60)

    random.seed(args.seed)
    random.shuffle(global_pool)

    pool_size = len(global_pool)

    if pool_size >= total_needed:
        # 충분한 경우: 요청한 크기대로 분할
        wp_end = args.wp_samples
        kd_end = wp_end + args.kd_samples
        fp8_end = kd_end + args.fp8_samples
    else:
        # 부족한 경우: 비율 기반 분할 (모든 split이 최소 1개씩 받도록)
        wp_ratio = args.wp_samples / total_needed
        kd_ratio = args.kd_samples / total_needed
        wp_count = max(1, int(pool_size * wp_ratio))
        kd_count = max(1, int(pool_size * kd_ratio))
        fp8_count = max(1, pool_size - wp_count - kd_count)
        wp_end = wp_count
        kd_end = wp_end + kd_count
        fp8_end = kd_end + fp8_count
        print(f"  [비율 분할] 풀({pool_size}) < 필요({total_needed}), "
              f"비율 기반 분할: wp={wp_count}, kd={kd_count}, fp8={fp8_count}")

    wp_split = global_pool[:wp_end]
    kd_split = global_pool[wp_end:kd_end]
    fp8_split = global_pool[kd_end:fp8_end]

    print(f"  wp_calib:  {len(wp_split)} 샘플")
    print(f"  kd_train:  {len(kd_split)} 샘플")
    print(f"  fp8_calib: {len(fp8_split)} 샘플")
    print(f"  미사용:    {pool_size - fp8_end} 샘플")
    print()

    # ── Step 3: 저장 ──
    print("=" * 60)
    print("Step 3: Arrow 포맷으로 저장")
    print("=" * 60)

    dataset_n, dataset_dir = find_next_dataset_dir(args.output)
    os.makedirs(dataset_dir, exist_ok=True)

    splits = {
        "wp_calib": wp_split,
        "kd_train": kd_split,
        "fp8_calib": fp8_split,
    }

    for split_name, split_data in splits.items():
        if not split_data:
            print(f"  [경고] {split_name}: 0 샘플 — 건너뜀 (풀이 부족합니다)")
            continue
        split_dir = os.path.join(dataset_dir, split_name)
        ds = Dataset.from_list(split_data)
        ds.save_to_disk(split_dir)
        print(f"  {split_name}: {len(ds)} 샘플 → {split_dir}/")

    # 메타데이터 저장
    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": args.seed,
        "scale": args.scale,
        "total_pool_size": pool_size,
        "splits": {
            "wp_calib": len(wp_split),
            "kd_train": len(kd_split),
            "fp8_calib": len(fp8_split),
        },
        "source_counts": source_counts,
        "dataset_configs": [
            {"id": c["id"], "name": c["name"], "split": c["split"],
             "requested": int(c["default_samples"] * args.scale)}
            for c in DATASET_CONFIGS
        ],
    }
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  metadata: {meta_path}")

    print()
    print("=" * 60)
    print(f"완료! 데이터셋 저장 위치: {dataset_dir}/")
    print("=" * 60)
    print(f"  wp_calib:  {len(wp_split)} 샘플 (Width Pruning 캘리브레이션)")
    print(f"  kd_train:  {len(kd_split)} 샘플 (KD 학습)")
    print(f"  fp8_calib: {len(fp8_split)} 샘플 (FP8 캘리브레이션)")


if __name__ == "__main__":
    main()
