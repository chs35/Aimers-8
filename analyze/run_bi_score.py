#!/usr/bin/env python3
"""
Block Influence (BI) Score 분석 스크립트 (ShortGPT 방식)

각 트랜스포머 블록의 입력/출력 히든 스테이트 간 코사인 유사도를 측정하여
BI Score = 1 - E[cos_sim(input, output)] 를 계산한다.

BI Score가 낮은 레이어는 입력을 거의 변형하지 않으므로 제거(pruning) 후보가 된다.

캘리브레이션 데이터: LGAI-EXAONE/MANTA-1M

출력물:
  - JSON:  analyze/results/{model_name}_bi_scores.json
  - Plot:  analyze/results/{model_name}_bi_scores.png (matplotlib 설치시)
  - 콘솔:  레이어별 BI Score 테이블
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_DIR = Path("models")
RESULTS_DIR = Path("analyze/results")
CALIBRATION_DATASET = "LGAI-EXAONE/MANTA-1M"


# ═══════════════════════════════════════════════════════════
#  모델 선택 (run_gsm8k.py 패턴 재사용)
# ═══════════════════════════════════════════════════════════

def discover_models(models_dir: Path) -> list[Path]:
    """models/ 하위에서 config.json이 존재하는 모델 디렉토리 목록을 반환한다."""
    if not models_dir.exists():
        return []
    return sorted(
        child for child in models_dir.iterdir()
        if child.is_dir() and (child / "config.json").exists()
    )


def select_model_interactive(models_dir: Path) -> Path:
    """인터랙티브 모델 선택 UI."""
    models = discover_models(models_dir)

    if not models:
        print(f"[ERROR] '{models_dir}' 디렉토리에 모델이 없습니다.")
        print("먼저 init_model.py를 실행하여 모델을 다운로드하세요.")
        sys.exit(1)

    print("=" * 50)
    print("사용 가능한 모델:")
    print("=" * 50)
    for idx, model_path in enumerate(models, 1):
        size_bytes = sum(
            f.stat().st_size for f in model_path.glob("*.safetensors")
        )
        size_gb = size_bytes / 1e9 if size_bytes > 0 else 0
        size_str = f" ({size_gb:.2f} GB)" if size_gb > 0 else ""
        print(f"  [{idx}] {model_path.name}{size_str}")
    print("=" * 50)

    while True:
        try:
            choice = input(f"모델 번호를 선택하세요 (1-{len(models)}): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(models):
                selected = models[idx - 1]
                print(f"선택된 모델: {selected}")
                return selected
            print(f"1부터 {len(models)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")
        except (KeyboardInterrupt, EOFError):
            print("\n취소되었습니다.")
            sys.exit(0)


def resolve_model_path(model_arg: str | None, models_dir: Path) -> Path:
    """CLI 인자 또는 인터랙티브 선택으로 모델 경로를 확정한다."""
    if model_arg is not None:
        model_path = Path(model_arg)
        if not model_path.exists():
            print(f"[ERROR] 모델 경로가 존재하지 않습니다: {model_path}")
            sys.exit(1)
        return model_path
    return select_model_interactive(models_dir)


# ═══════════════════════════════════════════════════════════
#  모델 아키텍처 탐색
# ═══════════════════════════════════════════════════════════

def get_decoder_layers(model) -> tuple[list, str]:
    """모델 아키텍처에서 디코더 레이어 리스트와 경로를 반환한다.

    Returns:
        (layers, layer_path_name)
    """
    # LLaMA, Mistral, Qwen2, EXAONE, Phi-3, Gemma 등
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), "model.model.layers"
    # GPT-2, GPT-Neo 등
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h), "model.transformer.h"
    # GPT-NeoX, Pythia 등
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers), "model.gpt_neox.layers"
    # Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h), "model.transformer.h"

    raise ValueError(
        f"지원하지 않는 모델 아키텍처입니다: {type(model).__name__}\n"
        f"model attributes: {[a for a in dir(model) if not a.startswith('_')]}"
    )


# ═══════════════════════════════════════════════════════════
#  캘리브레이션 데이터 로드
# ═══════════════════════════════════════════════════════════

def load_calibration_data(
    tokenizer: AutoTokenizer,
    num_samples: int,
    max_length: int,
) -> list[dict]:
    """MANTA-1M 데이터셋에서 캘리브레이션 샘플을 로드한다.

    각 샘플의 conversations를 chat template으로 변환 후 토크나이즈한다.
    """
    print(f"캘리브레이션 데이터 로드 중: {CALIBRATION_DATASET} ({num_samples}개)")
    ds = load_dataset(CALIBRATION_DATASET, split=f"train[:{num_samples}]")

    samples = []
    skipped = 0

    for item in tqdm(ds, desc="토크나이즈"):
        conversations = item["conversations"]

        # chat template 적용
        try:
            text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            skipped += 1
            continue

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        # 너무 짧은 시퀀스는 건너뛴다 (최소 10 토큰)
        if encoded["input_ids"].shape[1] < 10:
            skipped += 1
            continue

        samples.append(encoded)

    if skipped > 0:
        print(f"  건너뛴 샘플: {skipped}개 (chat template 실패 또는 너무 짧음)")
    print(f"  유효 샘플: {len(samples)}개")

    return samples


# ═══════════════════════════════════════════════════════════
#  BI Score 계산
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_bi_scores(
    model,
    layers: list,
    samples: list[dict],
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """각 레이어의 BI Score를 계산한다.

    Args:
        model: HuggingFace CausalLM 모델
        layers: 디코더 레이어 리스트
        samples: 토크나이즈된 캘리브레이션 샘플 리스트
        device: 연산 디바이스

    Returns:
        (bi_scores, avg_cos_sims) - 각 레이어별 BI 스코어와 평균 코사인 유사도
    """
    num_layers = len(layers)
    cos_sim_sum = [0.0] * num_layers
    token_count = [0] * num_layers

    # 각 레이어에 forward hook 등록
    layer_io: dict[int, dict] = {}

    def make_hook(idx: int):
        def hook_fn(module, inp, out):
            # inp: tuple, 첫 번째 원소가 히든 스테이트
            layer_io[idx] = {
                "input": inp[0].detach(),
                "output": out[0].detach() if isinstance(out, tuple) else out.detach(),
            }
        return hook_fn

    hooks = []
    for idx, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    try:
        for sample in tqdm(samples, desc="BI Score 계산"):
            # 샘플을 디바이스로 이동
            inputs = {k: v.to(device) for k, v in sample.items()}
            model(**inputs)

            # 각 레이어의 코사인 유사도 계산
            for idx in range(num_layers):
                if idx not in layer_io:
                    continue

                inp = layer_io[idx]["input"].float()
                out = layer_io[idx]["output"].float()

                # shape: [1, seq_len, hidden_dim]
                # 토큰별 코사인 유사도
                cos = F.cosine_similarity(inp, out, dim=-1)  # [1, seq_len]
                cos_sim_sum[idx] += cos.sum().item()
                token_count[idx] += cos.numel()

            layer_io.clear()

            # GPU 메모리 정리
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        for h in hooks:
            h.remove()

    # BI Score = 1 - avg(cos_sim)
    avg_cos_sims = []
    bi_scores = []
    for idx in range(num_layers):
        if token_count[idx] > 0:
            avg_cos = cos_sim_sum[idx] / token_count[idx]
        else:
            avg_cos = 0.0
        avg_cos_sims.append(avg_cos)
        bi_scores.append(1.0 - avg_cos)

    return bi_scores, avg_cos_sims


# ═══════════════════════════════════════════════════════════
#  결과 출력
# ═══════════════════════════════════════════════════════════

def print_bi_table(bi_scores: list[float], avg_cos_sims: list[float]) -> None:
    """레이어별 BI Score 테이블을 콘솔에 출력한다."""
    bar_width = 40
    max_bi = max(bi_scores) if bi_scores else 1.0

    print()
    print("=" * 75)
    print(f"{'Layer':>7} | {'BI Score':>10} | {'Cos Sim':>10} | {'Bar'}")
    print("-" * 75)

    for idx, (bi, cos) in enumerate(zip(bi_scores, avg_cos_sims)):
        bar_len = int((bi / max_bi) * bar_width) if max_bi > 0 else 0
        bar = "\u2588" * bar_len
        print(f"  {idx:>4}  | {bi:>10.6f} | {cos:>10.6f} | {bar}")

    print("=" * 75)

    # 요약 통계
    avg_bi = sum(bi_scores) / len(bi_scores) if bi_scores else 0
    min_bi = min(bi_scores) if bi_scores else 0
    max_bi_val = max(bi_scores) if bi_scores else 0
    min_idx = bi_scores.index(min_bi) if bi_scores else -1
    max_idx = bi_scores.index(max_bi_val) if bi_scores else -1

    print(f"\n평균 BI Score: {avg_bi:.6f}")
    print(f"최소 BI Score: {min_bi:.6f} (Layer {min_idx})")
    print(f"최대 BI Score: {max_bi_val:.6f} (Layer {max_idx})")

    # BI Score 하위 25% 레이어 (pruning 후보)
    sorted_layers = sorted(range(len(bi_scores)), key=lambda i: bi_scores[i])
    n_candidates = max(1, len(bi_scores) // 4)
    candidates = sorted_layers[:n_candidates]

    print(f"\nPruning 후보 (BI Score 하위 {n_candidates}개 레이어):")
    for idx in candidates:
        print(f"  Layer {idx:>3}: BI={bi_scores[idx]:.6f}")


def save_plot(
    bi_scores: list[float],
    model_name: str,
    output_path: Path,
) -> bool:
    """BI Score 막대 그래프를 PNG로 저장한다. matplotlib 미설치시 건너뛴다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 미설치: 그래프 저장을 건너뜁니다.")
        return False

    fig, ax = plt.subplots(figsize=(max(10, len(bi_scores) * 0.3), 6))

    colors = []
    avg_bi = sum(bi_scores) / len(bi_scores) if bi_scores else 0
    for bi in bi_scores:
        if bi < avg_bi * 0.5:
            colors.append("#2ecc71")  # 낮은 BI = 제거 후보 (녹색)
        elif bi < avg_bi:
            colors.append("#f39c12")  # 평균 이하 (주황)
        else:
            colors.append("#3498db")  # 평균 이상 (파랑)

    ax.bar(range(len(bi_scores)), bi_scores, color=colors, edgecolor="none")
    ax.axhline(y=avg_bi, color="red", linestyle="--", alpha=0.7, label=f"avg={avg_bi:.4f}")
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("BI Score", fontsize=12)
    ax.set_title(f"Block Influence Score - {model_name}", fontsize=14)
    ax.legend()
    ax.set_xticks(range(0, len(bi_scores), max(1, len(bi_scores) // 20)))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"그래프 저장: {output_path}")
    return True


def save_results_json(
    json_path: Path,
    model_path: Path,
    bi_scores: list[float],
    avg_cos_sims: list[float],
    num_samples: int,
    max_length: int,
    elapsed: float,
) -> None:
    """분석 결과를 JSON으로 저장한다."""
    avg_bi = sum(bi_scores) / len(bi_scores) if bi_scores else 0
    sorted_layers = sorted(range(len(bi_scores)), key=lambda i: bi_scores[i])
    n_candidates = max(1, len(bi_scores) // 4)

    report = {
        "meta": {
            "model": model_path.name,
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "calibration_dataset": CALIBRATION_DATASET,
            "num_samples": num_samples,
            "max_length": max_length,
            "duration_sec": round(elapsed, 2),
        },
        "summary": {
            "num_layers": len(bi_scores),
            "avg_bi_score": round(avg_bi, 6),
            "min_bi_score": round(min(bi_scores), 6) if bi_scores else 0,
            "max_bi_score": round(max(bi_scores), 6) if bi_scores else 0,
            "min_bi_layer": bi_scores.index(min(bi_scores)) if bi_scores else -1,
            "max_bi_layer": bi_scores.index(max(bi_scores)) if bi_scores else -1,
            "pruning_candidates": sorted_layers[:n_candidates],
        },
        "layers": [
            {
                "layer": idx,
                "bi_score": round(bi, 6),
                "cos_sim": round(cos, 6),
            }
            for idx, (bi, cos) in enumerate(zip(bi_scores, avg_cos_sims))
        ],
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"JSON 저장: {json_path}")


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block Influence (BI) Score 분석 (ShortGPT 방식)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="모델 경로 (미지정시 인터랙티브 선택)",
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=512,
        help="캘리브레이션 샘플 수 (기본값: 512)",
    )
    parser.add_argument(
        "--max-length", type=int, default=16384,
        help="최대 시퀀스 길이 (기본값: 16384)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(RESULTS_DIR),
        help=f"결과 저장 디렉토리 (기본값: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="모델 로드 dtype (기본값: float16)",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    start_time = time.time()

    # 1. 모델 경로 확정
    model_path = resolve_model_path(args.model, MODELS_DIR)
    model_name = model_path.name

    # 2. 토크나이저 로드
    print(f"\n토크나이저 로드: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 캘리브레이션 데이터 로드 + 토크나이즈
    samples = load_calibration_data(
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )

    if not samples:
        print("[ERROR] 유효한 캘리브레이션 샘플이 없습니다.")
        sys.exit(1)

    # 4. 모델 로드
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"\n모델 로드: {model_path} (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 디바이스 확인
    device = next(model.parameters()).device
    print(f"디바이스: {device}")

    # 5. 디코더 레이어 탐색
    layers, layer_path = get_decoder_layers(model)
    print(f"디코더 레이어: {len(layers)}개 ({layer_path})")

    # 6. BI Score 계산
    print(f"\nBI Score 계산 시작 ({len(samples)}개 샘플)...")
    bi_scores, avg_cos_sims = compute_bi_scores(
        model=model,
        layers=layers,
        samples=samples,
        device=device,
    )

    elapsed = time.time() - start_time

    # 7. 결과 출력
    print_bi_table(bi_scores, avg_cos_sims)
    print(f"\n소요 시간: {elapsed:.1f}초")

    # 8. 결과 저장
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"{model_name}_bi_scores.json"
    plot_path = output_dir / f"{model_name}_bi_scores.png"

    save_results_json(
        json_path=json_path,
        model_path=model_path,
        bi_scores=bi_scores,
        avg_cos_sims=avg_cos_sims,
        num_samples=len(samples),
        max_length=args.max_length,
        elapsed=elapsed,
    )
    save_plot(bi_scores, model_name, plot_path)

    # 9. 모델 메모리 해제
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n완료.")


if __name__ == "__main__":
    main()
