#!/usr/bin/env python3
"""
레이어 프루닝 스크립트 (ShortGPT 방식)

BI Score 분석 결과를 참고하여 지정한 디코더 레이어를 물리적으로 제거한 모델을 저장한다.
레이어 번호는 analyze/run_bi_score.py 기준 (model.model.layers[i])과 동일하다.

캘리브레이션 데이터 불필요 — 단순히 레이어를 삭제하고 config를 갱신하여 저장한다.

사용법:
  # 인터랙티브 모드
  python experiments/prune_layers.py

  # CLI 모드
  python experiments/prune_layers.py -m models/EXAONE-4.0-1.2B --layers 26,27,28

  # BI Score JSON 참고하여 하위 N개 자동 선택
  python experiments/prune_layers.py -m models/EXAONE-4.0-1.2B --auto-top 5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS_DIR = Path("models")
RESULTS_DIR = Path("analyze/results")


# ═══════════════════════════════════════════════════════════
#  모델 선택
# ═══════════════════════════════════════════════════════════

def discover_models(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(
        child for child in models_dir.iterdir()
        if child.is_dir() and (child / "config.json").exists()
    )


def select_model_interactive(models_dir: Path) -> Path:
    models = discover_models(models_dir)
    if not models:
        print(f"[ERROR] '{models_dir}' 디렉토리에 모델이 없습니다.")
        sys.exit(1)

    print("=" * 50)
    print("사용 가능한 모델:")
    print("=" * 50)
    for idx, model_path in enumerate(models, 1):
        size_bytes = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
        size_gb = size_bytes / 1e9 if size_bytes > 0 else 0
        size_str = f" ({size_gb:.2f} GB)" if size_gb > 0 else ""
        print(f"  [{idx}] {model_path.name}{size_str}")
    print("=" * 50)

    while True:
        try:
            choice = input(f"모델 번호를 선택하세요 (1-{len(models)}): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print(f"1부터 {len(models)} 사이의 번호를 입력하세요.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")
        except (KeyboardInterrupt, EOFError):
            print("\n취소되었습니다.")
            sys.exit(0)


def resolve_model_path(model_arg: str | None) -> Path:
    if model_arg is not None:
        p = Path(model_arg)
        if not p.exists():
            print(f"[ERROR] 모델 경로가 존재하지 않습니다: {p}")
            sys.exit(1)
        return p
    return select_model_interactive(MODELS_DIR)


# ═══════════════════════════════════════════════════════════
#  BI Score 로드 및 표시
# ═══════════════════════════════════════════════════════════

def load_bi_scores(model_name: str) -> dict | None:
    """analyze/results/{model_name}_bi_scores.json 을 로드한다."""
    json_path = RESULTS_DIR / f"{model_name}_bi_scores.json"
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def display_bi_scores(bi_data: dict) -> list[dict]:
    """BI Score를 테이블로 출력하고 레이어 정보를 반환한다."""
    layers = bi_data["layers"]
    sorted_by_bi = sorted(layers, key=lambda x: x["bi_score"])
    candidates = bi_data["summary"]["pruning_candidates"]

    bar_width = 35
    max_bi = max(l["bi_score"] for l in layers)

    print()
    print("=" * 72)
    print(f"{'Layer':>7} | {'BI Score':>10} | {'Cos Sim':>10} | {'':^6} | Bar")
    print("-" * 72)

    for l in layers:
        bar_len = int((l["bi_score"] / max_bi) * bar_width) if max_bi > 0 else 0
        bar = "\u2588" * bar_len
        tag = " [LOW]" if l["layer"] in candidates else ""
        print(f"  {l['layer']:>4}  | {l['bi_score']:>10.6f} | {l['cos_sim']:>10.6f} |{tag:>6} | {bar}")

    print("=" * 72)
    print(f"  평균 BI: {bi_data['summary']['avg_bi_score']:.6f}")
    print(f"  최소 BI: {bi_data['summary']['min_bi_score']:.6f} (Layer {bi_data['summary']['min_bi_layer']})")
    print(f"  최대 BI: {bi_data['summary']['max_bi_score']:.6f} (Layer {bi_data['summary']['max_bi_layer']})")
    print(f"  Pruning 후보 (하위 25%): {candidates}")
    print()

    return layers


# ═══════════════════════════════════════════════════════════
#  인터랙티브 레이어 선택
# ═══════════════════════════════════════════════════════════

def select_layers_interactive(num_layers: int, candidates: list[int] | None) -> list[int]:
    """제거할 레이어 번호를 인터랙티브로 입력받는다."""
    print("=" * 50)
    print("제거할 레이어를 선택하세요")
    print("=" * 50)
    print(f"  전체 레이어: 0 ~ {num_layers - 1} (총 {num_layers}개)")
    if candidates:
        print(f"  BI Score 기반 추천 (하위 25%): {candidates}")
    print()
    print("입력 형식:")
    print("  - 콤마 구분:  26,27,28")
    print("  - 범위 지정:  25-28")
    print("  - 혼합:       16,17,25-28")
    print("=" * 50)

    while True:
        try:
            raw = input("제거할 레이어 번호: ").strip()
            if not raw:
                print("하나 이상의 레이어를 입력하세요.")
                continue

            layers_to_remove = parse_layer_spec(raw, num_layers)
            if not layers_to_remove:
                print("유효한 레이어 번호가 없습니다.")
                continue

            remaining = num_layers - len(layers_to_remove)
            if remaining < 1:
                print(f"[ERROR] 최소 1개 레이어는 남겨야 합니다 (현재 제거: {len(layers_to_remove)}개)")
                continue

            print(f"\n제거할 레이어: {sorted(layers_to_remove)} ({len(layers_to_remove)}개)")
            print(f"남는 레이어: {remaining}개")

            confirm = input("진행하시겠습니까? (y/N): ").strip().lower()
            if confirm in ("y", "yes"):
                return sorted(layers_to_remove)
            print("다시 입력하세요.\n")

        except (KeyboardInterrupt, EOFError):
            print("\n취소되었습니다.")
            sys.exit(0)


def parse_layer_spec(spec: str, num_layers: int) -> list[int]:
    """'1,2,5-8,10' 형태의 문자열을 레이어 번호 리스트로 파싱한다."""
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            tokens = part.split("-", 1)
            try:
                start, end = int(tokens[0].strip()), int(tokens[1].strip())
                for i in range(start, end + 1):
                    if 0 <= i < num_layers:
                        result.add(i)
                    else:
                        print(f"  [경고] 레이어 {i}은(는) 범위 밖입니다 (0-{num_layers - 1})")
            except ValueError:
                print(f"  [경고] 파싱 실패: '{part}'")
        else:
            try:
                i = int(part)
                if 0 <= i < num_layers:
                    result.add(i)
                else:
                    print(f"  [경고] 레이어 {i}은(는) 범위 밖입니다 (0-{num_layers - 1})")
            except ValueError:
                print(f"  [경고] 파싱 실패: '{part}'")
    return sorted(result)


# ═══════════════════════════════════════════════════════════
#  레이어 프루닝
# ═══════════════════════════════════════════════════════════

def get_decoder_layers_module(model):
    """모델에서 디코더 레이어 nn.ModuleList를 반환한다."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, "layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer, "h"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox, "layers"
    raise ValueError(f"지원하지 않는 모델 아키텍처: {type(model).__name__}")


def prune_layers(model, layers_to_remove: list[int]) -> int:
    """지정한 레이어를 모델에서 제거한다. 남은 레이어 수를 반환한다."""
    parent, attr_name = get_decoder_layers_module(model)
    original_layers = getattr(parent, attr_name)
    num_original = len(original_layers)

    remove_set = set(layers_to_remove)
    kept = torch.nn.ModuleList(
        layer for idx, layer in enumerate(original_layers)
        if idx not in remove_set
    )

    setattr(parent, attr_name, kept)

    # config 업데이트
    model.config.num_hidden_layers = len(kept)

    print(f"레이어 프루닝 완료: {num_original} → {len(kept)} ({len(layers_to_remove)}개 제거)")
    return len(kept)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="레이어 프루닝 (ShortGPT 방식)"
    )
    parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="모델 경로 (미지정시 인터랙티브 선택)",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="제거할 레이어 (예: '26,27,28' 또는 '25-28'). 미지정시 인터랙티브 선택",
    )
    parser.add_argument(
        "--auto-top", type=int, default=None,
        help="BI Score 하위 N개 레이어를 자동 선택하여 제거",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="출력 모델 경로 (기본값: {model}-pruned)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="모델 로드 dtype (기본값: bfloat16)",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # 1. 모델 경로 확정
    model_path = resolve_model_path(args.model)
    model_name = model_path.name
    print(f"\n모델: {model_path}")

    # 2. BI Score 로드 (있으면 참고용으로 표시)
    bi_data = load_bi_scores(model_name)
    if bi_data:
        print(f"BI Score 데이터 발견: {RESULTS_DIR / f'{model_name}_bi_scores.json'}")
        display_bi_scores(bi_data)
        num_layers = bi_data["summary"]["num_layers"]
        candidates = bi_data["summary"]["pruning_candidates"]
    else:
        print("BI Score 데이터 없음 (analyze/run_bi_score.py 를 먼저 실행하면 추천을 볼 수 있습니다)")
        # config.json에서 레이어 수 읽기
        config_path = model_path / "config.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        num_layers = config.get("num_hidden_layers", 0)
        if num_layers == 0:
            print("[ERROR] config.json에서 num_hidden_layers를 찾을 수 없습니다.")
            sys.exit(1)
        print(f"전체 레이어 수: {num_layers}")
        candidates = None

    # 3. 제거할 레이어 결정
    if args.auto_top is not None:
        if bi_data is None:
            print("[ERROR] --auto-top 사용시 BI Score JSON이 필요합니다.")
            print(f"  먼저 실행: python analyze/run_bi_score.py -m {model_path}")
            sys.exit(1)
        sorted_layers = sorted(bi_data["layers"], key=lambda x: x["bi_score"])
        layers_to_remove = [l["layer"] for l in sorted_layers[:args.auto_top]]
        print(f"BI Score 하위 {args.auto_top}개 자동 선택: {layers_to_remove}")
    elif args.layers is not None:
        layers_to_remove = parse_layer_spec(args.layers, num_layers)
        if not layers_to_remove:
            print("[ERROR] 유효한 레이어 번호가 없습니다.")
            sys.exit(1)
    else:
        layers_to_remove = select_layers_interactive(num_layers, candidates)

    remaining = num_layers - len(layers_to_remove)
    if remaining < 1:
        print(f"[ERROR] 최소 1개 레이어는 남겨야 합니다.")
        sys.exit(1)

    print(f"\n제거 대상: {layers_to_remove} ({len(layers_to_remove)}개)")
    print(f"프루닝 후 레이어: {remaining}개")

    # 4. 출력 경로
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"pruned-{remaining}L"
        output_path = model_path.parent / f"{model_name}-{suffix}"

    print(f"출력 경로: {output_path}")

    # 5. 모델 로드
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"\n모델 로드 중: {model_path} (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map="cpu",  # 프루닝은 CPU에서 수행
        trust_remote_code=True,
    )

    # 6. 레이어 프루닝
    prune_layers(model, layers_to_remove)

    # 7. 저장
    print(f"\n프루닝된 모델 저장 중: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # 토크나이저도 복사
    print("토크나이저 저장 중...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # 프루닝 메타 정보 저장
    meta = {
        "source_model": str(model_path),
        "source_layers": num_layers,
        "removed_layers": layers_to_remove,
        "remaining_layers": remaining,
        "dtype": args.dtype,
    }
    if bi_data:
        meta["removed_layers_bi_scores"] = {
            l["layer"]: l["bi_score"]
            for l in bi_data["layers"]
            if l["layer"] in layers_to_remove
        }

    meta_path = output_path / "pruning_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 크기 비교
    orig_size = sum(f.stat().st_size for f in model_path.glob("*.safetensors"))
    new_size = sum(f.stat().st_size for f in output_path.glob("*.safetensors"))
    if orig_size > 0 and new_size > 0:
        print(f"\n원본 크기: {orig_size / 1e9:.2f} GB")
        print(f"프루닝 크기: {new_size / 1e9:.2f} GB")
        print(f"축소율: {(1 - new_size / orig_size) * 100:.1f}%")

    # 메모리 해제
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n완료. 프루닝된 모델: {output_path}")


if __name__ == "__main__":
    main()
