"""CLI 진입점 (python -m benchmark_hf 지원)"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# .env 파일에서 환경변수 로드 (python-dotenv 불필요)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from .runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Ko-LongRAG 벤치마크 (vLLM + GPT-4o judge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python script.py -m models/EXAONE-4.0-1.2B
  python script.py -m models/EXAONE-4.0-1.2B --subset singledocQA
  python script.py -m models/EXAONE-4.0-1.2B --max-samples 50
  python script.py -m models/EXAONE-4.0-1.2B --quick
  python script.py -m models/EXAONE-4.0-1.2B --quick 50
""",
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="모델 경로 (예: models/EXAONE-4.0-1.2B)",
    )
    parser.add_argument(
        "--subset", "-s", default=None,
        help="singledocQA / multidocQA (미지정 시 전체)",
    )
    parser.add_argument(
        "--split", default=None, help="데이터셋 split (기본: test)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="최대 샘플 수 (빠른 테스트용)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="생성 최대 토큰 수 (기본: 16384)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="샘플링 temperature (기본: 0.0, greedy)",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--thinking", "-t", action="store_true",
        help="enable_thinking 활성화",
    )
    parser.add_argument(
        "--quick", nargs="?", type=int, const=100, default=None, metavar="N",
        help="singledocQA 00~N-1번 평가 (미지정 시 100개, 빠른 검증용)",
    )
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        dataset_id="LGAI-EXAONE/Ko-LongRAG",
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        enable_thinking=args.thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        quick=args.quick if args.quick is not None else 0,
    )


if __name__ == "__main__":
    main()
