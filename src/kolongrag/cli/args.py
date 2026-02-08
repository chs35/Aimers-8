from __future__ import annotations

import argparse

from ..config.options import DEFAULT_PROMPT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ko-LongRAG 기반 평가 서버 MVP (README 조건 반영)."
    )
    parser.add_argument(
        "--model-path",
        default="EXAONE-4.0-1.2B",
        help="models/ 하위 모델 디렉토리명",
    )
    parser.add_argument(
        "--prompt-config",
        default=str(DEFAULT_PROMPT_CONFIG),
        help="프롬프트 설정 YAML 파일 경로.",
    )
    parser.add_argument(
        "--quick",
        nargs="?",
        const=100,
        type=int,
        help="singledocQA 앞 N개만 실행. 숫자 생략 시 100.",
    )
    parser.add_argument("--single-only", action="store_true", help="singledocQA만 평가.")
    parser.add_argument("--multi-only", action="store_true", help="multidocQA만 평가.")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="정답 채점 단계를 건너뛰고 추론/속도 지표만 기록.",
    )

    args = parser.parse_args()

    if args.single_only and args.multi_only:
        parser.error("--single-only와 --multi-only는 동시에 사용할 수 없습니다.")
    if args.quick is not None and args.multi_only:
        parser.error("--quick은 singledocQA 전용이므로 --multi-only와 함께 사용할 수 없습니다.")
    if args.quick is not None and args.quick <= 0:
        parser.error("--quick 값은 1 이상의 정수여야 합니다.")
    return args
