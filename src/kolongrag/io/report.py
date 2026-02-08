from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..eval.types import EvalRecord


def now_run_name(model_path: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).name or "model"
    return f"{ts}_{model_name}"


def to_nullable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def write_csv_artifacts(
    records: list[EvalRecord],
    results_dir: Path,
    output_dir: Path,
    run_name: str,
) -> dict[str, str]:
    import pandas as pd

    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_csv = results_dir / f"{run_name}.csv"
    submission_csv = output_dir / f"{run_name}_submission.csv"
    canonical_submission_csv = output_dir / "submission.csv"

    result_df = pd.DataFrame(
        [
            {
                "idx": r.idx,
                "id": r.sample_id,
                "task": r.task,
                "question": r.question,
                "reference": r.reference,
                "prediction": r.prediction,
                "short_ans": r.short_ans,
                "judge_long": "" if r.judge_long is None else r.judge_long,
                "judge_short": "" if r.judge_short is None else r.judge_short,
                "is_correct": "" if r.is_correct is None else r.is_correct,
                "output_tokens": r.output_tokens,
            }
            for r in records
        ]
    )
    result_df.to_csv(result_csv, index=False)

    submission_df = pd.DataFrame({"id": [r.sample_id for r in records], "prediction": [r.prediction for r in records]})
    submission_df.to_csv(submission_csv, index=False)
    submission_df.to_csv(canonical_submission_csv, index=False)

    return {
        "result_csv": str(result_csv),
        "submission_csv": str(submission_csv),
        "canonical_submission_csv": str(canonical_submission_csv),
    }


def write_json_payload(results_dir: Path, run_name: str, payload: dict[str, Any]) -> str:
    result_json = results_dir / f"{run_name}.json"
    with result_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return str(result_json)
