from __future__ import annotations

from typing import Any


def load_samples(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def filter_samples(
    samples: list[dict[str, Any]],
    quick: int | None,
    single_only: bool,
    multi_only: bool,
) -> list[dict[str, Any]]:
    if quick is not None:
        singles = [s for s in samples if s.get("task") == "singledocQA"]
        singles.sort(key=lambda x: str(x.get("id", "")))
        return singles[:quick]
    if single_only:
        return [s for s in samples if s.get("task") == "singledocQA"]
    if multi_only:
        return [s for s in samples if s.get("task") == "multidocQA"]
    return samples

