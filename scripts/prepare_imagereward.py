"""Parse ImageRewardDB (THUDM/ImageRewardDB) into a framework-ready pairwise-ranking CSV.

ImageRewardDB ships *per-image* human rankings (rank 1 = best) for each text prompt, not explicit
pairs — so this builds the pairs: within a prompt, every two images with different ranks form one
ordered comparison (smaller rank wins). Ties (equal rank) are skipped. The A/B side is randomized
(seeded) so the labels are ~50/50 instead of "A always wins".

Output row:

    image_a, image_b, prompt_id, margin_label, binary_label, rank_a, rank_b

- image_a / image_b : absolute paths to the extracted .webp images (view order matters; A is view 0).
- margin_label      : +1 if A is preferred (lower rank), -1 if B is  (MarginRankingCriterion target).
- binary_label      :  1 if A is preferred, 0 if B is               (BCE / logistic ranking losses).
- rank_a / rank_b   : the source human ranks (1 = best), for filtering / inspection.

Data comes straight from the HF repo files (datasets>=5 dropped the loading script): the metadata
parquet plus the per-chunk image zips. Only the zips referenced by the selected pairs are fetched.
Validation is the smallest split (~412 prompts; images ~1.1 GB across two zips). Apache-2.0.

Run (HF login not required):
    uv run python scripts/prepare_imagereward.py --no-images                 # CSV only, instant (no image download)
    uv run python scripts/prepare_imagereward.py --max-prompts 150           # quick test subset (+ images)
    uv run python scripts/prepare_imagereward.py --split train --adjacent-only
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_REPO = "THUDM/ImageRewardDB"
_FIELDS = ["image_a", "image_b", "prompt_id", "margin_label", "binary_label", "rank_a", "rank_b"]


def _load_metadata(split: str, max_prompts: int | None, chunk: str | None) -> pd.DataFrame:
    path = hf_hub_download(_REPO, f"metadata-{split}.parquet", repo_type="dataset")
    frame = pd.read_parquet(path, columns=["image_path", "prompt_id", "rank"])
    if chunk is not None:  # restrict to one image zip (e.g. "validation_2") — a single downloadable archive
        frame = frame[frame["image_path"].str.split("/").str[2] == chunk]
    if max_prompts is not None:
        keep = frame["prompt_id"].drop_duplicates().iloc[:max_prompts]
        frame = frame[frame["prompt_id"].isin(set(keep))]
    return frame


def _cap_by_images(records: list[dict], max_images: int) -> list[dict]:
    """Keep pairs greedily until the distinct-image budget is reached (for a small sample)."""
    seen: set[str] = set()
    kept: list[dict] = []
    for record in records:
        candidate = seen | {record["image_a"], record["image_b"]}
        if len(candidate) > max_images:
            continue
        kept.append(record)
        seen = candidate
    return kept


def _build_pairs(frame: pd.DataFrame, rng: random.Random, adjacent_only: bool) -> list[dict]:
    """One ordered comparison per cross-rank image pair within a prompt; A/B side randomized."""
    records: list[dict] = []
    for prompt_id, group in frame.groupby("prompt_id", sort=False):
        items = list(zip(group["image_path"], group["rank"].astype(int), strict=True))
        for (path_i, rank_i), (path_j, rank_j) in itertools.combinations(items, 2):
            if rank_i == rank_j or (adjacent_only and abs(rank_i - rank_j) != 1):
                continue
            better, worse = (
                ((path_i, rank_i), (path_j, rank_j)) if rank_i < rank_j else ((path_j, rank_j), (path_i, rank_i))
            )
            a, b, a_is_better = (better, worse, True) if rng.random() < 0.5 else (worse, better, False)
            records.append(
                {
                    "image_a": a[0],
                    "image_b": b[0],
                    "prompt_id": prompt_id,
                    "margin_label": 1 if a_is_better else -1,
                    "binary_label": 1 if a_is_better else 0,
                    "rank_a": a[1],
                    "rank_b": b[1],
                }
            )
    return records


def _ensure_images(image_paths: set[str], root: Path, split: str) -> None:
    """Download + extract only the image zips referenced by the selected pairs.

    A zip ``images/{split}/{chunk}.zip`` holds flat ``<uuid>.webp`` files; extracting it into
    ``root/images/{split}/{chunk}/`` makes ``root / image_path`` resolve for every row.
    """
    members_by_chunk: dict[str, set[str]] = {}
    for path in image_paths:
        members_by_chunk.setdefault(path.split("/")[2], set()).add(Path(path).name)  # zip members are flat <uuid>.webp
    for chunk, members in sorted(members_by_chunk.items()):
        target = root / "images" / split / chunk
        missing = [member for member in members if not (target / member).exists()]
        if not missing:
            print(f"  images/{split}/{chunk}: {len(members)} already present")
            continue
        print(f"  fetching images/{split}/{chunk}.zip → extracting {len(missing)} images …", flush=True)
        archive = hf_hub_download(_REPO, f"images/{split}/{chunk}.zip", repo_type="dataset")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            for member in missing:
                zipped.extract(member, target)


def _report(records: list[dict], root: Path, out: Path, fetched_images: bool) -> None:
    a_better = sum(record["margin_label"] == 1 for record in records)
    prompts = len({record["prompt_id"] for record in records})
    print(
        f"\nWrote {out} — {len(records)} pairs from {prompts} prompts (A>B: {a_better}, B>A: {len(records) - a_better})."
    )
    if not records:
        return
    first = records[0]["image_a"]
    if not fetched_images:
        print(f"  images not downloaded (--no-images): paths resolve after a run without the flag. e.g. {first}")
    elif not Path(first).exists():
        print(f"  WARNING: expected image missing, check extraction: {first}")
    else:
        print(f"  images verified under {root / 'images'}. e.g. {first}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse ImageRewardDB into a pairwise-ranking CSV.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument(
        "--out", default="data/imagereward/pairs.csv", help="output CSV (images go in <out-dir>/images)"
    )
    parser.add_argument("--max-prompts", type=int, help="keep only the first N prompts (quick test subset)")
    parser.add_argument("--max-images", type=int, help="cap the number of distinct images (small viewable sample)")
    parser.add_argument("--max-samples", type=int, help="cap the number of pairs (after shuffling)")
    parser.add_argument("--chunk", help="restrict to one image zip, e.g. validation_2 (sample from a single archive)")
    parser.add_argument("--adjacent-only", action="store_true", help="only pairs whose ranks differ by 1 (harder)")
    parser.add_argument("--no-images", action="store_true", help="write the CSV without downloading/extracting images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"ImageRewardDB ({_REPO}, Apache-2.0) — split={args.split}")
    rng = random.Random(args.seed)
    frame = _load_metadata(args.split, args.max_prompts, args.chunk)
    records = _build_pairs(frame, rng, args.adjacent_only)
    rng.shuffle(records)
    if args.max_images is not None:
        records = _cap_by_images(records, args.max_images)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    out = Path(args.out)
    root = out.parent
    root.mkdir(parents=True, exist_ok=True)
    if not args.no_images and records:
        _ensure_images(
            {record["image_a"] for record in records} | {record["image_b"] for record in records}, root, args.split
        )

    root_abs = root.resolve()
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["image_a"] = str(root_abs / record["image_a"])
            row["image_b"] = str(root_abs / record["image_b"])
            writer.writerow(row)

    _report(
        [{**record, "image_a": str(root_abs / record["image_a"])} for record in records],
        root_abs,
        out,
        fetched_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
