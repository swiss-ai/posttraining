from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from batch import keys as K


class OfflineRewardChunkStore:
    """Resumable flat-row store for offline reward recomputation."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)

    def chunk_path(self, chunk_index: int) -> Path:
        if int(chunk_index) < 0:
            raise ValueError(f"chunk_index must be non-negative, got {chunk_index}.")
        return self.chunks_dir / f"chunk_{int(chunk_index):06d}.json"

    def save_chunk(
        self,
        *,
        chunk_index: int,
        rows: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        overwrite: bool = False,
    ) -> None:
        path = self.chunk_path(chunk_index)
        if path.exists() and not overwrite:
            self.load_chunk(chunk_index=chunk_index, manifest=manifest)
            return

        payload = {
            "manifest": dict(manifest),
            "rows": [dict(row) for row in rows],
        }
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}.json")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def load_chunk(
        self,
        *,
        chunk_index: int,
        manifest: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        path = self.chunk_path(chunk_index)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        stored_manifest = payload.get("manifest", {})
        for key, expected_value in manifest.items():
            if stored_manifest.get(key) != expected_value:
                raise ValueError(
                    f"Offline reward chunk {path} was produced with different "
                    "metadata. Use a new work_dir or remove stale chunks."
                )

        rows = payload.get("rows", None)
        if not isinstance(rows, list):
            raise ValueError(f"Offline reward chunk {path} has no row list.")

        return [dict(row) for row in rows]

    def load_existing_chunks(
        self,
        *,
        manifest: Mapping[str, Any],
    ) -> dict[int, list[dict[str, Any]]]:
        chunks: dict[int, list[dict[str, Any]]] = {}
        for path in sorted(self.chunks_dir.glob("chunk_*.json")):
            chunk_index = int(path.stem.removeprefix("chunk_"))
            chunks[chunk_index] = self.load_chunk(
                chunk_index=chunk_index,
                manifest=manifest,
            )
        return chunks


def apply_offline_reward_rows_to_dataset(
    *,
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    offline_trajectories_key: str = "offline_trajectories",
    offline_rewards_key: str = "offline_rewards",
    reward_extra_info_key: str | None = "offline_reward_extra_info",
    require_all: bool = True,
) -> Any:
    """Return a dataset with offline reward columns replaced from flat rows."""

    grouped = _group_rows_by_dataset_index(rows)

    if require_all:
        missing_dataset_indices = [
            index for index in range(len(dataset))
            if (
                index not in grouped
                and len(dataset[int(index)][offline_trajectories_key]) > 0
            )
        ]
        if missing_dataset_indices:
            preview = missing_dataset_indices[:10]
            raise ValueError(
                "Offline reward rows do not cover all dataset rows. "
                f"Missing indices start with {preview}."
            )

    replacements: dict[int, dict[str, list[Any]]] = {}
    for dataset_index, indexed_rows in grouped.items():
        row = dataset[int(dataset_index)]
        trajectory_count = len(row[offline_trajectories_key])
        rewards: list[float | None] = [None] * trajectory_count
        extra_infos: list[Any] = [None] * trajectory_count

        for offline_index, reward_row in indexed_rows.items():
            if offline_index < 0 or offline_index >= trajectory_count:
                raise ValueError(
                    f"Offline reward row for dataset_index={dataset_index} has "
                    f"offline_completion_index={offline_index}, but row has "
                    f"{trajectory_count} offline trajectories."
                )
            rewards[offline_index] = float(reward_row["reward"])
            extra_infos[offline_index] = reward_row.get("reward_extra_info")

        if require_all and any(reward is None for reward in rewards):
            missing = [
                index for index, reward in enumerate(rewards)
                if reward is None
            ]
            raise ValueError(
                f"Offline reward rows for dataset_index={dataset_index} are "
                f"missing offline completion indices {missing}."
            )

        existing_rewards = row.get(offline_rewards_key, [0.0] * trajectory_count)
        merged_rewards = [
            float(existing_rewards[index]) if reward is None else float(reward)
            for index, reward in enumerate(rewards)
        ]

        replacement = {offline_rewards_key: merged_rewards}
        if reward_extra_info_key is not None:
            existing_extra_infos = row.get(
                reward_extra_info_key,
                [None] * trajectory_count,
            )
            replacement[reward_extra_info_key] = [
                existing_extra_infos[index] if extra_info is None else extra_info
                for index, extra_info in enumerate(extra_infos)
            ]
        replacements[int(dataset_index)] = replacement

    def update_row(example: dict[str, Any], index: int) -> dict[str, Any]:
        replacement = replacements.get(int(index))
        if replacement is None:
            return example
        updated = dict(example)
        updated.update(replacement)
        return updated

    return dataset.map(update_row, with_indices=True)


def _group_rows_by_dataset_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[int, Mapping[str, Any]]]:
    grouped: dict[int, dict[int, Mapping[str, Any]]] = {}

    for row in rows:
        dataset_index = int(row[K.DATASET_INDEX])
        offline_index = int(row[K.OFFLINE_COMPLETION_INDEX])
        indexed_rows = grouped.setdefault(dataset_index, {})
        if offline_index in indexed_rows:
            raise ValueError(
                f"Duplicate offline reward row for dataset_index={dataset_index}, "
                f"offline_completion_index={offline_index}."
            )
        indexed_rows[offline_index] = row

    return grouped
