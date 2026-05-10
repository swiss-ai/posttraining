from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from batch import keys as K
from data.schemas import PromptRecord


class RefRewardStore:
    """Append-only store for versioned QRPO reference rewards.

    Each version is saved as one HF Dataset row per prompt:

      prompt_id
      dataset_index
      ref_version
      ref_completions
      ref_rewards
      reward_extra_info

    Training should resolve PromptRecord.ref_rewards from this store instead of
    treating dataset ref_rewards as authoritative.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.in_progress_dir = self.root / "in_progress"
        self.model_checkpoints_dir = self.root / "model_checkpoints"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.in_progress_dir.mkdir(parents=True, exist_ok=True)
        self.model_checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, tuple[float, ...]]] = {}

    def version_dir(self, ref_version: str) -> Path:
        return self.versions_dir / _safe_version_name(ref_version)

    def manifest_path(self, ref_version: str) -> Path:
        return self.version_dir(ref_version) / "manifest.json"

    def dataset_path(self, ref_version: str) -> Path:
        return self.version_dir(ref_version) / "dataset"

    def model_checkpoint_path(self, ref_version: str) -> Path:
        # Keep model checkpoints outside versions/. save_version_rows replaces
        # version directories atomically and must not delete completed weights.
        return self.model_checkpoints_dir / _safe_version_name(ref_version)

    def generation_chunk_dir(self, ref_version: str) -> Path:
        return self.in_progress_dir / _safe_version_name(ref_version) / "chunks"

    def generation_chunk_path(self, ref_version: str, chunk_index: int) -> Path:
        if int(chunk_index) < 0:
            raise ValueError(f"chunk_index must be non-negative, got {chunk_index}.")
        return (
            self.generation_chunk_dir(ref_version)
            / f"chunk_{int(chunk_index):06d}.json"
        )

    def has_complete_version(self, ref_version: str) -> bool:
        manifest = self._read_manifest_if_exists(ref_version)
        return bool(manifest and manifest.get("status") == "complete")

    def load_manifest(self, ref_version: str) -> dict[str, Any]:
        manifest = self._read_manifest_if_exists(ref_version)
        if not manifest or manifest.get("status") != "complete":
            raise FileNotFoundError(
                f"Ref reward version {ref_version!r} is missing or incomplete."
            )
        return manifest

    def check_complete_version_metadata(
        self,
        *,
        ref_version: str,
        metadata: Mapping[str, Any],
    ) -> None:
        manifest = self._read_manifest_if_exists(ref_version)
        if not manifest or manifest.get("status") != "complete":
            raise FileNotFoundError(
                f"Ref reward version {ref_version!r} is missing or incomplete."
            )

        for key, expected_value in metadata.items():
            if expected_value is None:
                continue
            if key not in manifest:
                raise ValueError(
                    f"Ref reward version {ref_version!r} manifest is missing "
                    f"expected key {key!r}."
                )
            if manifest[key] != expected_value:
                raise ValueError(
                    f"Ref reward version {ref_version!r} has manifest "
                    f"{key}={manifest[key]!r}, expected {expected_value!r}."
                )

    def import_dataset_ref_rewards(
        self,
        *,
        dataset: Any,
        dataset_config: Mapping[str, Any],
        ref_version: str,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Import existing dataset ref_rewards into a store version."""

        if self.has_complete_version(ref_version) and not overwrite:
            self._check_existing_dataset_import(
                ref_version=ref_version,
                dataset=dataset,
                metadata=metadata,
            )
            return

        prompt_id_key = str(dataset_config.get("prompt_id_key", "prompt_id"))
        ref_rewards_key = str(dataset_config.get("ref_rewards_key", "ref_rewards"))
        ref_completions_key = dataset_config.get("ref_completions_key", None)

        rows = []
        seen_prompt_ids: set[str] = set()
        for dataset_index in range(len(dataset)):
            row = dataset[int(dataset_index)]
            prompt_id = str(_require(row, prompt_id_key, dataset_index=dataset_index))
            if prompt_id in seen_prompt_ids:
                raise ValueError(
                    f"Dataset has duplicate prompt_id={prompt_id!r}; "
                    "RefRewardStore requires unique prompt ids."
                )
            seen_prompt_ids.add(prompt_id)

            ref_rewards = _float_list(
                _require(row, ref_rewards_key, dataset_index=dataset_index),
                field_name=ref_rewards_key,
                prompt_id=prompt_id,
            )
            ref_completions = _optional_list(
                row.get(ref_completions_key)
                if ref_completions_key is not None
                else None,
                length=len(ref_rewards),
            )

            rows.append(
                {
                    "prompt_id": prompt_id,
                    "dataset_index": int(dataset_index),
                    "ref_version": str(ref_version),
                    "ref_completions": ref_completions,
                    "ref_rewards": ref_rewards,
                    "reward_extra_info": [None] * len(ref_rewards),
                }
            )

        manifest = {
            "ref_version": str(ref_version),
            "status": "complete",
            "source": "dataset",
            "prompt_count": len(rows),
            "num_ref_completions": _shared_ref_count(rows),
        }
        if metadata is not None:
            manifest.update(dict(metadata))

        self.save_version_rows(
            ref_version=ref_version,
            rows=rows,
            manifest=manifest,
            overwrite=overwrite,
        )
        self._check_existing_dataset_import(
            ref_version=ref_version,
            dataset=dataset,
            metadata=metadata,
        )

    def save_version_rows(
        self,
        *,
        ref_version: str,
        rows: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        overwrite: bool = False,
    ) -> None:
        if self.has_complete_version(ref_version) and not overwrite:
            return

        rows = list(rows)
        _validate_rows(rows, ref_version=ref_version)

        version_dir = self.version_dir(ref_version)
        tmp_dir = version_dir.with_name(
            f"{version_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )

        if version_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Ref reward version already exists: {version_dir}")
            shutil.rmtree(version_dir)

        try:
            tmp_dir.mkdir(parents=True)

            from datasets import Dataset

            Dataset.from_list([dict(row) for row in rows]).save_to_disk(
                str(tmp_dir / "dataset")
            )

            full_manifest = dict(manifest)
            full_manifest["ref_version"] = str(ref_version)
            full_manifest["status"] = "complete"
            full_manifest.setdefault("prompt_count", len(rows))
            full_manifest.setdefault("num_ref_completions", _shared_ref_count(rows))

            with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(full_manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")

            try:
                tmp_dir.rename(version_dir)
            except OSError as exc:
                if not overwrite and self.has_complete_version(ref_version):
                    return
                if not overwrite and version_dir.exists():
                    raise FileExistsError(
                        "Ref reward version directory already exists but is not a "
                        f"complete reusable version: {version_dir}. Remove it, "
                        "choose a new ref_version, or set overwrite=True."
                    ) from exc
                raise
            finally:
                self._cache.pop(str(ref_version), None)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def save_generation_chunk(
        self,
        *,
        ref_version: str,
        chunk_index: int,
        dataset_indices: Sequence[int],
        rows: Sequence[Mapping[str, Any]],
        manifest: Mapping[str, Any],
        overwrite: bool = False,
    ) -> None:
        """Persist one generated ref-reward chunk for resumable generation."""

        path = self.generation_chunk_path(ref_version, chunk_index)
        if path.exists() and not overwrite:
            self.load_generation_chunk(
                ref_version=ref_version,
                chunk_index=chunk_index,
                dataset_indices=dataset_indices,
                manifest=manifest,
            )
            return

        rows = [dict(row) for row in rows]
        dataset_indices = [int(index) for index in dataset_indices]
        _validate_rows(rows, ref_version=ref_version)
        _check_rows_cover_dataset_indices(rows, dataset_indices=dataset_indices)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ref_version": str(ref_version),
            "chunk_index": int(chunk_index),
            "dataset_indices": dataset_indices,
            "manifest": dict(manifest),
            "rows": rows,
        }

        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)

    def load_generation_chunk(
        self,
        *,
        ref_version: str,
        chunk_index: int,
        dataset_indices: Sequence[int],
        manifest: Mapping[str, Any],
    ) -> list[dict[str, Any]] | None:
        """Load a matching in-progress generated chunk, if present."""

        path = self.generation_chunk_path(ref_version, chunk_index)
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        expected_dataset_indices = [int(index) for index in dataset_indices]
        expected_manifest = dict(manifest)

        if payload.get("ref_version") != str(ref_version):
            raise ValueError(
                f"Ref reward generation chunk {path} has ref_version="
                f"{payload.get('ref_version')!r}, expected {ref_version!r}."
            )
        if int(payload.get("chunk_index", -1)) != int(chunk_index):
            raise ValueError(
                f"Ref reward generation chunk {path} has chunk_index="
                f"{payload.get('chunk_index')!r}, expected {chunk_index!r}."
            )
        if payload.get("dataset_indices") != expected_dataset_indices:
            raise ValueError(
                f"Ref reward generation chunk {path} has dataset_indices="
                f"{payload.get('dataset_indices')!r}, expected "
                f"{expected_dataset_indices!r}."
            )
        if payload.get("manifest") != expected_manifest:
            raise ValueError(
                f"Ref reward generation chunk {path} was produced with different "
                "generation metadata. Use a new ref_version or remove the stale "
                "in-progress chunk."
            )

        rows = [dict(row) for row in payload.get("rows", [])]
        _validate_rows(rows, ref_version=ref_version)
        _check_rows_cover_dataset_indices(
            rows,
            dataset_indices=expected_dataset_indices,
        )
        return rows

    def load_ref_rewards(self, ref_version: str) -> dict[str, tuple[float, ...]]:
        if ref_version in self._cache:
            return self._cache[ref_version]

        manifest = self._read_manifest_if_exists(ref_version)
        if not manifest or manifest.get("status") != "complete":
            raise FileNotFoundError(
                f"Ref reward version {ref_version!r} is missing or incomplete."
            )

        from datasets import load_from_disk

        dataset = load_from_disk(str(self.dataset_path(ref_version)))
        ref_rewards_by_prompt_id: dict[str, tuple[float, ...]] = {}

        for row in dataset:
            prompt_id = str(row["prompt_id"])
            if prompt_id in ref_rewards_by_prompt_id:
                raise ValueError(
                    f"Ref reward version {ref_version!r} has duplicate "
                    f"prompt_id={prompt_id!r}."
                )
            ref_rewards_by_prompt_id[prompt_id] = tuple(
                _float_list(
                    row["ref_rewards"],
                    field_name="ref_rewards",
                    prompt_id=prompt_id,
                )
            )

        self._cache[ref_version] = ref_rewards_by_prompt_id
        return ref_rewards_by_prompt_id

    def attach_ref_rewards(
        self,
        prompt_records: Sequence[PromptRecord],
        *,
        ref_version: str,
    ) -> list[PromptRecord]:
        ref_rewards_by_prompt_id = self.load_ref_rewards(ref_version)
        resolved = []

        for prompt in prompt_records:
            if prompt.prompt_id not in ref_rewards_by_prompt_id:
                raise KeyError(
                    f"Ref reward version {ref_version!r} has no rewards for "
                    f"prompt_id={prompt.prompt_id!r}."
                )

            resolved.append(
                replace(
                    prompt,
                    ref_rewards=ref_rewards_by_prompt_id[prompt.prompt_id],
                    metadata={
                        **prompt.metadata,
                        K.REF_VERSION: ref_version,
                    },
                )
            )

        return resolved

    def _read_manifest_if_exists(self, ref_version: str) -> dict[str, Any] | None:
        path = self.manifest_path(ref_version)
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _check_existing_dataset_import(
        self,
        *,
        ref_version: str,
        dataset: Any,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        manifest = self._read_manifest_if_exists(ref_version)
        if manifest is None:
            raise FileNotFoundError(
                f"Ref reward version {ref_version!r} is missing a manifest."
            )

        prompt_count = manifest.get("prompt_count")
        if prompt_count != len(dataset):
            raise ValueError(
                f"Existing ref reward version {ref_version!r} has prompt_count="
                f"{prompt_count}, expected {len(dataset)} for the current dataset."
            )

        if metadata is None:
            return

        self.check_complete_version_metadata(
            ref_version=ref_version,
            metadata=metadata,
        )


def _safe_version_name(ref_version: str) -> str:
    ref_version = str(ref_version)
    if not ref_version:
        raise ValueError("ref_version must be non-empty.")
    if "/" in ref_version or "\\" in ref_version:
        raise ValueError(f"ref_version must not contain path separators: {ref_version!r}.")
    return ref_version


def _require(row: Mapping[str, Any], key: str, *, dataset_index: int) -> Any:
    if key not in row:
        raise KeyError(f"Dataset row {dataset_index} is missing required key {key!r}.")
    return row[key]


def _optional_list(value: Any, *, length: int) -> list[Any]:
    if value is None:
        return [None] * length

    values = _list_like(value)
    if len(values) != length:
        raise ValueError(
            f"ref_completions has length {len(values)}, expected {length}."
        )
    return values


def _float_list(value: Any, *, field_name: str, prompt_id: str) -> list[float]:
    values = _list_like(value)
    if not values:
        raise ValueError(f"Prompt {prompt_id!r} has empty {field_name}.")

    floats = [float(item) for item in values]
    for reward in floats:
        if not math.isfinite(reward):
            raise ValueError(
                f"Prompt {prompt_id!r} has non-finite {field_name}: {reward!r}."
            )

    return floats


def _list_like(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    raise TypeError(f"Expected list-like value, got {type(value).__name__}.")


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, ref_version: str) -> None:
    if not rows:
        raise ValueError("Cannot save an empty ref reward version.")

    seen_prompt_ids: set[str] = set()
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"Duplicate prompt_id in ref reward version: {prompt_id!r}.")
        seen_prompt_ids.add(prompt_id)

        if str(row["ref_version"]) != str(ref_version):
            raise ValueError(
                f"Row for prompt_id={prompt_id!r} has ref_version="
                f"{row['ref_version']!r}, expected {ref_version!r}."
            )

        ref_rewards = _float_list(
            row["ref_rewards"],
            field_name="ref_rewards",
            prompt_id=prompt_id,
        )
        ref_completions = _list_like(row["ref_completions"])
        reward_extra_info = _list_like(row["reward_extra_info"])
        if len(ref_completions) != len(ref_rewards):
            raise ValueError(
                f"Prompt {prompt_id!r} has {len(ref_completions)} ref_completions "
                f"but {len(ref_rewards)} ref_rewards."
            )
        if len(reward_extra_info) != len(ref_rewards):
            raise ValueError(
                f"Prompt {prompt_id!r} has {len(reward_extra_info)} reward_extra_info "
                f"items but {len(ref_rewards)} ref_rewards."
            )


def _check_rows_cover_dataset_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_indices: Sequence[int],
) -> None:
    expected = [int(index) for index in dataset_indices]
    observed = [int(row["dataset_index"]) for row in rows]
    if observed != expected:
        raise ValueError(
            "Ref reward rows do not match expected dataset indices. "
            f"observed={observed}, expected={expected}."
        )


def _shared_ref_count(rows: Sequence[Mapping[str, Any]]) -> int:
    counts = {len(_list_like(row["ref_rewards"])) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"All prompts must have the same ref reward count, got {counts}.")
    return counts.pop()
