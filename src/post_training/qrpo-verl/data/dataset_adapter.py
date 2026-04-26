from __future__ import annotations

from typing import Any, Mapping, Sequence

from datasets import Dataset, load_dataset

from data.schemas import ChatMessage, PromptRecord, normalize_messages


def load_hf_dataset_from_config(config: Mapping[str, Any]) -> Dataset:
    path = config.get("path")
    split = config.get("split", "train")

    if path is None:
        raise ValueError("Dataset config path must be set.")

    from datasets import load_from_disk

    try:
        return load_from_disk(path)
    except FileNotFoundError:
        return load_dataset(path, split=split)


def row_to_prompt_record(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    row_index: int | None = None,
) -> PromptRecord:
    prompt_id_key = config.get("prompt_id_key", "prompt_id")
    prompt_messages_key = config.get("prompt_messages_key", "prompt_messages")
    ref_rewards_key = config.get("ref_rewards_key", "ref_rewards")
    offline_trajectories_key = config.get("offline_trajectories_key", "offline_trajectories")
    offline_rewards_key = config.get("offline_rewards_key", "offline_rewards")
    tools_key = config.get("tools_key", "tools")
    metadata_keys = config.get("metadata_keys", [])

    validate_records = bool(config.get("validate_records", False))
    deep_validate_messages = bool(config.get("deep_validate_messages", True))
    num_ref_rewards = config.get("num_ref_rewards", None)

    prompt_id = _require(row, prompt_id_key, row_index=row_index)

    prompt_messages = _message_list(
        _require(row, prompt_messages_key, row_index=row_index),
        field_name=prompt_messages_key,
        validate=validate_records and deep_validate_messages,
    )

    offline_trajectories = _trajectory_list(
        _require(row, offline_trajectories_key, row_index=row_index),
        field_name=offline_trajectories_key,
        validate=validate_records and deep_validate_messages,
    )

    ref_rewards = _list_like(_require(row, ref_rewards_key, row_index=row_index))
    offline_rewards = _list_like(_require(row, offline_rewards_key, row_index=row_index))

    tools = None
    if tools_key is not None and tools_key in row and row[tools_key] is not None:
        tools = tuple(_list_like(row[tools_key]))

    metadata = {key: row[key] for key in metadata_keys if key in row}

    record = PromptRecord(
        prompt_id=str(prompt_id),
        prompt_messages=prompt_messages,
        ref_rewards=tuple(float(x) for x in ref_rewards),
        offline_trajectories=offline_trajectories,
        offline_rewards=tuple(float(x) for x in offline_rewards),
        tools=tools,
        metadata=metadata,
    )

    if validate_records:
        record.validate(num_ref_rewards=num_ref_rewards, deep=deep_validate_messages)
    elif num_ref_rewards is not None and len(record.ref_rewards) != int(num_ref_rewards):
        raise ValueError(
            f"Prompt {record.prompt_id!r} has {len(record.ref_rewards)} ref rewards, "
            f"expected {num_ref_rewards}."
        )

    return record


def rows_to_prompt_records(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[PromptRecord]:
    return [row_to_prompt_record(row, config, row_index=i) for i, row in enumerate(rows)]


def dataset_batch_to_prompt_records(
    dataset: Dataset,
    indices: Sequence[int],
    config: Mapping[str, Any],
) -> list[PromptRecord]:
    return [
        row_to_prompt_record(dataset[int(i)], config, row_index=int(i))
        for i in indices
    ]


def _require(row: Mapping[str, Any], key: str, *, row_index: int | None) -> Any:
    if key not in row:
        prefix = f"Dataset row {row_index}" if row_index is not None else "Dataset row"
        raise KeyError(f"{prefix} is missing required key {key!r}.")
    return row[key]


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


def _message_list(
    value: Any,
    *,
    field_name: str,
    validate: bool,
) -> tuple[ChatMessage, ...]:
    messages = _list_like(value)
    if validate:
        return normalize_messages(messages, field_name=field_name)
    return tuple(messages)


def _trajectory_list(
    value: Any,
    *,
    field_name: str,
    validate: bool,
) -> tuple[tuple[ChatMessage, ...], ...]:
    trajectories = _list_like(value)

    output = []
    for i, trajectory in enumerate(trajectories):
        trajectory = _list_like(trajectory)
        if validate:
            output.append(normalize_messages(trajectory, field_name=f"{field_name}[{i}]"))
        else:
            output.append(tuple(trajectory))

    return tuple(output)
