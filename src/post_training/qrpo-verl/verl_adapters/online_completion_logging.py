from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from verl.protocol import DataProto

from batch import keys as K


def build_online_completion_log_rows(
        *,
        config: Any,
        online_requests: Sequence[Any],
        rollout_output: DataProto,
        rewards: torch.Tensor,
        online_rollout_config: Mapping[str, Any],
        tokenizer: Any,
        meta_info: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    logging_config = resolve_completion_logging_config(
        config,
        online_rollout_config,
    )
    if not bool(logging_config["enabled"]):
        return []

    if tokenizer is None:
        raise ValueError(
            "online_rollout.completion_logging requires a tokenizer to decode "
            "generated response tokens."
        )

    if rollout_output.batch is None:
        raise ValueError("rollout_output.batch is required.")

    if rewards.ndim != 1 or rewards.shape[0] != len(rollout_output):
        raise ValueError(
            f"rewards must have shape ({len(rollout_output)},), got "
            f"{tuple(rewards.shape)}."
        )

    step = None if meta_info is None else meta_info.get("global_steps")
    indices = _select_completion_log_indices(
        rewards=rewards,
        logging_config=logging_config,
        step=step,
    )
    max_chars = int(logging_config["max_chars"])
    raw_prompts = rollout_output.non_tensor_batch.get(K.RAW_PROMPT)
    reward_extra_keys = (rollout_output.meta_info or {}).get("reward_extra_keys", [])

    rows: list[dict[str, Any]] = []
    for index in indices:
        request = online_requests[index] if index < len(online_requests) else None
        prompt_messages = _array_value(
            raw_prompts,
            index,
            default=getattr(request, "prompt_messages", ()),
        )
        prompt_id = _array_value(
            rollout_output.non_tensor_batch.get(K.PROMPT_ID),
            index,
            default=getattr(request, "prompt_id", None),
        )
        trajectory_id = _array_value(
            rollout_output.non_tensor_batch.get(K.TRAJECTORY_ID),
            index,
            default=getattr(request, "trajectory_id", None),
        )
        ref_rewards = _array_value(
            rollout_output.non_tensor_batch.get(K.REF_REWARDS),
            index,
            default=getattr(request, "ref_rewards", ()),
        )

        row: dict[str, Any] = {
            "step": step,
            "epoch": None if meta_info is None else meta_info.get("epoch"),
            "sample_index": index,
            K.PROMPT_ID: _to_debug_value(prompt_id),
            K.TRAJECTORY_ID: _to_debug_value(trajectory_id),
            K.TRAJECTORY_REWARD: float(rewards[index].detach().cpu().item()),
            K.REF_REWARDS: [float(value) for value in tuple(ref_rewards)],
            "prompt": _truncate_debug_text(
                _format_debug_messages(prompt_messages),
                max_chars=max_chars,
            ),
            "completion": _truncate_debug_text(
                _decode_online_completion(
                    rollout_output=rollout_output,
                    index=index,
                    tokenizer=tokenizer,
                ),
                max_chars=max_chars,
            ),
        }

        for key in reward_extra_keys:
            if key in rollout_output.non_tensor_batch:
                row[key] = _to_debug_value(
                    _array_value(rollout_output.non_tensor_batch[key], index)
                )

        rows.append(row)

    return rows


def enrich_online_completion_log_rows(
        *,
        train_batch: DataProto,
        rows: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if train_batch.batch is None:
        raise ValueError("train_batch.batch is required.")

    trajectory_ids = train_batch.non_tensor_batch.get(K.TRAJECTORY_ID)
    if trajectory_ids is None:
        raise KeyError(f"train_batch.non_tensor_batch is missing {K.TRAJECTORY_ID!r}.")

    row_by_trajectory_id = {
        str(row[K.TRAJECTORY_ID]): dict(row)
        for row in rows
    }
    ordered_ids = [str(row[K.TRAJECTORY_ID]) for row in rows]

    enrich_tensor_keys = (
        K.TRAJECTORY_REWARD,
        K.REF_QUANTILE,
        K.TRANSFORMED_REWARD,
        K.TRAJECTORY_LENGTH,
        K.EFFECTIVE_BETA,
        K.BETA_LOG_PARTITION,
    )
    for batch_index, trajectory_id in enumerate(trajectory_ids):
        row = row_by_trajectory_id.get(str(trajectory_id))
        if row is None:
            continue

        if K.REF_REWARDS in train_batch.batch:
            row[K.REF_REWARDS] = [
                float(value)
                for value in train_batch.batch[K.REF_REWARDS][batch_index]
                .detach()
                .cpu()
                .tolist()
            ]

        for key in enrich_tensor_keys:
            if key in train_batch.batch:
                row[key] = float(
                    train_batch.batch[key][batch_index].detach().cpu().item()
                )

        row_by_trajectory_id[str(trajectory_id)] = row

    missing = [
        trajectory_id
        for trajectory_id in ordered_ids
        if K.REF_QUANTILE not in row_by_trajectory_id[trajectory_id]
    ]
    if missing:
        raise ValueError(
            "Could not match online completion log rows to QRPO train batch "
            f"trajectory_ids: {missing}."
        )

    return [row_by_trajectory_id[trajectory_id] for trajectory_id in ordered_ids]


def log_online_completion_rows(
        *,
        config: Any,
        rows: Sequence[Mapping[str, Any]],
        step: int,
) -> None:
    if not rows:
        return

    logging_config = resolve_completion_logging_config(config, None)
    outputs = {str(output) for output in logging_config["outputs"]}

    if "console" in outputs:
        for row in rows:
            print(
                "[online completion] "
                f"step={step} sample_index={row.get('sample_index')} "
                f"prompt_id={row.get(K.PROMPT_ID)} "
                f"trajectory_id={row.get(K.TRAJECTORY_ID)} "
                f"reward={row.get(K.TRAJECTORY_REWARD)} "
                f"ref_quantile={row.get(K.REF_QUANTILE)}"
            )
            print(f"prompt: {row.get('prompt')}")
            print(f"completion: {row.get('completion')}")

    if "wandb" not in outputs:
        return

    try:
        import wandb
    except Exception:
        return

    if getattr(wandb, "run", None) is None:
        return

    preferred_columns = [
        "step",
        "epoch",
        "sample_index",
        K.PROMPT_ID,
        K.TRAJECTORY_ID,
        K.TRAJECTORY_REWARD,
        K.REF_QUANTILE,
        K.TRANSFORMED_REWARD,
        K.TRAJECTORY_LENGTH,
        K.EFFECTIVE_BETA,
        K.BETA_LOG_PARTITION,
        K.REF_REWARDS,
        "prompt",
        "completion",
    ]
    columns = [
        column
        for column in preferred_columns
        if any(column in row for row in rows)
    ]
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)

    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*(row.get(column) for column in columns))

    wandb.log({"online/completions": table}, step=step)


def resolve_completion_logging_config(
        config: Any,
        online_rollout_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw_config = None
    plain_rollout_config = _to_plain_container(online_rollout_config)
    if isinstance(plain_rollout_config, Mapping):
        raw_config = plain_rollout_config.get("completion_logging")

    if raw_config is None:
        raw_config = _select(
            config,
            "online_rollout.completion_logging",
            default={},
        )

    raw_config = _to_plain_container(raw_config) or {}
    if not isinstance(raw_config, Mapping):
        raise TypeError("online_rollout.completion_logging must be a mapping.")

    resolved = {
        "enabled": False,
        "outputs": ["wandb"],
        "selection": "random",
        "n": 8,
        "seed": 0,
        "max_chars": 2000,
    }
    resolved.update(raw_config)

    if isinstance(resolved["outputs"], str):
        resolved["outputs"] = [resolved["outputs"]]
    resolved["outputs"] = list(resolved["outputs"])

    return resolved


def _decode_online_completion(
        *,
        rollout_output: DataProto,
        index: int,
        tokenizer: Any,
) -> str:
    batch = rollout_output.batch
    responses = batch[K.RESPONSES][index]
    response_len = int(responses.shape[0])
    prompt_len = int(batch[K.PROMPTS].shape[1])
    response_attention = batch[K.ATTENTION_MASK][
        index,
        prompt_len:prompt_len + response_len,
    ]
    valid_response_len = int(
        response_attention.detach().cpu().long().sum().item()
    )
    valid_response_len = max(0, min(valid_response_len, response_len))

    token_ids = responses[:valid_response_len].detach().cpu().tolist()
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return " ".join(str(token_id) for token_id in token_ids)

    return str(decode(token_ids, skip_special_tokens=True))


def _select_completion_log_indices(
        *,
        rewards: torch.Tensor,
        logging_config: Mapping[str, Any],
        step: Any,
) -> list[int]:
    batch_size = int(rewards.shape[0])
    if batch_size == 0:
        return []

    selection = str(logging_config["selection"])
    if selection == "all":
        return list(range(batch_size))

    n = min(max(int(logging_config["n"]), 0), batch_size)
    if n == 0:
        return []

    if selection == "first":
        return list(range(n))

    detached_rewards = rewards.detach().cpu().float()
    if selection == "top_reward":
        return torch.argsort(detached_rewards, descending=True)[:n].tolist()

    if selection == "bottom_reward":
        return torch.argsort(detached_rewards, descending=False)[:n].tolist()

    if selection == "random":
        seed = int(logging_config["seed"])
        if step is not None:
            try:
                seed += int(step)
            except Exception:
                pass
        return random.Random(seed).sample(range(batch_size), n)

    raise ValueError(
        "online_rollout.completion_logging.selection must be one of "
        "'all', 'first', 'random', 'top_reward', or 'bottom_reward'."
    )


def _array_value(values: Any, index: int, *, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        return values[index]
    except Exception:
        return default


def _format_debug_messages(messages: Any) -> str:
    if messages is None:
        return ""

    if isinstance(messages, Mapping):
        messages = [messages]

    if isinstance(messages, str):
        return messages

    try:
        iterable = list(messages)
    except Exception:
        return str(messages)

    parts: list[str] = []
    for message in iterable:
        if isinstance(message, Mapping):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(message))

    return "\n".join(parts)


def _truncate_debug_text(value: Any, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


def _to_debug_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, tuple):
        return [_to_debug_value(item) for item in value]

    if isinstance(value, list):
        return [_to_debug_value(item) for item in value]

    if isinstance(value, Mapping):
        return {
            str(key): _to_debug_value(item)
            for key, item in value.items()
        }

    return value


def _select(config: Any, path: str, *, default: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, path, default=default)

    cur = config
    for part in path.split("."):
        try:
            cur = cur[part]
        except Exception:
            return default

    return cur


def _to_plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value
