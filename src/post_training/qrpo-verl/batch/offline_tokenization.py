from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

from batch import keys as K
from data.schemas import OfflineTrajectoryCandidate


def offline_candidates_to_dataproto(
    *,
    candidates: Sequence[OfflineTrajectoryCandidate],
    tokenizer: Any,
    config: Mapping[str, Any] | None = None,
    meta_info: dict[str, Any] | None = None,
) -> DataProto:
    """Tokenize offline candidates and return a VERL DataProto directly.

    Output tensors:
      input_ids
      attention_mask
      position_ids
      loss_mask
      trajectory_reward
      ref_rewards

    Non-tensor metadata:
      prompt_id
      trajectory_id
      source
    """

    cfg = config or {}
    candidates = tuple(candidates)

    if not candidates:
        raise ValueError("Cannot tokenize an empty offline candidate list.")

    require_assistant_mask = bool(cfg.get("require_assistant_mask", True))
    verify_prompt_prefix = bool(cfg.get("verify_prompt_prefix", True))
    require_same_tools = bool(cfg.get("require_same_tools", True))

    tools = candidates[0].tools
    if require_same_tools and any(candidate.tools != tools for candidate in candidates[1:]):
        raise ValueError(
            "Offline tokenization currently requires all candidates in a batch "
            "to share the same tools spec. Group by tools before tokenization."
        )

    full_conversations = [
        list(candidate.prompt_messages) + list(candidate.trajectory_messages)
        for candidate in candidates
    ]
    prompt_conversations = [list(candidate.prompt_messages) for candidate in candidates]

    common_kwargs: dict[str, Any] = {
        "tokenize": True,
        "padding": cfg.get("padding", True),
        "truncation": cfg.get("truncation", False),
        "return_tensors": "pt",
        "return_dict": True,
    }

    if cfg.get("max_length") is not None:
        common_kwargs["max_length"] = cfg["max_length"]

    if tools is not None:
        common_kwargs["tools"] = list(tools)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"

    try:
        full = tokenizer.apply_chat_template(
            full_conversations,
            add_generation_prompt=False,
            return_assistant_tokens_mask=require_assistant_mask,
            **common_kwargs,
        )

        prompt = tokenizer.apply_chat_template(
            prompt_conversations,
            add_generation_prompt=True,
            **common_kwargs,
        )
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"].bool()

    prompt_input_ids = prompt["input_ids"]
    prompt_attention_mask = prompt["attention_mask"].bool()

    assistant_mask = full["assistant_masks"].bool() if require_assistant_mask else attention_mask
    loss_mask = torch.zeros_like(attention_mask, dtype=torch.bool)

    for i, candidate in enumerate(candidates):
        prompt_len = int(prompt_attention_mask[i].sum().item())
        full_len = int(attention_mask[i].sum().item())

        if prompt_len > full_len:
            raise ValueError(
                f"Prompt prefix for trajectory {candidate.trajectory_id!r} is longer "
                "than the full conversation."
            )

        if verify_prompt_prefix and not torch.equal(
            prompt_input_ids[i, :prompt_len],
            input_ids[i, :prompt_len],
        ):
            raise ValueError(
                f"Prompt prefix tokens are not a prefix of the full conversation for "
                f"trajectory {candidate.trajectory_id!r}."
            )

        if prompt_len == full_len:
            raise ValueError(
                f"Offline trajectory {candidate.trajectory_id!r} tokenized to empty suffix."
            )

        candidate_loss_mask = assistant_mask[i, prompt_len:full_len]

        if require_assistant_mask and not candidate_loss_mask.any():
            raise ValueError(
                f"Offline trajectory {candidate.trajectory_id!r} has no trainable "
                "assistant/model tokens according to the chat template mask."
            )

        loss_mask[i, prompt_len:full_len] = candidate_loss_mask

    attention_mask_long = attention_mask.long()

    tensors = {
        K.INPUT_IDS: input_ids,
        K.ATTENTION_MASK: attention_mask_long,
        K.POSITION_IDS: compute_position_id_with_mask(attention_mask_long),
        K.LOSS_MASK: loss_mask,
        K.TRAJECTORY_REWARD: torch.tensor(
            [float(candidate.reward) for candidate in candidates],
            dtype=torch.float32,
        ),
        K.REF_REWARDS: torch.tensor(
            [tuple(float(x) for x in candidate.ref_rewards) for candidate in candidates],
            dtype=torch.float32,
        ),
    }

    non_tensors = {
        K.PROMPT_ID: np.asarray([candidate.prompt_id for candidate in candidates], dtype=object),
        K.TRAJECTORY_ID: np.asarray(
            [candidate.trajectory_id for candidate in candidates],
            dtype=object,
        ),
        K.SOURCE: np.asarray([K.SOURCE_OFFLINE] * len(candidates), dtype=object),
    }

    merged_meta_info = {
        "qrpo_batch_format": "full_sequence_with_loss_mask",
        "source": K.SOURCE_OFFLINE,
    }
    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=merged_meta_info,
    )
