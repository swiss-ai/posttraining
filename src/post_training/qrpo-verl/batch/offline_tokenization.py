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
    """Tokenize offline candidates into a VERL-compatible training DataProto.

    Layout follows VERL AgentLoop conventions:

      prompts:
        left-padded prompt token block, shape [B, prompt_len]

      responses:
        right-padded trajectory token block, shape [B, response_len]

      response_mask:
        right-padded trainable-token mask over responses, shape [B, response_len]
        1 = assistant/model-generated token
        0 = tool/environment/padding token

      input_ids:
        concat(prompts, responses)

      attention_mask:
        concat(prompt_attention_mask, response_attention_mask)

      position_ids:
        computed from full attention_mask with VERL utility
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
        "padding": True,
        "truncation": bool(cfg.get("truncation", False)),
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

    full_input_ids = full["input_ids"]
    full_attention_mask = full["attention_mask"].bool()

    prompt_input_ids = prompt["input_ids"]
    prompt_attention_mask_for_lengths = prompt["attention_mask"].bool()

    if require_assistant_mask:
        full_assistant_mask = full["assistant_masks"].bool()
    else:
        full_assistant_mask = full_attention_mask

    prompt_rows: list[torch.Tensor] = []
    response_rows: list[torch.Tensor] = []
    response_mask_rows: list[torch.Tensor] = []

    for i, candidate in enumerate(candidates):
        prompt_len = int(prompt_attention_mask_for_lengths[i].sum().item())
        full_len = int(full_attention_mask[i].sum().item())

        if prompt_len > full_len:
            raise ValueError(
                f"Prompt prefix for trajectory {candidate.trajectory_id!r} is longer "
                "than the full conversation."
            )

        prompt_ids = prompt_input_ids[i, :prompt_len]
        full_prefix = full_input_ids[i, :prompt_len]

        if verify_prompt_prefix and not torch.equal(prompt_ids, full_prefix):
            raise ValueError(
                f"Prompt prefix tokens are not a prefix of the full conversation for "
                f"trajectory {candidate.trajectory_id!r}."
            )

        response_ids = full_input_ids[i, prompt_len:full_len]
        response_mask = full_assistant_mask[i, prompt_len:full_len]

        if response_ids.numel() == 0:
            raise ValueError(
                f"Offline trajectory {candidate.trajectory_id!r} tokenized to empty response."
            )

        if require_assistant_mask and not response_mask.any():
            raise ValueError(
                f"Offline trajectory {candidate.trajectory_id!r} has no trainable "
                "assistant/model tokens according to the chat template mask."
            )

        prompt_rows.append(prompt_ids)
        response_rows.append(response_ids)
        response_mask_rows.append(response_mask)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    old_padding_side = getattr(tokenizer, "padding_side", "right")

    try:
        tokenizer.padding_side = "left"
        prompt_output = tokenizer.pad(
            {"input_ids": prompt_rows},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        tokenizer.padding_side = "right"
        response_output = tokenizer.pad(
            {"input_ids": response_rows},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    finally:
        tokenizer.padding_side = old_padding_side

    prompts = prompt_output["input_ids"]
    prompt_attention_mask = prompt_output["attention_mask"].long()

    responses = response_output["input_ids"]
    response_attention_mask = response_output["attention_mask"].long()

    response_mask = torch.nn.utils.rnn.pad_sequence(
        [row.bool() for row in response_mask_rows],
        batch_first=True,
        padding_value=False,
    )

    if response_mask.shape != responses.shape:
        raise ValueError(
            "Padded response_mask shape does not match responses shape: "
            f"{tuple(response_mask.shape)} vs {tuple(responses.shape)}."
        )

    input_ids = torch.cat([prompts, responses], dim=-1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)
    position_ids = compute_position_id_with_mask(attention_mask)

    tensors = {
        K.PROMPTS: prompts,
        K.RESPONSES: responses,
        K.RESPONSE_MASK: response_mask,
        K.INPUT_IDS: input_ids,
        K.ATTENTION_MASK: attention_mask,
        K.POSITION_IDS: position_ids,
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
        "qrpo_batch_format": "verl_prompt_response",
        "source": K.SOURCE_OFFLINE,
    }
    if meta_info is not None:
        merged_meta_info.update(meta_info)

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info=merged_meta_info,
    )
