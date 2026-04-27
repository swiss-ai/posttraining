from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_sequence_to_length

from batch import keys as K


def concat_qrpo_training_dataprotos(
    dataprotos: Sequence[DataProto],
    *,
    pad_token_id: int,
    meta_info: dict | None = None,
) -> DataProto:
    """Pad and concatenate QRPO training DataProtos using VERL-native concat.

    Inputs must use the VERL prompt/response layout:

      prompts:       [B, prompt_len], left-padded
      responses:     [B, response_len], right-padded
      response_mask: [B, response_len]
      input_ids:     concat(prompts, responses)
      attention_mask: concat(prompt_attention, response_attention)
      position_ids:  computed from attention_mask

    This function pads prompts and responses separately, then recomputes
    input_ids, attention_mask, and position_ids so the final mixed batch keeps
    VERL's response-suffix convention.
    """

    dataprotos = tuple(dataprotos)

    if not dataprotos:
        raise ValueError("Cannot concatenate an empty list of DataProtos.")

    if len(dataprotos) == 1:
        result = dataprotos[0]
        if meta_info is not None:
            result.meta_info.update(meta_info)
        return result

    _check_training_schema(dataprotos)

    max_prompt_len = max(int(data.batch[K.PROMPTS].shape[-1]) for data in dataprotos)
    max_response_len = max(int(data.batch[K.RESPONSES].shape[-1]) for data in dataprotos)

    padded = [
        _pad_training_dataproto(
            data,
            max_prompt_len=max_prompt_len,
            max_response_len=max_response_len,
            pad_token_id=pad_token_id,
        )
        for data in dataprotos
    ]

    # DataProto.concat asserts on conflicting meta_info values. Online/offline
    # batches naturally have different source meta_info, so clear per-source
    # meta_info before concat and set mixed-batch meta_info afterwards.
    concat_inputs = [
        DataProto(
            batch=data.batch,
            non_tensor_batch=data.non_tensor_batch,
            meta_info={},
        )
        for data in padded
    ]

    result = DataProto.concat(concat_inputs)

    result.meta_info.update(
        {
            "qrpo_batch_format": "verl_prompt_response",
            "source": "mixed",
        }
    )

    if meta_info is not None:
        result.meta_info.update(meta_info)

    return result


def _pad_training_dataproto(
    data: DataProto,
    *,
    max_prompt_len: int,
    max_response_len: int,
    pad_token_id: int,
) -> DataProto:
    tensors = {key: tensor for key, tensor in data.batch.items()}

    prompt_len = int(tensors[K.PROMPTS].shape[-1])
    response_len = int(tensors[K.RESPONSES].shape[-1])

    prompt_attention_mask = tensors[K.ATTENTION_MASK][:, :prompt_len]
    response_attention_mask = tensors[K.ATTENTION_MASK][
        :, prompt_len : prompt_len + response_len
    ]

    tensors[K.PROMPTS] = pad_sequence_to_length(
        tensors[K.PROMPTS],
        max_seq_len=max_prompt_len,
        pad_token_id=pad_token_id,
        left_pad=True,
    )

    prompt_attention_mask = pad_sequence_to_length(
        prompt_attention_mask,
        max_seq_len=max_prompt_len,
        pad_token_id=0,
        left_pad=True,
    )

    tensors[K.RESPONSES] = pad_sequence_to_length(
        tensors[K.RESPONSES],
        max_seq_len=max_response_len,
        pad_token_id=pad_token_id,
        left_pad=False,
    )

    response_attention_mask = pad_sequence_to_length(
        response_attention_mask,
        max_seq_len=max_response_len,
        pad_token_id=0,
        left_pad=False,
    )

    tensors[K.RESPONSE_MASK] = pad_sequence_to_length(
        tensors[K.RESPONSE_MASK],
        max_seq_len=max_response_len,
        pad_token_id=0,
        left_pad=False,
    ).bool()

    for key in (K.LOG_PROBS, K.REF_LOG_PROBS):
        if key in tensors:
            tensors[key] = pad_sequence_to_length(
                tensors[key],
                max_seq_len=max_response_len,
                pad_token_id=0,
                left_pad=False,
            )

    tensors[K.INPUT_IDS] = torch.cat([tensors[K.PROMPTS], tensors[K.RESPONSES]], dim=-1)
    tensors[K.ATTENTION_MASK] = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1).long()
    tensors[K.POSITION_IDS] = compute_position_id_with_mask(tensors[K.ATTENTION_MASK])

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=data.non_tensor_batch,
        meta_info=data.meta_info,
    )


def _check_training_schema(dataprotos: tuple[DataProto, ...]) -> None:
    required_tensor_keys = {
        K.PROMPTS,
        K.RESPONSES,
        K.RESPONSE_MASK,
        K.INPUT_IDS,
        K.ATTENTION_MASK,
        K.POSITION_IDS,
        K.TRAJECTORY_REWARD,
        K.REF_REWARDS,
    }
    required_non_tensor_keys = {
        K.PROMPT_ID,
        K.TRAJECTORY_ID,
        K.SOURCE,
    }

    if dataprotos[0].batch is None:
        raise ValueError("DataProto 0 has no tensor batch.")

    first_tensor_keys = set(dataprotos[0].batch.keys())
    first_non_tensor_keys = set(dataprotos[0].non_tensor_batch.keys())

    for i, data in enumerate(dataprotos):
        if data.batch is None:
            raise ValueError(f"DataProto {i} has no tensor batch.")

        tensor_keys = set(data.batch.keys())
        missing_tensor_keys = required_tensor_keys - tensor_keys
        if missing_tensor_keys:
            raise KeyError(f"DataProto {i} is missing tensor keys: {sorted(missing_tensor_keys)}")

        if tensor_keys != first_tensor_keys:
            raise ValueError(
                f"DataProto {i} tensor keys do not match the first DataProto. "
                f"Got {sorted(tensor_keys)}, expected {sorted(first_tensor_keys)}."
            )

        non_tensor_keys = set(data.non_tensor_batch.keys())
        missing_non_tensor_keys = required_non_tensor_keys - non_tensor_keys
        if missing_non_tensor_keys:
            raise KeyError(
                f"DataProto {i} is missing non-tensor keys: {sorted(missing_non_tensor_keys)}"
            )

        if non_tensor_keys != first_non_tensor_keys:
            raise ValueError(
                f"DataProto {i} non-tensor keys do not match the first DataProto. "
                f"Got {sorted(non_tensor_keys)}, expected {sorted(first_non_tensor_keys)}."
            )

        _check_prompt_response_shapes(data, index=i)
        _check_non_tensor_batch(data, index=i)


def _check_prompt_response_shapes(data: DataProto, *, index: int) -> None:
    prompts = data.batch[K.PROMPTS]
    responses = data.batch[K.RESPONSES]
    response_mask = data.batch[K.RESPONSE_MASK]
    input_ids = data.batch[K.INPUT_IDS]
    attention_mask = data.batch[K.ATTENTION_MASK]
    position_ids = data.batch[K.POSITION_IDS]

    if prompts.ndim != 2:
        raise ValueError(f"DataProto {index} prompts must have shape [B, P].")

    if responses.ndim != 2:
        raise ValueError(f"DataProto {index} responses must have shape [B, R].")

    if response_mask.shape != responses.shape:
        raise ValueError(
            f"DataProto {index} response_mask shape {tuple(response_mask.shape)} "
            f"does not match responses shape {tuple(responses.shape)}."
        )

    expected_seq_len = prompts.shape[1] + responses.shape[1]

    if input_ids.shape[1] != expected_seq_len:
        raise ValueError(
            f"DataProto {index} input_ids sequence length must equal "
            f"prompt_len + response_len: {input_ids.shape[1]} != "
            f"{prompts.shape[1]} + {responses.shape[1]}."
        )

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"DataProto {index} attention_mask shape {tuple(attention_mask.shape)} "
            f"does not match input_ids shape {tuple(input_ids.shape)}."
        )

    if position_ids.shape != input_ids.shape:
        raise ValueError(
            f"DataProto {index} position_ids shape {tuple(position_ids.shape)} "
            f"does not match input_ids shape {tuple(input_ids.shape)}."
        )

    for key in (K.LOG_PROBS, K.REF_LOG_PROBS):
        if key in data.batch and data.batch[key].shape != responses.shape:
            raise ValueError(
                f"DataProto {index} {key} shape {tuple(data.batch[key].shape)} "
                f"does not match responses shape {tuple(responses.shape)}."
            )


def _check_non_tensor_batch(data: DataProto, *, index: int) -> None:
    for key, value in data.non_tensor_batch.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"DataProto {index} non_tensor_batch[{key!r}] must be a NumPy array, "
                f"got {type(value).__name__}."
            )

        if value.ndim != 1:
            raise ValueError(
                f"DataProto {index} non_tensor_batch[{key!r}] must be 1D with shape "
                f"(batch_size,), got shape {value.shape}."
            )

        if value.shape[0] != len(data):
            raise ValueError(
                f"DataProto {index} non_tensor_batch[{key!r}] has length "
                f"{value.shape[0]}, but DataProto length is {len(data)}."
            )