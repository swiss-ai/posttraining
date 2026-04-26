from __future__ import annotations

from typing import Sequence

import numpy as np

from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length

from batch import keys as K


_SEQUENCE_KEYS = (
    K.INPUT_IDS,
    K.ATTENTION_MASK,
    K.POSITION_IDS,
    K.LOSS_MASK,
)


def concat_qrpo_training_dataprotos(
    dataprotos: Sequence[DataProto],
    *,
    pad_token_id: int,
    meta_info: dict | None = None,
) -> DataProto:
    """Pad and concatenate QRPO training DataProtos using VERL-native concat.

    All inputs must already use the QRPO full-sequence training schema:
      input_ids
      attention_mask
      position_ids
      loss_mask
      trajectory_reward
      ref_rewards

    Sequence tensors are right-padded to a common length before calling
    DataProto.concat, because DataProto.concat concatenates only along batch dim.
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

    max_seq_len = max(int(data.batch[K.INPUT_IDS].shape[-1]) for data in dataprotos)

    padded = [
        _pad_training_dataproto(
            data,
            max_seq_len=max_seq_len,
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
            "qrpo_batch_format": "full_sequence_with_loss_mask",
            "source": "mixed",
        }
    )

    if meta_info is not None:
        result.meta_info.update(meta_info)

    return result


def _pad_training_dataproto(
    data: DataProto,
    *,
    max_seq_len: int,
    pad_token_id: int,
) -> DataProto:
    tensors = {key: tensor for key, tensor in data.batch.items()}

    tensors[K.INPUT_IDS] = pad_sequence_to_length(
        tensors[K.INPUT_IDS],
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        left_pad=False,
    )
    tensors[K.ATTENTION_MASK] = pad_sequence_to_length(
        tensors[K.ATTENTION_MASK],
        max_seq_len=max_seq_len,
        pad_token_id=0,
        left_pad=False,
    )
    tensors[K.POSITION_IDS] = pad_sequence_to_length(
        tensors[K.POSITION_IDS],
        max_seq_len=max_seq_len,
        pad_token_id=0,
        left_pad=False,
    )
    tensors[K.LOSS_MASK] = pad_sequence_to_length(
        tensors[K.LOSS_MASK],
        max_seq_len=max_seq_len,
        pad_token_id=0,
        left_pad=False,
    )

    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=data.non_tensor_batch,
        meta_info=data.meta_info,
    )


def _check_training_schema(dataprotos: tuple[DataProto, ...]) -> None:
    required_tensor_keys = {
        K.INPUT_IDS,
        K.ATTENTION_MASK,
        K.POSITION_IDS,
        K.LOSS_MASK,
        K.TRAJECTORY_REWARD,
        K.REF_REWARDS,
    }
    required_non_tensor_keys = {
        K.PROMPT_ID,
        K.TRAJECTORY_ID,
        K.SOURCE,
    }

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

        for key, value in data.non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"DataProto {i} non_tensor_batch[{key!r}] must be a NumPy array, "
                    f"got {type(value).__name__}."
                )

            if value.ndim != 1:
                raise ValueError(
                    f"DataProto {i} non_tensor_batch[{key!r}] must be 1D with shape "
                    f"(batch_size,), got shape {value.shape}."
                )

            if value.shape[0] != len(data):
                raise ValueError(
                    f"DataProto {i} non_tensor_batch[{key!r}] has length "
                    f"{value.shape[0]}, but DataProto length is {len(data)}."
                )
