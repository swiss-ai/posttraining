"""Tests for loading already-tokenized (Megatron .bin/.idx) preference data.

Tier 1 (pure numpy + datasets) exercises the loader: parquet-driven pairing, token-level
common-prefix boundary, and completion truncation. Tier 2 (guarded by torch/trl availability,
i.e. inside the training container) checks collator parity.

Run on the login node with an env that has numpy + datasets, or via srun in the container:
    pytest tests/test_tokenized_preference.py -q
"""

import struct

import numpy as np
import pytest

from post_training.data_alignment.tokenized_preference import (
    _longest_common_prefix_len,
    build_hf_dataset_from_tokenized,
)

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODE = {np.uint16: 8, np.int32: 4, np.int64: 5}


def _write_megatron_indexed(prefix, sequences, dtype=np.int32):
    """Write a minimal Megatron indexed dataset (one sequence == one document)."""
    seq_lengths = [len(s) for s in sequences]
    itemsize = np.dtype(dtype).itemsize
    pointers, curr = [], 0
    for length in seq_lengths:
        pointers.append(curr)
        curr += length * itemsize
    # one document per sequence -> document_indices = [0, 1, ..., n]
    document_indices = list(range(len(sequences) + 1))

    with open(prefix + ".bin", "wb") as f:
        for s in sequences:
            f.write(np.array(s, dtype=dtype).tobytes(order="C"))

    with open(prefix + ".idx", "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", _DTYPE_CODE[dtype]))
        f.write(struct.pack("<Q", len(sequences)))  # sequence_count
        f.write(struct.pack("<Q", len(document_indices)))  # document_count
        f.write(np.array(seq_lengths, dtype=np.int32).tobytes(order="C"))
        f.write(np.array(pointers, dtype=np.int64).tobytes(order="C"))
        f.write(np.array(document_indices, dtype=np.int64).tobytes(order="C"))


def _build_fixture(root):
    # accepted/rejected = tokens[prompt + response]; prompt is the shared prefix.
    accepted = [
        [10, 11, 12, 20, 21],  # prompt [10,11,12] + chosen response [20,21]
        [
            40,
            41,
            50,
            51,
            52,
            53,
            54,
            55,
        ],  # prompt [40,41] + long chosen response (len 6)
    ]
    rejected = [
        [10, 11, 12, 30, 31, 32],  # prompt [10,11,12] + rejected response [30,31,32]
        [40, 41, 60, 61],  # prompt [40,41] + rejected response [60,61]
    ]
    _write_megatron_indexed(str(root / "accepted"), accepted)
    _write_megatron_indexed(str(root / "rejected"), rejected)

    # Non-identity mapping: prove the parquet drives the pairing, not row position.
    from datasets import Dataset

    Dataset.from_dict({"chosen_index": [1, 0], "rejected_index": [1, 0]}).to_parquet(
        str(root / "pairs.parquet")
    )
    return accepted, rejected


def test_lcp_helper():
    assert _longest_common_prefix_len(np.array([1, 2, 3]), np.array([1, 2, 9])) == 2
    assert (
        _longest_common_prefix_len(np.array([1, 2]), np.array([1, 2, 3])) == 2
    )  # prefix
    assert _longest_common_prefix_len(np.array([5]), np.array([6])) == 0
    assert _longest_common_prefix_len(np.array([], dtype=int), np.array([1])) == 0


def test_load_tokenized_preference(tmp_path):
    _build_fixture(tmp_path)
    max_completion_length = 4

    ds = build_hf_dataset_from_tokenized(
        root=str(tmp_path),
        max_completion_length=max_completion_length,
    )

    # __len__ equals the number of parquet rows.
    assert len(ds) == 2
    assert set(ds.column_names) == {
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
    }

    # Row 0 -> accepted[1]/rejected[1] (the cap case). prompt = shared prefix [40,41].
    row0 = ds[0]
    assert row0["prompt_input_ids"] == [40, 41]
    assert row0["chosen_input_ids"] == [50, 51, 52, 53]  # len 6 truncated to 4
    assert row0["rejected_input_ids"] == [60, 61]
    assert len(row0["chosen_input_ids"]) <= max_completion_length

    # Row 1 -> accepted[0]/rejected[0]. prompt = shared prefix [10,11,12].
    row1 = ds[1]
    assert row1["prompt_input_ids"] == [10, 11, 12]
    assert row1["chosen_input_ids"] == [20, 21]
    assert row1["rejected_input_ids"] == [30, 31, 32]


def test_hf_dataset_supports_shuffle_select(tmp_path):
    _build_fixture(tmp_path)
    ds = build_hf_dataset_from_tokenized(root=str(tmp_path), max_completion_length=2048)
    sub = ds.shuffle(seed=0).select(range(1))
    assert len(sub) == 1
    assert set(sub.column_names) == {
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
    }


def test_bad_index_raises(tmp_path):
    _build_fixture(tmp_path)
    from datasets import Dataset

    # chosen_index 5 is out of bounds (accepted dataset has length 2).
    Dataset.from_dict({"chosen_index": [5], "rejected_index": [0]}).to_parquet(
        str(tmp_path / "pairs.parquet")
    )
    with pytest.raises(ValueError, match="out of bounds"):
        build_hf_dataset_from_tokenized(root=str(tmp_path), max_completion_length=2048)


def test_empty_completion_pairs_dropped(tmp_path):
    from datasets import Dataset

    # accepted[0] == rejected[0]: the common prefix consumes the whole sequence, leaving empty
    # completions. accepted[1]/rejected[1] diverge normally.
    _write_megatron_indexed(
        str(tmp_path / "accepted"), [[10, 11, 12], [40, 41, 50, 51]]
    )
    _write_megatron_indexed(str(tmp_path / "rejected"), [[10, 11, 12], [40, 41, 60]])
    Dataset.from_dict({"chosen_index": [0, 1], "rejected_index": [0, 1]}).to_parquet(
        str(tmp_path / "pairs.parquet")
    )

    ds = build_hf_dataset_from_tokenized(root=str(tmp_path), max_completion_length=2048)

    # The degenerate identical pair is dropped; only the diverging pair survives.
    assert len(ds) == 1
    assert ds[0]["prompt_input_ids"] == [40, 41]
    assert ds[0]["chosen_input_ids"] == [50, 51]
    assert ds[0]["rejected_input_ids"] == [60]


def test_collator_parity(tmp_path):
    """Tier 2: the produced rows collate to the same batch keys as the runtime path."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from post_training.trainers.preference import PreferenceTrainerCollator

    _build_fixture(tmp_path)
    ds = build_hf_dataset_from_tokenized(root=str(tmp_path), max_completion_length=2048)

    collator = PreferenceTrainerCollator(pad_token_id=0)
    batch = collator([ds[0], ds[1]])

    for key in (
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    ):
        assert key in batch
        assert isinstance(batch[key], torch.Tensor)
        assert batch[key].shape[0] == 2
