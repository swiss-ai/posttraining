# Vendored, trimmed read-only reader for Megatron-LM indexed datasets (.bin/.idx).
#
# Adapted from Megatron-LM `megatron/core/datasets/indexed_dataset.py`
# (https://github.com/NVIDIA/Megatron-LM, source commit 82eb50beb), which is
# licensed under the MIT license (Copyright (c) Facebook, Inc. and its affiliates).
#
# Only the read path is kept (mmap-based): `DType`, `_IndexReader`, `_MMapBinReader`,
# `IndexedDataset`, and the `get_idx_path`/`get_bin_path` helpers. The S3 / multi-storage-client
# readers, the dataset builder, and the `megatron.core.*` imports were removed so this module
# depends only on `numpy`. This lets us load already-tokenized preference data without adding a
# `megatron-core` dependency (which would require rebuilding the training container).

import os
import struct
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from typing import List, Optional, Tuple, Type, Union

import numpy

_INDEX_HEADER = b"MMIDIDX\x00\x00"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices."""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[numpy.number]:
        """Get the dtype from the code."""
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes."""
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError


class _IndexReader(object):
    """Object class to read the index (.idx) file."""

    def __init__(self, idx_path: str, multimodal: bool = False) -> None:
        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset
            + self.sequence_lengths.nbytes
            + self.sequence_pointers.nbytes,
        )

        self.sequence_modes = None
        if multimodal:
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(
        self, idx: int
    ) -> Tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index."""
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class _MMapBinReader(object):
    """A reader that memory maps the data (.bin) file."""

    def __init__(self, bin_path: str) -> None:
        self._bin_file_reader = open(bin_path, mode="rb")
        self._bin_buffer_mmap = numpy.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read `count` items of `dtype` from the data file starting at byte `offset`."""
        return numpy.frombuffer(
            self._bin_buffer, dtype=dtype, count=count, offset=offset
        )

    def __del__(self) -> None:
        if getattr(self, "_bin_buffer_mmap", None) is not None:
            self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        if getattr(self, "_bin_file_reader", None) is not None:
            self._bin_file_reader.close()


class IndexedDataset(object):
    """The low-level read-only interface for a Megatron-LM indexed dataset.

    Args:
        path_prefix (str): The shared prefix of the `.idx` and `.bin` files.
        multimodal (bool): Whether the dataset stores per-sequence modes. Defaults to False.
    """

    def __init__(self, path_prefix: str, multimodal: bool = False) -> None:
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self._initialize(path_prefix, multimodal)

        assert self.index.sequence_lengths.shape[0] == self.index.document_indices[-1]
        assert self.index.sequence_lengths.shape[0] == len(self.index)
        assert self.index.sequence_lengths.shape[0] == self.index.sequence_count

    def _initialize(self, path_prefix: str, multimodal: bool) -> None:
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)
        assert os.path.exists(idx_path) and os.path.exists(bin_path), (
            "One or both of the .idx and .bin files cannot be found at the "
            f"path prefix {path_prefix}"
        )
        self.bin_reader = _MMapBinReader(bin_path)
        self.index = _IndexReader(idx_path, self.multimodal)

    def __getstate__(self) -> Tuple[str, bool]:
        return self.path_prefix, self.multimodal

    def __setstate__(self, state: Tuple[str, bool]) -> None:
        path_prefix, multimodal = state
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self._initialize(path_prefix, multimodal)

    def __del__(self) -> None:
        if hasattr(self, "bin_reader"):
            del self.bin_reader
        if hasattr(self, "index"):
            del self.index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.number], List[numpy.ndarray]]:
        """Return the token sequence (and mode, if multimodal) at the index or slice."""
        if isinstance(idx, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = self.index.sequence_modes[idx] if self.multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (
                (sequences, sequence_modes) if sequence_modes is not None else sequences
            )
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        return self.index.sequence_lengths


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix."""
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix."""
    return path_prefix + ".bin"
