"""Load already-tokenized preference data (Megatron `.bin`/`.idx`) for offline DPO.

Layout expected on disk under ``<root>``:

- ``<accepted_prefix>.bin`` / ``.idx`` — accepted (chosen) sequences, each = ``tokens[prompt + response]``
- ``<rejected_prefix>.bin`` / ``.idx`` — rejected sequences, each = ``tokens[prompt + response]``
- ``<parquet_path>`` — a parquet "driver": one row per preference pair, with integer columns giving
  the index into the accepted dataset (``chosen_index_col``) and the rejected dataset
  (``rejected_index_col``). The parquet defines the number of training samples and the pairing; the
  accepted/rejected datasets need not be equal length or positionally aligned.

The prompt/completion boundary is recovered per pair as the token-level longest common prefix of the
paired accepted and rejected sequences (the prompt is shared, the responses diverge). This mirrors
TRL's ``extract_prompt`` and means no extra prompt-length metadata is required.

The result is wrapped into a HuggingFace ``datasets.Dataset`` whose rows hold exactly the keys the
``PreferenceTrainerCollator`` consumes (``prompt_input_ids`` / ``chosen_input_ids`` /
``rejected_input_ids``), so the existing collate/forward path is reused unchanged and runtime
tokenization is skipped.
"""

import json
import logging
import os
from typing import Optional

import numpy as np
from datasets import Dataset

from post_training.data_alignment.indexed_dataset import IndexedDataset

logger = logging.getLogger(__name__)

# Defaults for on-disk naming; all overridable via the dataset config.
DEFAULT_ACCEPTED_PREFIX = "accepted"
DEFAULT_REJECTED_PREFIX = "rejected"
DEFAULT_PARQUET_NAME = "pairs.parquet"
DEFAULT_CHOSEN_INDEX_COL = "chosen_index"
DEFAULT_REJECTED_INDEX_COL = "rejected_index"


def _longest_common_prefix_len(a: np.ndarray, b: np.ndarray) -> int:
    """Return the length of the longest common (token id) prefix of two 1-D arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    mismatch = a[:n] != b[:n]
    if not mismatch.any():
        return n  # one sequence is a prefix of the other (or they are identical)
    return int(np.argmax(mismatch))


class TokenizedPreferenceDataset:
    """Parquet-driven reader over two Megatron indexed datasets (accepted / rejected).

    ``__len__`` is the number of parquet rows; ``__getitem__(row)`` returns a dict with
    ``prompt_input_ids`` / ``chosen_input_ids`` / ``rejected_input_ids`` (lists of ints).
    """

    def __init__(
        self,
        root: str,
        max_completion_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        accepted_prefix: str = DEFAULT_ACCEPTED_PREFIX,
        rejected_prefix: str = DEFAULT_REJECTED_PREFIX,
        parquet_path: Optional[str] = None,
        chosen_index_col: str = DEFAULT_CHOSEN_INDEX_COL,
        rejected_index_col: str = DEFAULT_REJECTED_INDEX_COL,
        tokenizer=None,
        tokenizer_consistency: str = "warn",
    ) -> None:
        self.root = root
        self.max_completion_length = max_completion_length
        self.max_prompt_length = max_prompt_length
        self.chosen_index_col = chosen_index_col
        self.rejected_index_col = rejected_index_col

        self.accepted = IndexedDataset(os.path.join(root, accepted_prefix))
        self.rejected = IndexedDataset(os.path.join(root, rejected_prefix))

        parquet_path = parquet_path or os.path.join(root, DEFAULT_PARQUET_NAME)
        # The driver is small (two integer columns); load it eagerly into memory.
        pairs = Dataset.from_parquet(parquet_path)
        for col in (chosen_index_col, rejected_index_col):
            if col not in pairs.column_names:
                raise ValueError(
                    f"Parquet driver {parquet_path} is missing column '{col}'. "
                    f"Found columns: {pairs.column_names}"
                )
        self._chosen_idx = np.asarray(pairs[chosen_index_col], dtype=np.int64)
        self._rejected_idx = np.asarray(pairs[rejected_index_col], dtype=np.int64)

        self._validate()
        self._maybe_check_tokenizer(root, tokenizer, tokenizer_consistency)

    def _validate(self) -> None:
        if len(self._chosen_idx) != len(self._rejected_idx):
            raise ValueError(
                "Parquet driver has mismatched index columns: "
                f"{len(self._chosen_idx)} chosen vs {len(self._rejected_idx)} rejected."
            )
        n_acc, n_rej = len(self.accepted), len(self.rejected)
        if len(self._chosen_idx) and (
            self._chosen_idx.min() < 0 or self._chosen_idx.max() >= n_acc
        ):
            raise ValueError(
                f"chosen_index out of bounds for accepted dataset of length {n_acc} "
                f"(range [{self._chosen_idx.min()}, {self._chosen_idx.max()}])."
            )
        if len(self._rejected_idx) and (
            self._rejected_idx.min() < 0 or self._rejected_idx.max() >= n_rej
        ):
            raise ValueError(
                f"rejected_index out of bounds for rejected dataset of length {n_rej} "
                f"(range [{self._rejected_idx.min()}, {self._rejected_idx.max()}])."
            )
        logger.info(
            "TokenizedPreferenceDataset: %d pairs over accepted=%d / rejected=%d sequences.",
            len(self),
            n_acc,
            n_rej,
        )

    def _maybe_check_tokenizer(
        self, root, tokenizer, tokenizer_consistency: str
    ) -> None:
        """Compare the producer's tokenizer (manifest.json, if present) to the training one."""
        if tokenizer_consistency == "off" or tokenizer is None:
            return
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.exists(manifest_path):
            return
        with open(manifest_path) as f:
            manifest = json.load(f)
        produced = manifest.get("tokenizer_name_or_path")
        current = getattr(tokenizer, "name_or_path", None)
        if produced is not None and current is not None and produced != current:
            msg = (
                f"Tokenized data was produced with tokenizer '{produced}' but training uses "
                f"'{current}'. Token ids may be incompatible."
            )
            if tokenizer_consistency == "error":
                raise ValueError(msg)
            logger.warning(msg)

    def __len__(self) -> int:
        return len(self._chosen_idx)

    def __getitem__(self, row: int) -> dict:
        accepted_tokens = self.accepted[int(self._chosen_idx[row])]
        rejected_tokens = self.rejected[int(self._rejected_idx[row])]

        plen = _longest_common_prefix_len(accepted_tokens, rejected_tokens)

        prompt_ids = accepted_tokens[:plen]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[-self.max_prompt_length :]
        chosen_ids = accepted_tokens[plen:]
        rejected_ids = rejected_tokens[plen:]
        if self.max_completion_length is not None:
            chosen_ids = chosen_ids[: self.max_completion_length]
            rejected_ids = rejected_ids[: self.max_completion_length]

        return {
            "prompt_input_ids": prompt_ids.tolist(),
            "chosen_input_ids": chosen_ids.tolist(),
            "rejected_input_ids": rejected_ids.tolist(),
        }

    def to_generator(self):
        """Return a zero-arg generator of rows for ``datasets.Dataset.from_generator``.

        Pairs whose chosen or rejected completion is empty are dropped: an empty completion
        (e.g. ``accepted == rejected``, so the common prefix consumes the whole sequence) has
        length 0 and would divide by zero in the length-normalized DPO loss.
        """

        def _gen():
            dropped = 0
            for i in range(len(self)):
                row = self[i]
                if not row["chosen_input_ids"] or not row["rejected_input_ids"]:
                    dropped += 1
                    continue
                yield row
            if dropped:
                logger.warning(
                    "TokenizedPreferenceDataset: dropped %d/%d pairs with an empty chosen or "
                    "rejected completion (would break the length-normalized loss).",
                    dropped,
                    len(self),
                )

        return _gen


def build_hf_dataset_from_tokenized(
    root: str,
    tokenizer=None,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    tokenizer_consistency: str = "warn",
    accepted_prefix: str = DEFAULT_ACCEPTED_PREFIX,
    rejected_prefix: str = DEFAULT_REJECTED_PREFIX,
    parquet_path: Optional[str] = None,
    chosen_index_col: str = DEFAULT_CHOSEN_INDEX_COL,
    rejected_index_col: str = DEFAULT_REJECTED_INDEX_COL,
) -> Dataset:
    """Load tokenized preference data and return it shaped like ``load_dataset_flexible``.

    Args:
        root: Directory holding the accepted/rejected ``.bin``/``.idx`` files and the parquet driver.
        tokenizer: Training tokenizer, used only for the optional manifest consistency check.
        max_prompt_length / max_completion_length: Truncation caps (completions capped at e.g. 2048;
            prompt left-truncated, mirroring ``PreferenceTrainer.tokenize_row``).
        tokenizer_consistency: ``warn`` | ``error`` | ``off``.
        accepted_prefix / rejected_prefix / parquet_path / chosen_index_col / rejected_index_col:
            On-disk naming overrides.

    Returns:
        A ``datasets.Dataset`` whose rows hold ``prompt_input_ids`` / ``chosen_input_ids`` /
        ``rejected_input_ids``. Pairs with an empty chosen or rejected completion are dropped, so the
        returned dataset may have fewer rows than the parquet driver.
    """
    ds = TokenizedPreferenceDataset(
        root=root,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        tokenizer_consistency=tokenizer_consistency,
        accepted_prefix=accepted_prefix,
        rejected_prefix=rejected_prefix,
        parquet_path=parquet_path,
        chosen_index_col=chosen_index_col,
        rejected_index_col=rejected_index_col,
    )
    # Materialize once into an Arrow-backed HF Dataset so downstream code can use
    # .shuffle()/.select()/.sort()/len() exactly as it does for HF-tokenized data.
    return Dataset.from_generator(ds.to_generator())
