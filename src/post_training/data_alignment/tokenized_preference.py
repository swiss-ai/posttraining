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

This module also supports MIXING several preference datasets (``build_mixed_preference_dataset``): each
entry is either a tokenized (Megatron) source or a raw HuggingFace preference source (string/conversational
``chosen``/``rejected``, tokenized here with the same pipeline the trainer uses). Each entry has an
optional ``max_samples`` cap (subselected via ``selection`` = ``random``|``head``). Entries are combined
either by ``interleave`` (probabilistic, per-entry ``weight`` summing to 1, ``stopping_strategy``) or by
``concatenate`` (each source in full up to its cap, in order), with an optional ``total_samples`` cap on
the result. The mixed dataset is uniformly pre-tokenized, so the trainer still skips its own tokenization.
"""

import json
import logging
import os
import shutil
from hashlib import blake2b
from typing import Optional

import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_from_disk,
)

from post_training.data_alignment.indexed_dataset import IndexedDataset

logger = logging.getLogger(__name__)

# The three columns every preference source is reduced to before mixing / collation.
PREFERENCE_TOKEN_COLUMNS = [
    "prompt_input_ids",
    "chosen_input_ids",
    "rejected_input_ids",
]

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


def _drop_empty_completions(
    dataset: Dataset, num_proc: Optional[int] = None
) -> Dataset:
    """Drop rows whose chosen or rejected completion is empty (would break length-normalized DPO)."""
    n = len(dataset)
    filtered = dataset.filter(
        lambda ex: len(ex["chosen_input_ids"]) > 0
        and len(ex["rejected_input_ids"]) > 0,
        num_proc=num_proc,
    )
    if len(filtered) < n:
        logger.warning(
            "Dropped %d/%d rows with an empty chosen/rejected completion.",
            n - len(filtered),
            n,
        )
    return filtered


def tokenize_raw_preference_dataset(
    dataset: Dataset,
    tokenizer,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    is_encoder_decoder: bool = False,
    num_proc: Optional[int] = None,
) -> Dataset:
    """Tokenize a raw HF preference dataset (string/conversational ``chosen``/``rejected``) into the
    ``prompt_input_ids`` / ``chosen_input_ids`` / ``rejected_input_ids`` schema.

    Reuses the exact runtime pipeline (``maybe_extract_prompt`` -> ``maybe_apply_chat_template`` ->
    ``PreferenceTrainer.tokenize_row``) so the tokens match the non-mixed path. trl / the trainer are
    imported lazily so this module stays importable without torch/trl (e.g. on a login node).

    TEXT-ONLY: this uses ``tokenize_row`` (not the vision ``process_row``); any image/vision columns must
    be handled before calling this (the mixture builder drops them and warns). Multimodal raw sources are
    not supported on this path.
    """
    from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt

    from post_training.trainers.preference import PreferenceTrainer

    dataset = dataset.map(
        maybe_extract_prompt, num_proc=num_proc, desc="Extracting prompt"
    )
    dataset = dataset.map(
        maybe_apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=num_proc,
        desc="Applying chat template",
    )
    dataset = dataset.map(
        PreferenceTrainer.tokenize_row,
        fn_kwargs={
            "processing_class": tokenizer,
            "max_prompt_length": max_prompt_length,
            "max_completion_length": max_completion_length,
            "add_special_tokens": is_encoder_decoder,
        },
        num_proc=num_proc,
        desc="Tokenizing raw preference dataset",
    )
    extra = [c for c in dataset.column_names if c not in PREFERENCE_TOKEN_COLUMNS]
    if extra:
        dataset = dataset.remove_columns(extra)
    return _drop_empty_completions(dataset, num_proc=num_proc)


def _subselect(
    dataset: Dataset, max_samples: Optional[int], selection: str, seed: int
) -> Dataset:
    """Cap ``dataset`` to ``max_samples`` rows using ``selection`` (``random`` | ``head``)."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if selection == "head":
        return dataset.select(range(max_samples))
    if selection == "random":
        return dataset.shuffle(seed=seed).select(range(max_samples))
    raise ValueError(f"Unknown selection '{selection}'; expected 'random' or 'head'.")


def _load_raw_for_mixture(entry, seed: int) -> Dataset:
    """Load a raw HF preference entry, drop image columns (text-only path), and subselect into memory.

    Returns an in-memory ``Dataset`` (chosen/rejected only), pre-tokenization. Materializing via
    ``from_dict`` matters: a ``load_from_disk`` dataset writes transform caches next to its source files
    (which may be read-only); an in-memory dataset writes to the (writable) HF cache instead.
    """
    raw = load_from_disk(entry["path"])
    if isinstance(raw, DatasetDict):
        raw = raw[entry["split"]]
    # Raw entries are tokenized TEXT-only (tokenize_row, not process_row); any image/vision columns
    # are dropped. Warn rather than silently lose multimodal inputs.
    image_cols = [
        c
        for c in raw.column_names
        if c
        in ("images", "image", "pixel_values", "pixel_attention_mask", "image_sizes")
    ]
    if image_cols:
        logger.warning(
            "Raw mixture entry %s has image columns %s that will be DROPPED — the raw mixture "
            "path is text-only (uses tokenize_row).",
            entry["path"],
            image_cols,
        )
    keep = [c for c in ("chosen", "rejected") if c in raw.column_names]
    raw = raw.select_columns(keep)

    max_samples = entry.get("max_samples", None)
    selection = entry.get("selection", "random")
    n = len(raw)
    if max_samples is not None and max_samples < n:
        if selection == "random":
            # NOTE: this numpy RNG is independent of the tokenized path's _subselect (HF .shuffle), so
            # the same seed selects DIFFERENT rows across the two source types — not cross-source aligned.
            indices = np.random.default_rng(seed).permutation(n)[:max_samples]
        elif selection == "head":
            indices = np.arange(max_samples)
        else:
            raise ValueError(
                f"Unknown selection '{selection}'; expected 'random' or 'head'."
            )
        return Dataset.from_dict(raw[sorted(int(i) for i in indices)])
    # No effective cap (max_samples is None or >= source size): materialize ALL rows into memory.
    logger.warning(
        "Raw mixture entry %s has no effective max_samples cap; materializing all %d rows into memory "
        "before tokenization. Set max_samples to bound memory for large sources.",
        entry["path"],
        n,
    )
    return Dataset.from_dict(raw[:])


def _raw_cache_key(
    entry, tokenizer, max_prompt_length, max_completion_length, is_encoder_decoder, seed
) -> str:
    """Stable key for an hf entry's tokenized subset, so identical runs reuse the on-disk cache."""
    payload = {
        "path": entry["path"],
        "split": entry.get("split"),
        "max_samples": entry.get("max_samples"),
        "selection": entry.get("selection", "random"),
        "seed": seed,
        "max_prompt_length": max_prompt_length,
        "max_completion_length": max_completion_length,
        "is_encoder_decoder": is_encoder_decoder,
        # Keyed on the tokenizer name/path only (not its vocab/chat-template contents). If a tokenizer's
        # contents change at the same name_or_path, clear cache_dir to avoid stale tokens.
        "tokenizer": getattr(tokenizer, "name_or_path", None),
    }
    return blake2b(
        json.dumps(payload, sort_keys=True).encode(), digest_size=8
    ).hexdigest()


def _build_mixture_entry(
    entry,
    tokenizer,
    max_prompt_length: Optional[int],
    max_completion_length: Optional[int],
    is_encoder_decoder: bool,
    num_proc: Optional[int],
    seed: int,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Build one mixture entry (tokenized or raw HF), applying its max_samples/selection cap.

    For ``hf`` entries the cap is applied to the raw rows BEFORE tokenization (so we only tokenize what
    we keep), and the tokenized subset is persisted to ``cache_dir`` (if set) so identical reruns skip
    re-tokenization. For ``tokenized`` entries the build is cheap (mmap) + already cached by
    ``Dataset.from_generator``, so the cap is applied after.
    """
    entry_type = entry["type"]
    if entry_type == "tokenized":
        ds = build_hf_dataset_from_tokenized(
            root=entry["path"],
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            tokenizer_consistency=entry.get("tokenizer_consistency", "warn"),
            accepted_prefix=entry.get("accepted_prefix", DEFAULT_ACCEPTED_PREFIX),
            rejected_prefix=entry.get("rejected_prefix", DEFAULT_REJECTED_PREFIX),
            parquet_path=entry.get("parquet_path", None),
            chosen_index_col=entry.get("chosen_index_col", DEFAULT_CHOSEN_INDEX_COL),
            rejected_index_col=entry.get(
                "rejected_index_col", DEFAULT_REJECTED_INDEX_COL
            ),
        )
        return _subselect(
            ds, entry.get("max_samples"), entry.get("selection", "random"), seed
        )
    elif entry_type == "hf":
        cache_path = None
        if cache_dir:
            key = _raw_cache_key(
                entry,
                tokenizer,
                max_prompt_length,
                max_completion_length,
                is_encoder_decoder,
                seed,
            )
            cache_path = os.path.join(cache_dir, f"raw-{key}")
            if os.path.isdir(cache_path):
                logger.info(
                    "Reusing cached tokenized raw mixture entry: %s", cache_path
                )
                return load_from_disk(cache_path)
        raw = _load_raw_for_mixture(entry, seed)
        ds = tokenize_raw_preference_dataset(
            raw,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            is_encoder_decoder=is_encoder_decoder,
            num_proc=num_proc,
        )
        if cache_path is not None and not os.path.isdir(cache_path):
            # Publish atomically: save to a temp dir then rename, so a crash mid-save can't leave a
            # partial cache_path that a later run would load as complete.
            os.makedirs(cache_dir, exist_ok=True)
            tmp_path = f"{cache_path}.tmp-{os.getpid()}"
            ds.save_to_disk(tmp_path)
            if os.path.isdir(cache_path):
                shutil.rmtree(
                    tmp_path, ignore_errors=True
                )  # another process won the race
            else:
                os.replace(tmp_path, cache_path)
                logger.info("Cached tokenized raw mixture entry to %s", cache_path)
        return ds
    raise ValueError(
        f"Unknown mixture entry type '{entry_type}'; expected 'tokenized' or 'hf'."
    )


def build_mixed_preference_dataset(
    mixture,
    tokenizer,
    max_prompt_length: Optional[int] = None,
    max_completion_length: Optional[int] = None,
    dataset_num_proc: Optional[int] = None,
    is_encoder_decoder: bool = False,
) -> Dataset:
    """Build a mixed preference dataset from several sources (via interleave or concatenate).

    ``mixture`` is a dict with: ``datasets`` (list of entries), ``seed``, ``combine``
    (``interleave`` (default) | ``concatenate``), and optional ``total_samples`` / ``cache_dir``. Each
    entry: ``type`` (``tokenized``|``hf``), ``path``, optional ``max_samples`` + ``selection``
    (``random``|``head``), plus type-specific fields (tokenized index columns / hf ``split``).

    - ``combine=interleave``: probabilistic mix via ``interleave_datasets`` using per-entry ``weight``
      (weights must sum to 1.0) and ``stopping_strategy`` (``all_exhausted`` default | ``first_exhausted``).
    - ``combine=concatenate``: take each entry fully (up to its ``max_samples``) and concatenate IN ORDER,
      then a final ``total_samples`` cap keeps the first N rows. List the source you want used FULLY first
      so the cap trims only from later entries — this is how you get "source A completely + fill with B
      up to N". The trainer applies the final shuffle, so order here only affects which rows the cap keeps.

    ``total_samples`` is a cap clamped to the available size. Returns a single uniformly-tokenized Dataset.
    """
    entries = list(mixture["datasets"])
    if not entries:
        raise ValueError("mixture.datasets is empty.")
    seed = int(mixture.get("seed", 42))
    combine = mixture.get("combine", "interleave")
    cache_dir = mixture.get("cache_dir", None)

    built = [
        _build_mixture_entry(
            e,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            is_encoder_decoder=is_encoder_decoder,
            num_proc=dataset_num_proc,
            seed=seed,
            cache_dir=cache_dir,
        )
        for e in entries
    ]

    # interleave_datasets / concatenate_datasets require identical features across inputs.
    target_features = built[0].features
    built = [ds.cast(target_features) for ds in built]

    total_samples = mixture.get("total_samples", None)
    if combine == "interleave":
        if any("weight" not in e for e in entries):
            raise ValueError(
                "interleave mixture requires a 'weight' on every entry (weights must sum to 1.0)."
            )
        weights = [float(e["weight"]) for e in entries]
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(
                f"mixture weights must sum to 1.0, got {sum(weights)} ({weights})."
            )
        stopping_strategy = mixture.get("stopping_strategy", "all_exhausted")
        combined = interleave_datasets(
            built, probabilities=weights, seed=seed, stopping_strategy=stopping_strategy
        )
        combine_info = f"interleave(weights={weights}, stopping={stopping_strategy})"
    elif combine == "concatenate":
        # The total_samples cap keeps the first N rows, so it only trims later entries. If it would cut
        # into the first entry, that entry is NOT used fully — warn (the "first source kept fully" rule).
        if total_samples is not None and total_samples < len(built[0]):
            logger.warning(
                "concatenate total_samples=%d < first entry size %d: the first source will be TRIMMED "
                "(not used fully). List the must-keep-fully source first with a large enough total_samples.",
                total_samples,
                len(built[0]),
            )
        combined = concatenate_datasets(built)
        combine_info = "concatenate"
    else:
        raise ValueError(
            f"Unknown mixture combine '{combine}'; expected 'interleave' or 'concatenate'."
        )

    if total_samples is not None:
        combined = combined.select(range(min(int(total_samples), len(combined))))

    logger.info(
        "Mixed preference dataset: per-source kept=%s, combine=%s, total=%d",
        [len(ds) for ds in built],
        combine_info,
        len(combined),
    )
    return combined
