from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from data.schemas import PromptRecord


@dataclass(frozen=True)
class SourceCounts:
    """Number of online/offline completions requested for one prompt."""

    prompt_index: int
    prompt_id: str
    n_online: int
    n_offline: int


class SourceScheduler(Protocol):
    """Assign online/offline training counts to prompt records."""

    @property
    def can_emit_online(self) -> bool:
        ...

    @property
    def can_emit_offline(self) -> bool:
        ...

    def plan(
        self,
        prompts: Sequence[PromptRecord],
        *,
        global_step: int | None = None,
        online_count_divisible_by: int = 1,
    ) -> list[SourceCounts]:
        ...


class FixedCountsSourceScheduler:
    """Assign the same online/offline counts to every prompt."""

    def __init__(self, *, n_online: int, n_offline: int) -> None:
        if n_online < 0:
            raise ValueError(f"n_online must be non-negative, got {n_online}.")
        if n_offline < 0:
            raise ValueError(f"n_offline must be non-negative, got {n_offline}.")
        if n_online + n_offline <= 0:
            raise ValueError("At least one of n_online or n_offline must be positive.")

        self.n_online = int(n_online)
        self.n_offline = int(n_offline)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FixedCountsSourceScheduler":
        return cls(
            n_online=int(config.get("n_online", 0)),
            n_offline=int(config.get("n_offline", 0)),
        )

    @property
    def can_emit_online(self) -> bool:
        return self.n_online > 0

    @property
    def can_emit_offline(self) -> bool:
        return self.n_offline > 0

    def plan(
        self,
        prompts: Sequence[PromptRecord],
        *,
        global_step: int | None = None,
        online_count_divisible_by: int = 1,
    ) -> list[SourceCounts]:
        return [
            SourceCounts(
                prompt_index=i,
                prompt_id=prompt.prompt_id,
                n_online=self.n_online,
                n_offline=self.n_offline,
            )
            for i, prompt in enumerate(prompts)
        ]


class SingleCompletionMixtureSourceScheduler:
    """Choose exactly one training completion source per prompt.

    Each prompt receives either one offline completion or one online completion.
    The offline/online choice is deterministic from seed, global step, prompt
    index, and prompt id, so checkpoint resume does not require scheduler RNG
    state. When AgentLoop requires a divisible online request count, the
    scheduler uses deterministic probabilistic rounding to preserve the expected
    offline fraction across steps.
    """

    def __init__(self, *, offline_probability: float, seed: int = 0) -> None:
        offline_probability = float(offline_probability)
        if not 0.0 <= offline_probability <= 1.0:
            raise ValueError(
                "offline_probability must be in [0, 1], got "
                f"{offline_probability}."
            )

        self.offline_probability = offline_probability
        self.seed = int(seed)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "SingleCompletionMixtureSourceScheduler":
        return cls(
            offline_probability=float(config.get("offline_probability", 0.5)),
            seed=int(config.get("seed", 0)),
        )

    @property
    def can_emit_online(self) -> bool:
        return self.offline_probability < 1.0

    @property
    def can_emit_offline(self) -> bool:
        return self.offline_probability > 0.0

    def plan(
        self,
        prompts: Sequence[PromptRecord],
        *,
        global_step: int | None = None,
        online_count_divisible_by: int = 1,
    ) -> list[SourceCounts]:
        if global_step is None:
            global_step = 0
        if online_count_divisible_by <= 0:
            raise ValueError(
                "online_count_divisible_by must be positive, got "
                f"{online_count_divisible_by}."
            )

        scores: list[tuple[float, int]] = []
        for i, prompt in enumerate(prompts):
            score = _stable_unit_interval(
                seed=self.seed,
                global_step=int(global_step),
                prompt_index=i,
                prompt_id=prompt.prompt_id,
            )
            scores.append((score, i))

        offline_count = _target_offline_count(
            prompt_count=len(prompts),
            offline_probability=self.offline_probability,
            online_count_divisible_by=online_count_divisible_by,
            seed=self.seed,
            global_step=int(global_step),
        )
        offline_indices = {index for _, index in sorted(scores)[:offline_count]}

        online_count = len(prompts) - offline_count
        if online_count % online_count_divisible_by != 0:
            raise ValueError(
                "single_completion_mixture produced an online request count "
                "that is not divisible by online_count_divisible_by: "
                f"{online_count} % {online_count_divisible_by} != 0."
            )

        counts: list[SourceCounts] = []
        for i, prompt in enumerate(prompts):
            is_offline = i in offline_indices
            counts.append(
                SourceCounts(
                    prompt_index=i,
                    prompt_id=prompt.prompt_id,
                    n_online=0 if is_offline else 1,
                    n_offline=1 if is_offline else 0,
                )
            )

        return counts


def build_source_scheduler(config: Mapping[str, Any]) -> SourceScheduler:
    name = str(config.get("name", "fixed_counts"))

    if name == "fixed_counts":
        return FixedCountsSourceScheduler.from_config(config)

    if name == "single_completion_mixture":
        return SingleCompletionMixtureSourceScheduler.from_config(config)

    raise ValueError(
        f"Unknown source schedule {name!r}. Expected one of "
        "'fixed_counts' or 'single_completion_mixture'."
    )


def _stable_unit_interval(
    *,
    seed: int,
    global_step: int,
    prompt_index: int,
    prompt_id: str,
) -> float:
    key = f"{seed}\0{global_step}\0{prompt_index}\0{prompt_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
    return value / float(1 << 64)


def _target_offline_count(
    *,
    prompt_count: int,
    offline_probability: float,
    online_count_divisible_by: int,
    seed: int,
    global_step: int,
) -> int:
    """Return an offline count whose complementary online count is valid."""

    if prompt_count <= 0:
        return 0
    if offline_probability <= 0.0:
        return 0
    if offline_probability >= 1.0:
        return prompt_count

    target = offline_probability * prompt_count
    valid_counts = [
        offline_count
        for offline_count in range(prompt_count + 1)
        if (prompt_count - offline_count) % online_count_divisible_by == 0
    ]
    lower = max(
        (count for count in valid_counts if count <= target),
        default=valid_counts[0],
    )
    upper = min(
        (count for count in valid_counts if count >= target),
        default=valid_counts[-1],
    )

    if lower == upper:
        return lower

    upper_probability = (target - lower) / (upper - lower)
    score = _stable_unit_interval(
        seed=seed,
        global_step=global_step,
        prompt_index=-1,
        prompt_id="single_completion_mixture.offline_count",
    )
    return upper if score < upper_probability else lower


def build_rollout_prompt_indices(source_counts: Sequence[SourceCounts]) -> list[int]:
    """
    Return prompt indices expanded according to requested online completions.

    Example:
        prompt 0 needs 2 online completions
        prompt 1 needs 0 online completions
        prompt 2 needs 1 online completion

        returns [0, 0, 2]
    """

    rollout_indices: list[int] = []

    for counts in source_counts:
        if counts.n_online < 0:
            raise ValueError(
                f"Prompt {counts.prompt_id!r} has negative n_online={counts.n_online}."
            )

        rollout_indices.extend([counts.prompt_index] * counts.n_online)

    return rollout_indices
