from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from data.schemas import PromptRecord


@dataclass(frozen=True)
class SourceCounts:
    """Number of online/offline completions requested for one prompt."""

    prompt_index: int
    prompt_id: str
    n_online: int
    n_offline: int


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

    def plan(self, prompts: Sequence[PromptRecord]) -> list[SourceCounts]:
        return [
            SourceCounts(
                prompt_index=i,
                prompt_id=prompt.prompt_id,
                n_online=self.n_online,
                n_offline=self.n_offline,
            )
            for i, prompt in enumerate(prompts)
        ]


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
