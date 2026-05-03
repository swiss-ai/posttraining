from __future__ import annotations

import random
from typing import Any, Mapping

from data.schemas import OfflineSelection, PromptRecord


class MinMaxRewardSelector:
    """Select highest- and lowest-reward offline trajectories.

    Order:
      1. max_reward trajectory
      2. min_reward trajectory
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.deduplicate_if_same_index = bool(config.get("deduplicate_if_same_index", True))

    def select(self, prompt: PromptRecord, *, n_offline: int | None = None) -> list[OfflineSelection]:
        if n_offline is not None:
            if n_offline < 0:
                raise ValueError(f"n_offline must be non-negative, got {n_offline}.")
            if n_offline == 0:
                return []

        num_trajectories = len(prompt.offline_trajectories)

        if num_trajectories == 0:
            raise ValueError(f"Prompt {prompt.prompt_id!r} has no offline trajectories.")

        if num_trajectories != len(prompt.offline_rewards):
            raise ValueError(
                f"Prompt {prompt.prompt_id!r} has {num_trajectories} offline trajectories "
                f"but {len(prompt.offline_rewards)} offline rewards."
            )

        max_index = max(range(num_trajectories), key=lambda i: prompt.offline_rewards[i])
        min_index = min(range(num_trajectories), key=lambda i: prompt.offline_rewards[i])

        selections = [
            OfflineSelection(
                offline_index=max_index,
                selection_reason="max_reward",
            )
        ]

        if min_index != max_index:
            selections.append(
                OfflineSelection(
                    offline_index=min_index,
                    selection_reason="min_reward",
                )
            )
        elif not self.deduplicate_if_same_index:
            selections.append(
                OfflineSelection(
                    offline_index=min_index,
                    selection_reason="min_reward",
                )
            )

        if n_offline is not None:
            if n_offline > len(selections):
                raise ValueError(
                    f"MinMaxRewardSelector can provide at most {len(selections)} selections "
                    f"for prompt {prompt.prompt_id!r}, but n_offline={n_offline} was requested."
                )

            selections = selections[:n_offline]

        return selections


class RandomOfflineSelector:
    """Select random offline trajectories.

    By default, samples without replacement, so each selected trajectory index
    appears at most once per prompt.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}

        seed = config.get("seed", None)
        self.rng = random.Random(None if seed is None else int(seed))

        self.replacement = bool(config.get("replacement", False))

        # Used only when select(..., n_offline=None). In the trainer path,
        # n_offline comes from the source scheduler.
        self.default_n_offline = int(config.get("default_n_offline", 1))
        if self.default_n_offline < 0:
            raise ValueError(
                f"default_n_offline must be non-negative, got {self.default_n_offline}."
            )

    def select(self, prompt: PromptRecord, *, n_offline: int | None = None) -> list[OfflineSelection]:
        if n_offline is None:
            n_offline = self.default_n_offline

        if n_offline < 0:
            raise ValueError(f"n_offline must be non-negative, got {n_offline}.")
        if n_offline == 0:
            return []

        num_trajectories = len(prompt.offline_trajectories)

        if num_trajectories == 0:
            raise ValueError(f"Prompt {prompt.prompt_id!r} has no offline trajectories.")

        if num_trajectories != len(prompt.offline_rewards):
            raise ValueError(
                f"Prompt {prompt.prompt_id!r} has {num_trajectories} offline trajectories "
                f"but {len(prompt.offline_rewards)} offline rewards."
            )

        if self.replacement:
            indices = [
                self.rng.randrange(num_trajectories)
                for _ in range(n_offline)
            ]
        else:
            if n_offline > num_trajectories:
                raise ValueError(
                    f"RandomOfflineSelector cannot sample {n_offline} trajectories "
                    f"without replacement for prompt {prompt.prompt_id!r}, which has only "
                    f"{num_trajectories} offline trajectories."
                )

            indices = self.rng.sample(range(num_trajectories), k=n_offline)

        return [
            OfflineSelection(
                offline_index=index,
                selection_reason="random",
            )
            for index in indices
        ]


def build_offline_selector(config: Mapping[str, Any] | None = None):
    config = config or {}
    name = str(config.get("name", "minmax_rewards"))

    if name == "minmax_rewards":
        return MinMaxRewardSelector(config)

    if name == "random":
        return RandomOfflineSelector(config)

    raise ValueError(
        f"Unknown offline selector {name!r}. Expected one of "
        "'minmax_rewards' or 'random'."
    )
