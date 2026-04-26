from __future__ import annotations

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
