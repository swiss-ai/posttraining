from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ChatMessage = Mapping[str, Any]


def normalize_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
) -> tuple[dict[str, Any], ...]:
    """Validate and copy chat messages.

    Use this during dataset validation/tests, not repeatedly in the hot path.
    """

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError(f"{field_name} must be a sequence of message dictionaries.")

    normalized: list[dict[str, Any]] = []

    for i, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"{field_name}[{i}] must be a mapping, got {type(message).__name__}."
            )

        if "role" not in message:
            raise ValueError(f"{field_name}[{i}] is missing required key 'role'.")

        role = message["role"]
        if not isinstance(role, str) or not role:
            raise ValueError(f"{field_name}[{i}]['role'] must be a non-empty string.")

        normalized.append(dict(message))

    return tuple(normalized)


@dataclass(frozen=True)
class PromptRecord:
    """One prompt-centric dataset row."""

    prompt_id: str
    prompt_messages: tuple[ChatMessage, ...]

    ref_rewards: tuple[float, ...]

    offline_trajectories: tuple[tuple[ChatMessage, ...], ...]
    offline_rewards: tuple[float, ...]

    tools: tuple[Mapping[str, Any], ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        num_ref_rewards: int | None = None,
        deep: bool = True,
    ) -> None:
        if not self.prompt_id:
            raise ValueError("PromptRecord.prompt_id must be non-empty.")

        if deep:
            normalize_messages(self.prompt_messages, field_name="prompt_messages")

        if len(self.ref_rewards) == 0:
            raise ValueError(f"Prompt {self.prompt_id!r} has no reference rewards.")

        if num_ref_rewards is not None and len(self.ref_rewards) != num_ref_rewards:
            raise ValueError(
                f"Prompt {self.prompt_id!r} has {len(self.ref_rewards)} ref rewards, "
                f"expected {num_ref_rewards}."
            )

        for reward in self.ref_rewards:
            if not math.isfinite(float(reward)):
                raise ValueError(
                    f"Prompt {self.prompt_id!r} contains non-finite ref reward: {reward!r}."
                )

        if len(self.offline_trajectories) != len(self.offline_rewards):
            raise ValueError(
                f"Prompt {self.prompt_id!r} has {len(self.offline_trajectories)} "
                f"offline trajectories but {len(self.offline_rewards)} offline rewards."
            )

        if deep:
            for i, trajectory in enumerate(self.offline_trajectories):
                if len(trajectory) == 0:
                    raise ValueError(
                        f"Prompt {self.prompt_id!r} offline trajectory {i} is empty."
                    )
                normalize_messages(trajectory, field_name=f"offline_trajectories[{i}]")

        for i, reward in enumerate(self.offline_rewards):
            if not math.isfinite(float(reward)):
                raise ValueError(
                    f"Prompt {self.prompt_id!r} offline reward {i} is non-finite: {reward!r}."
                )


@dataclass(frozen=True)
class OfflineSelection:
    """Index-only selection result.

    The selected trajectory and reward remain in PromptRecord.
    """

    offline_index: int
    selection_reason: str


@dataclass(frozen=True)
class OfflineTrajectoryCandidate:
    """Selected offline trajectory before tokenization.

    CPU metadata only. No dense token tensors here.
    """

    prompt_id: str
    trajectory_id: str
    prompt_messages: tuple[ChatMessage, ...]
    trajectory_messages: tuple[ChatMessage, ...]

    reward: float
    ref_rewards: tuple[float, ...]

    tools: tuple[Mapping[str, Any], ...] | None = None
    offline_index: int | None = None
    ref_version: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, deep: bool = True) -> None:
        if not self.prompt_id:
            raise ValueError("OfflineTrajectoryCandidate.prompt_id must be non-empty.")

        if not self.trajectory_id:
            raise ValueError("OfflineTrajectoryCandidate.trajectory_id must be non-empty.")

        if len(self.trajectory_messages) == 0:
            raise ValueError(
                f"Offline candidate {self.trajectory_id!r} has empty trajectory_messages."
            )

        if not math.isfinite(float(self.reward)):
            raise ValueError(
                f"Offline candidate {self.trajectory_id!r} has non-finite reward: {self.reward!r}."
            )

        if len(self.ref_rewards) == 0:
            raise ValueError(f"Offline candidate {self.trajectory_id!r} has no ref_rewards.")

        if deep:
            normalize_messages(self.prompt_messages, field_name="prompt_messages")
            normalize_messages(self.trajectory_messages, field_name="trajectory_messages")
