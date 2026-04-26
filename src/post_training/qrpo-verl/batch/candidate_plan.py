from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from batch.source_schedule import SourceCounts, build_rollout_prompt_indices
from data.offline_selector import MinMaxRewardSelector
from data.schemas import ChatMessage, OfflineSelection, OfflineTrajectoryCandidate, PromptRecord


@dataclass(frozen=True)
class OnlineRolloutRequest:
    """One expanded prompt request for online generation.

    CPU metadata only. Dense online rollout tokens should stay in VERL/DataProto
    tensors later.
    """

    prompt_index: int
    prompt_id: str
    online_index: int

    prompt_messages: tuple[ChatMessage, ...]
    ref_rewards: tuple[float, ...]
    trajectory_id: str

    tools: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class CandidatePlan:
    """Pure Python plan for constructing one QRPO training batch."""

    prompts: tuple[PromptRecord, ...]
    source_counts: tuple[SourceCounts, ...]

    online_requests: tuple[OnlineRolloutRequest, ...]
    offline_candidates: tuple[OfflineTrajectoryCandidate, ...]

    actor_version: int | None
    ref_version: int | None

    @property
    def rollout_prompt_indices(self) -> list[int]:
        return build_rollout_prompt_indices(self.source_counts)

    @property
    def num_online_requests(self) -> int:
        return len(self.online_requests)

    @property
    def num_offline_candidates(self) -> int:
        return len(self.offline_candidates)


def build_candidate_plan(
    *,
    prompts: Sequence[PromptRecord],
    source_counts: Sequence[SourceCounts],
    offline_selector: MinMaxRewardSelector,
    actor_version: int | None,
    ref_version: int | None,
) -> CandidatePlan:
    prompts_tuple = tuple(prompts)
    counts_tuple = tuple(source_counts)

    _validate_source_counts(prompts_tuple, counts_tuple)

    return CandidatePlan(
        prompts=prompts_tuple,
        source_counts=counts_tuple,
        online_requests=tuple(
            _build_online_requests(
                prompts=prompts_tuple,
                source_counts=counts_tuple,
                actor_version=actor_version,
            )
        ),
        offline_candidates=tuple(
            _build_offline_candidates(
                prompts=prompts_tuple,
                source_counts=counts_tuple,
                offline_selector=offline_selector,
                ref_version=ref_version,
            )
        ),
        actor_version=actor_version,
        ref_version=ref_version,
    )


def _validate_source_counts(
    prompts: tuple[PromptRecord, ...],
    source_counts: tuple[SourceCounts, ...],
) -> None:
    if len(prompts) != len(source_counts):
        raise ValueError(
            f"Got {len(prompts)} prompts but {len(source_counts)} source-count entries."
        )

    for i, (prompt, counts) in enumerate(zip(prompts, source_counts, strict=True)):
        if counts.prompt_index != i:
            raise ValueError(
                f"SourceCounts for prompt {prompt.prompt_id!r} has prompt_index="
                f"{counts.prompt_index}, expected {i}."
            )

        if counts.prompt_id != prompt.prompt_id:
            raise ValueError(
                f"SourceCounts prompt_id={counts.prompt_id!r} does not match "
                f"PromptRecord prompt_id={prompt.prompt_id!r} at index {i}."
            )

        if counts.n_online < 0:
            raise ValueError(f"Prompt {prompt.prompt_id!r} has negative n_online.")

        if counts.n_offline < 0:
            raise ValueError(f"Prompt {prompt.prompt_id!r} has negative n_offline.")

        if counts.n_online + counts.n_offline <= 0:
            raise ValueError(f"Prompt {prompt.prompt_id!r} has zero requested completions.")


def _build_online_requests(
    *,
    prompts: tuple[PromptRecord, ...],
    source_counts: tuple[SourceCounts, ...],
    actor_version: int | None,
) -> list[OnlineRolloutRequest]:
    requests: list[OnlineRolloutRequest] = []

    for counts in source_counts:
        prompt = prompts[counts.prompt_index]

        for online_index in range(counts.n_online):
            requests.append(
                OnlineRolloutRequest(
                    prompt_index=counts.prompt_index,
                    prompt_id=prompt.prompt_id,
                    online_index=online_index,
                    prompt_messages=prompt.prompt_messages,
                    ref_rewards=prompt.ref_rewards,
                    trajectory_id=_make_online_trajectory_id(
                        prompt_id=prompt.prompt_id,
                        online_index=online_index,
                        actor_version=actor_version,
                    ),
                    tools=prompt.tools,
                )
            )

    return requests


def _build_offline_candidates(
    *,
    prompts: tuple[PromptRecord, ...],
    source_counts: tuple[SourceCounts, ...],
    offline_selector: MinMaxRewardSelector,
    ref_version: int | None,
) -> list[OfflineTrajectoryCandidate]:
    candidates: list[OfflineTrajectoryCandidate] = []

    for counts in source_counts:
        prompt = prompts[counts.prompt_index]

        selections = offline_selector.select(prompt, n_offline=counts.n_offline)

        for selection in selections:
            candidates.append(
                _offline_selection_to_candidate(
                    prompt=prompt,
                    selection=selection,
                    ref_version=ref_version,
                )
            )

    return candidates


def _offline_selection_to_candidate(
    *,
    prompt: PromptRecord,
    selection: OfflineSelection,
    ref_version: int | None,
) -> OfflineTrajectoryCandidate:
    offline_index = selection.offline_index

    return OfflineTrajectoryCandidate(
        prompt_id=prompt.prompt_id,
        trajectory_id=_make_offline_trajectory_id(
            prompt_id=prompt.prompt_id,
            offline_index=offline_index,
        ),
        prompt_messages=prompt.prompt_messages,
        trajectory_messages=prompt.offline_trajectories[offline_index],
        reward=float(prompt.offline_rewards[offline_index]),
        ref_rewards=prompt.ref_rewards,
        tools=prompt.tools,
        offline_index=offline_index,
        ref_version=ref_version,
        metadata={"selection_reason": selection.selection_reason},
    )


def _make_online_trajectory_id(
    *,
    prompt_id: str,
    online_index: int,
    actor_version: int | None,
) -> str:
    if actor_version is None:
        return f"{prompt_id}::online::{online_index}"
    return f"{prompt_id}::online::actor_{actor_version}::{online_index}"


def _make_offline_trajectory_id(
    *,
    prompt_id: str,
    offline_index: int,
) -> str:
    return f"{prompt_id}::offline::{offline_index}"
