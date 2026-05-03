from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from batch import keys as K
from batch.candidate_plan import OnlineRolloutRequest
from batch.source_schedule import SourceCounts
from data.schemas import OfflineSelection, OfflineTrajectoryCandidate, PromptRecord


class OfflineSelectorProtocol(Protocol):
    def select(
        self,
        prompt: PromptRecord,
        *,
        n_offline: int | None = None,
    ) -> Sequence[OfflineSelection]:
        ...


OfflineSelector = Callable[[PromptRecord, int], Sequence[OfflineSelection]] | OfflineSelectorProtocol


def build_training_candidates(
    *,
    prompt_records: Sequence[PromptRecord],
    source_counts: Sequence[SourceCounts],
    offline_selector: OfflineSelector,
    actor_version: str | int,
) -> tuple[list[OfflineTrajectoryCandidate], list[OnlineRolloutRequest]]:
    """Build offline candidates and online rollout requests from source counts.

    This bridges:

      source scheduler + offline selector
      ->
      offline candidates + online rollout requests

    It intentionally does not tokenize, generate, score, concatenate, or train.

    The offline selector returns OfflineSelection objects. This function turns
    them into OfflineTrajectoryCandidate objects by reading the selected
    trajectory/reward from the corresponding PromptRecord.
    """

    prompt_records = tuple(prompt_records)
    source_counts = tuple(source_counts)

    if len(prompt_records) != len(source_counts):
        raise ValueError(
            f"prompt_records and source_counts must have the same length, got "
            f"{len(prompt_records)} and {len(source_counts)}."
        )

    offline_candidates: list[OfflineTrajectoryCandidate] = []
    online_requests: list[OnlineRolloutRequest] = []

    for prompt_record, counts in zip(prompt_records, source_counts, strict=True):
        _validate_counts_match_prompt(prompt_record, counts)

        if counts.n_offline:
            selections = _select_offline(
                offline_selector=offline_selector,
                prompt=prompt_record,
                n_offline=counts.n_offline,
            )

            if len(selections) != counts.n_offline:
                raise ValueError(
                    f"offline_selector returned {len(selections)} selections for "
                    f"prompt_id={prompt_record.prompt_id!r}, expected {counts.n_offline}."
                )

            for offline_slot, selection in enumerate(selections):
                offline_candidates.append(
                    _offline_selection_to_candidate(
                        prompt=prompt_record,
                        selection=selection,
                        offline_slot=offline_slot,
                    )
                )

        for online_index in range(counts.n_online):
            online_requests.append(
                OnlineRolloutRequest(
                    prompt_index=counts.prompt_index,
                    prompt_id=prompt_record.prompt_id,
                    online_index=online_index,
                    prompt_messages=prompt_record.prompt_messages,
                    ref_rewards=prompt_record.ref_rewards,
                    trajectory_id=_online_trajectory_id(
                        prompt_id=prompt_record.prompt_id,
                        actor_version=actor_version,
                        online_index=online_index,
                    ),
                    tools=prompt_record.tools,
                )
            )

    return offline_candidates, online_requests


def _select_offline(
    *,
    offline_selector: OfflineSelector,
    prompt: PromptRecord,
    n_offline: int,
) -> list[OfflineSelection]:
    if hasattr(offline_selector, "select"):
        return list(offline_selector.select(prompt, n_offline=n_offline))

    return list(offline_selector(prompt, n_offline))


def _offline_selection_to_candidate(
    *,
    prompt: PromptRecord,
    selection: OfflineSelection,
    offline_slot: int,
) -> OfflineTrajectoryCandidate:
    offline_index = int(selection.offline_index)

    if offline_index < 0 or offline_index >= len(prompt.offline_trajectories):
        raise IndexError(
            f"Offline selection index {offline_index} is out of range for "
            f"prompt_id={prompt.prompt_id!r} with {len(prompt.offline_trajectories)} "
            "offline trajectories."
        )

    if offline_index >= len(prompt.offline_rewards):
        raise IndexError(
            f"Offline selection index {offline_index} has no reward for "
            f"prompt_id={prompt.prompt_id!r}; only {len(prompt.offline_rewards)} "
            "offline rewards are available."
        )

    return OfflineTrajectoryCandidate(
        prompt_id=prompt.prompt_id,
        trajectory_id=_offline_trajectory_id(
            prompt_id=prompt.prompt_id,
            offline_slot=offline_slot,
            offline_index=offline_index,
            selection_reason=selection.selection_reason,
        ),
        prompt_messages=prompt.prompt_messages,
        trajectory_messages=prompt.offline_trajectories[offline_index],
        reward=float(prompt.offline_rewards[offline_index]),
        ref_rewards=prompt.ref_rewards,
        tools=prompt.tools,
    )


def _validate_counts_match_prompt(
    prompt_record: PromptRecord,
    counts: SourceCounts,
) -> None:
    if counts.prompt_id != prompt_record.prompt_id:
        raise ValueError(
            f"SourceCounts prompt_id={counts.prompt_id!r} does not match "
            f"PromptRecord prompt_id={prompt_record.prompt_id!r}."
        )

    if counts.n_online < 0:
        raise ValueError(
            f"SourceCounts for prompt_id={counts.prompt_id!r} has negative "
            f"n_online={counts.n_online}."
        )

    if counts.n_offline < 0:
        raise ValueError(
            f"SourceCounts for prompt_id={counts.prompt_id!r} has negative "
            f"n_offline={counts.n_offline}."
        )

    if counts.n_online + counts.n_offline <= 0:
        raise ValueError(
            f"SourceCounts for prompt_id={counts.prompt_id!r} requests no samples."
        )


def _online_trajectory_id(
    *,
    prompt_id: str,
    actor_version: str | int,
    online_index: int,
) -> str:
    return f"{prompt_id}::{K.SOURCE_ONLINE}::actor_{actor_version}::{online_index}"


def _offline_trajectory_id(
    *,
    prompt_id: str,
    offline_slot: int,
    offline_index: int,
    selection_reason: str,
) -> str:
    return (
        f"{prompt_id}::{K.SOURCE_OFFLINE}::slot_{offline_slot}"
        f"::idx_{offline_index}::{selection_reason}"
    )
