from __future__ import annotations

from collections.abc import Callable, Sequence

from batch import keys as K
from batch.candidate_plan import OnlineRolloutRequest
from batch.source_schedule import SourceCounts
from data.schemas import OfflineTrajectoryCandidate, PromptRecord


OfflineSelector = Callable[[PromptRecord, int], Sequence[OfflineTrajectoryCandidate]]


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

    Args:
        prompt_records:
            Prompt-level dataset records.

        source_counts:
            One SourceCounts object per prompt, usually produced by
            FixedCountsSourceScheduler.plan(...).

        offline_selector:
            Function selecting offline candidates for one prompt.

            Contract:
                offline_selector(prompt_record, n_offline)
                -> Sequence[OfflineTrajectoryCandidate]

        actor_version:
            Used only for deterministic online trajectory ids.

    Returns:
        offline_candidates:
            Selected offline trajectories.

        online_requests:
            Online rollout requests to pass into
            online_rollout_requests_to_dataproto(...).
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
            selected_offline = list(offline_selector(prompt_record, counts.n_offline))

            if len(selected_offline) != counts.n_offline:
                raise ValueError(
                    f"offline_selector returned {len(selected_offline)} candidates for "
                    f"prompt_id={prompt_record.prompt_id!r}, expected {counts.n_offline}."
                )

            offline_candidates.extend(selected_offline)

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
                    tools=getattr(prompt_record, "tools", None),
                )
            )

    return offline_candidates, online_requests


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
