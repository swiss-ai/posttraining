from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
VLLM_RE = re.compile(
    r"(?:INFO\s+)?(?P<month>\d{2})-(?P<day>\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}).*?"
    r"Engine\s+(?P<engine>\d+):\s+"
    r"Avg prompt throughput:\s+(?P<prompt_tps>[\d.]+)\s+tokens/s,\s+"
    r"Avg generation throughput:\s+(?P<generation_tps>[\d.]+)\s+tokens/s,\s+"
    r"Running:\s+(?P<running>\d+)\s+reqs,\s+"
    r"Waiting:\s+(?P<waiting>\d+)\s+reqs,\s+"
    r"GPU KV cache usage:\s+(?P<kv_cache_usage>[\d.]+)%,\s+"
    r"Prefix cache hit rate:\s+(?P<prefix_cache_hit_rate>[\d.]+)%"
)
ROUTER_FINISH_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*?"
    r"http_request.*?"
    r"method(?:\x1b\[[0-9;]*[A-Za-z])?=POST.*?"
    r"uri(?:\x1b\[[0-9;]*[A-Za-z])?=/v1/chat/completions.*?"
    r"request_id(?:\x1b\[[0-9;]*[A-Za-z])?=\"(?P<request_id>[^\"]+)\".*?"
    r"status_code(?:\x1b\[[0-9;]*[A-Za-z])?=(?P<status_code>\d+).*?"
    r"latency(?:\x1b\[[0-9;]*[A-Za-z])?=(?P<latency_us>\d+)"
)
TQDM_RE = re.compile(
    r"Generating Initial Ref Rewards:\s+"
    r"(?P<percent>\d+)%\|.*?\|\s+"
    r"(?P<current>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s+"
    r"(?P<rate>[\d.]+)(?P<unit>it/s|s/it)"
)


@dataclass(frozen=True)
class VLLMSample:
    source: str
    timestamp: datetime | None
    prompt_tps: float
    generation_tps: float
    running: int
    waiting: int
    kv_cache_usage_pct: float
    prefix_cache_hit_rate_pct: float


@dataclass(frozen=True)
class RouterSample:
    timestamp: datetime
    request_id: str
    status_code: int
    latency_s: float


@dataclass(frozen=True)
class ProgressSample:
    current: int
    total: int
    elapsed_s: float | None
    eta_s: float | None
    seconds_per_chunk: float | None


def main() -> None:
    args = parse_args()

    trainer_logs = [Path(path) for path in args.trainer_log]
    judge_log_dir = Path(args.judge_log_dir) if args.judge_log_dir else None
    ref_store = Path(args.ref_store) if args.ref_store else None

    report = {
        "inputs": {
            "trainer_logs": [str(path) for path in trainer_logs],
            "judge_log_dir": str(judge_log_dir) if judge_log_dir else None,
            "ref_store": str(ref_store) if ref_store else None,
            "ref_version": args.ref_version,
            "since": args.since,
            "until": args.until,
            "recent_router_lines": args.recent_router_lines,
            "recent_worker_lines": args.recent_worker_lines,
        },
        "trainer_progress": summarize_progress(
            collect_progress_samples(trainer_logs)
        ),
        "trainer_vllm": summarize_vllm(
            filter_vllm_samples(
                collect_vllm_samples(
                    trainer_logs,
                    source_prefix="trainer",
                    recent_lines=args.recent_worker_lines,
                ),
                since=parse_datetime_arg(args.since),
                until=parse_datetime_arg(args.until),
            )
        ),
        "judge_vllm": summarize_vllm(
            filter_vllm_samples(
                collect_judge_vllm_samples(
                    judge_log_dir,
                    recent_lines=args.recent_worker_lines,
                ),
                since=parse_datetime_arg(args.since),
                until=parse_datetime_arg(args.until),
            )
        ),
        "judge_router": summarize_router(
            filter_router_samples(
                collect_router_samples(
                    judge_log_dir,
                    recent_lines=args.recent_router_lines,
                ),
                since=parse_datetime_arg(args.since),
                until=parse_datetime_arg(args.until),
            )
        ),
        "ref_store": summarize_ref_store(
            ref_store,
            ref_version=args.ref_version,
            recent_chunks=args.recent_chunks,
        ),
    }

    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse QRPO ref-reward/trainer logs and judge logs into efficiency "
            "and saturation metrics."
        )
    )
    parser.add_argument(
        "--trainer-log",
        action="append",
        default=[],
        help="Trainer stdout/stderr log path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--judge-log-dir",
        default=None,
        help="Judge serving log directory containing router_*.out and worker*.out.",
    )
    parser.add_argument(
        "--ref-store",
        default=None,
        help="RefRewardStore directory to inspect chunk completion and mtimes.",
    )
    parser.add_argument(
        "--ref-version",
        default="ref_step_000000",
        help="Ref reward version to inspect inside the ref store.",
    )
    parser.add_argument(
        "--recent-chunks",
        type=int,
        default=20,
        help="Number of latest chunk-save intervals used for recent cadence.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "Only include timestamped judge/vLLM samples at or after this time. "
            "Use 'YYYY-MM-DD HH:MM:SS' or ISO format."
        ),
    )
    parser.add_argument(
        "--until",
        default=None,
        help=(
            "Only include timestamped judge/vLLM samples before or at this time. "
            "Use 'YYYY-MM-DD HH:MM:SS' or ISO format."
        ),
    )
    parser.add_argument(
        "--recent-router-lines",
        type=int,
        default=None,
        help="Parse only the last N lines from each router log.",
    )
    parser.add_argument(
        "--recent-worker-lines",
        type=int,
        default=None,
        help="Parse only the last N lines from each vLLM worker/trainer log.",
    )
    return parser.parse_args()


def collect_vllm_samples(
    paths: Sequence[Path],
    *,
    source_prefix: str,
    recent_lines: int | None,
) -> list[VLLMSample]:
    samples: list[VLLMSample] = []
    for path in progress(paths, desc="Parsing trainer vLLM logs"):
        if not path.exists():
            continue
        for line in iter_log_lines(path, recent_lines=recent_lines):
            sample = parse_vllm_sample(
                line,
                source=f"{source_prefix}:{path.name}",
            )
            if sample is not None:
                samples.append(sample)
    return samples


def collect_judge_vllm_samples(
    judge_log_dir: Path | None,
    *,
    recent_lines: int | None,
) -> list[VLLMSample]:
    if judge_log_dir is None or not judge_log_dir.exists():
        return []

    samples: list[VLLMSample] = []
    worker_logs = sorted(judge_log_dir.glob("worker*_node0_*.out"))
    for path in progress(worker_logs, desc="Parsing judge vLLM logs"):
        for line in iter_log_lines(path, recent_lines=recent_lines):
            sample = parse_vllm_sample(line, source=path.name)
            if sample is not None:
                samples.append(sample)
    return samples


def collect_router_samples(
    judge_log_dir: Path | None,
    *,
    recent_lines: int | None,
) -> list[RouterSample]:
    if judge_log_dir is None or not judge_log_dir.exists():
        return []

    samples: list[RouterSample] = []
    router_logs = sorted(judge_log_dir.glob("router_*.out"))
    for path in router_logs:
        for raw_line in iter_log_lines(
            path,
            recent_lines=recent_lines,
            progress_desc=f"Parsing judge router {path.name}",
        ):
            line = strip_ansi(raw_line)
            match = ROUTER_FINISH_RE.search(line)
            if match is None:
                continue
            samples.append(
                RouterSample(
                    timestamp=datetime.strptime(
                        match.group("ts"),
                        "%Y-%m-%d %H:%M:%S",
                    ),
                    request_id=match.group("request_id"),
                    status_code=int(match.group("status_code")),
                    latency_s=int(match.group("latency_us")) / 1_000_000.0,
                )
            )
    return samples


def collect_progress_samples(paths: Sequence[Path]) -> list[ProgressSample]:
    samples: list[ProgressSample] = []
    for path in progress(paths, desc="Parsing trainer progress logs"):
        if not path.exists():
            continue
        text = strip_ansi(path.read_text(errors="replace"))
        for match in TQDM_RE.finditer(text):
            rate = float(match.group("rate"))
            unit = match.group("unit")
            seconds_per_chunk = (1.0 / rate) if unit == "it/s" else rate
            samples.append(
                ProgressSample(
                    current=int(match.group("current")),
                    total=int(match.group("total")),
                    elapsed_s=parse_duration(match.group("elapsed")),
                    eta_s=parse_duration(match.group("eta")),
                    seconds_per_chunk=seconds_per_chunk,
                )
            )
    return samples


def parse_vllm_sample(line: str, *, source: str) -> VLLMSample | None:
    line = strip_ansi(line)
    match = VLLM_RE.search(line)
    if match is None:
        return None

    timestamp = parse_month_day_timestamp(
        int(match.group("month")),
        int(match.group("day")),
        match.group("time"),
    )
    return VLLMSample(
        source=source,
        timestamp=timestamp,
        prompt_tps=float(match.group("prompt_tps")),
        generation_tps=float(match.group("generation_tps")),
        running=int(match.group("running")),
        waiting=int(match.group("waiting")),
        kv_cache_usage_pct=float(match.group("kv_cache_usage")),
        prefix_cache_hit_rate_pct=float(match.group("prefix_cache_hit_rate")),
    )


def summarize_vllm(samples: Sequence[VLLMSample]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}

    by_source: dict[str, list[VLLMSample]] = {}
    for sample in samples:
        by_source.setdefault(sample.source, []).append(sample)

    return {
        "samples": len(samples),
        "sources": {
            source: summarize_vllm_group(group)
            for source, group in sorted(by_source.items())
        },
        "aggregate": summarize_vllm_group(samples),
    }


def summarize_vllm_group(samples: Sequence[VLLMSample]) -> dict[str, Any]:
    waiting = [sample.waiting for sample in samples]
    running = [sample.running for sample in samples]
    prompt_tps = [sample.prompt_tps for sample in samples]
    generation_tps = [sample.generation_tps for sample in samples]
    kv_cache = [sample.kv_cache_usage_pct for sample in samples]
    prefix_cache = [sample.prefix_cache_hit_rate_pct for sample in samples]

    return {
        "samples": len(samples),
        "time_start": format_dt(min_dt(sample.timestamp for sample in samples)),
        "time_end": format_dt(max_dt(sample.timestamp for sample in samples)),
        "prompt_tps": describe(prompt_tps),
        "generation_tps": describe(generation_tps),
        "running_reqs": describe(running),
        "waiting_reqs": describe(waiting),
        "kv_cache_usage_pct": describe(kv_cache),
        "prefix_cache_hit_rate_pct": describe(prefix_cache),
        "saturation_fraction_waiting_gt_0": fraction(value > 0 for value in waiting),
        "heavy_queue_fraction_waiting_ge_100": fraction(value >= 100 for value in waiting),
        "idle_fraction_running_eq_0_waiting_eq_0": fraction(
            sample.running == 0 and sample.waiting == 0 for sample in samples
        ),
    }


def summarize_router(samples: Sequence[RouterSample]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}

    latencies = [sample.latency_s for sample in samples]
    status_counts = Counter(sample.status_code for sample in samples)
    timestamps = [sample.timestamp for sample in samples]
    duration_s = max((max(timestamps) - min(timestamps)).total_seconds(), 0.0)

    return {
        "samples": len(samples),
        "time_start": format_dt(min(timestamps)),
        "time_end": format_dt(max(timestamps)),
        "duration_s": duration_s,
        "requests_per_s": (len(samples) / duration_s) if duration_s > 0 else None,
        "status_counts": dict(sorted(status_counts.items())),
        "error_fraction": fraction(sample.status_code >= 400 for sample in samples),
        "latency_s": describe(latencies),
    }


def summarize_progress(samples: Sequence[ProgressSample]) -> dict[str, Any]:
    if not samples:
        return {"samples": 0}

    latest = samples[-1]
    rates = [
        sample.seconds_per_chunk
        for sample in samples
        if sample.seconds_per_chunk is not None
    ]
    slow_rates = [
        value
        for value in rates
        if value >= 1.0
    ]

    return {
        "samples": len(samples),
        "latest_current": latest.current,
        "latest_total": latest.total,
        "latest_fraction": latest.current / latest.total if latest.total else None,
        "latest_elapsed_s": latest.elapsed_s,
        "latest_eta_s": latest.eta_s,
        "latest_seconds_per_chunk": latest.seconds_per_chunk,
        "seconds_per_chunk_all_samples": describe(rates),
        "seconds_per_chunk_slow_samples": describe(slow_rates),
        "inferred_generated_chunks_per_hour": (
            3600.0 / median(slow_rates) if slow_rates else None
        ),
    }


def summarize_ref_store(
    ref_store: Path | None,
    *,
    ref_version: str,
    recent_chunks: int,
) -> dict[str, Any]:
    if ref_store is None or not ref_store.exists():
        return {"available": False}

    chunks_dir = ref_store / "in_progress" / ref_version / "chunks"
    if not chunks_dir.exists():
        complete_dir = ref_store / "versions" / ref_version / "chunks"
        chunks_dir = complete_dir if complete_dir.exists() else chunks_dir

    files = sorted(chunks_dir.glob("chunk_*.json"))
    if not files:
        return {
            "available": True,
            "chunks_dir": str(chunks_dir),
            "chunks": 0,
        }

    chunk_numbers = [chunk_number(path) for path in files]
    mtimes = [path.stat().st_mtime for path in files]
    intervals = [
        right - left
        for left, right in zip(mtimes, mtimes[1:])
        if right >= left
    ]
    recent_intervals = intervals[-recent_chunks:] if recent_chunks > 0 else []
    missing = missing_ranges(chunk_numbers)

    reward_stats = summarize_chunk_rewards(files)
    work_stats = summarize_chunk_work(files)
    recent_chunks_per_hour = (
        3600.0 / median(recent_intervals) if recent_intervals else None
    )

    return {
        "available": True,
        "chunks_dir": str(chunks_dir),
        "chunks": len(files),
        "first_chunk": min(chunk_numbers),
        "last_chunk": max(chunk_numbers),
        "missing_ranges": missing,
        "first_mtime": datetime.fromtimestamp(min(mtimes)).isoformat(sep=" "),
        "last_mtime": datetime.fromtimestamp(max(mtimes)).isoformat(sep=" "),
        "save_interval_s": describe(intervals),
        "recent_save_interval_s": describe(recent_intervals),
        "recent_chunks_per_hour": recent_chunks_per_hour,
        "recent_prompts_per_hour": multiply_or_none(
            recent_chunks_per_hour,
            work_stats.get("prompts_per_chunk_mean"),
        ),
        "recent_ref_completions_per_hour": multiply_or_none(
            recent_chunks_per_hour,
            work_stats.get("ref_completions_per_chunk_mean"),
        ),
        "recent_judge_calls_per_hour_estimate": multiply_or_none(
            multiply_or_none(
                recent_chunks_per_hour,
                work_stats.get("ref_completions_per_chunk_mean"),
            ),
            work_stats.get("aspect_count_estimate"),
        ),
        "work": work_stats,
        "rewards": reward_stats,
    }


def summarize_chunk_rewards(files: Sequence[Path]) -> dict[str, Any]:
    zero_chunks: list[int] = []
    nonzero_chunks: list[int] = []
    means: list[float] = []
    mins: list[float] = []
    maxes: list[float] = []
    parse_errors: list[str] = []

    for path in progress(files, desc="Scanning chunk rewards"):
        number = chunk_number(path)
        try:
            payload = json.loads(path.read_text())
            rewards = [
                float(value)
                for row in payload.get("rows", [])
                for value in row.get("ref_rewards", [])
            ]
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")
            continue

        if not rewards:
            continue

        chunk_min = min(rewards)
        chunk_max = max(rewards)
        chunk_mean = statistics.fmean(rewards)
        mins.append(chunk_min)
        maxes.append(chunk_max)
        means.append(chunk_mean)

        if chunk_min == 0.0 and chunk_max == 0.0:
            zero_chunks.append(number)
        else:
            nonzero_chunks.append(number)

    return {
        "chunks_with_rewards": len(zero_chunks) + len(nonzero_chunks),
        "all_zero_chunks": ranges(zero_chunks),
        "nonzero_chunks": ranges(nonzero_chunks),
        "chunk_mean_reward": describe(means),
        "chunk_min_reward": describe(mins),
        "chunk_max_reward": describe(maxes),
        "parse_errors": parse_errors[:20],
        "parse_error_count": len(parse_errors),
    }


def summarize_chunk_work(files: Sequence[Path]) -> dict[str, Any]:
    prompts_per_chunk: list[int] = []
    completions_per_chunk: list[int] = []
    ref_rewards_per_prompt: list[int] = []
    aspect_counts: list[int] = []
    parse_errors: list[str] = []

    for path in progress(files, desc="Scanning chunk work"):
        try:
            payload = json.loads(path.read_text())
            rows = payload.get("rows", [])
            prompts_per_chunk.append(len(rows))

            chunk_completions = 0
            for row in rows:
                rewards = row.get("ref_rewards", [])
                extras = row.get("reward_extra_info", [])
                chunk_completions += len(rewards)
                ref_rewards_per_prompt.append(len(rewards))
                if extras:
                    aspect_counts.append(count_reward_aspects(extras[0]))
            completions_per_chunk.append(chunk_completions)
        except Exception as exc:
            parse_errors.append(f"{path.name}: {exc}")

    return {
        "prompts_per_chunk": describe(prompts_per_chunk),
        "ref_completions_per_chunk": describe(completions_per_chunk),
        "ref_rewards_per_prompt": describe(ref_rewards_per_prompt),
        "prompts_per_chunk_mean": (
            statistics.fmean(prompts_per_chunk) if prompts_per_chunk else None
        ),
        "ref_completions_per_chunk_mean": (
            statistics.fmean(completions_per_chunk)
            if completions_per_chunk
            else None
        ),
        "aspect_count_estimate": (
            statistics.median(aspect_counts) if aspect_counts else None
        ),
        "parse_errors": parse_errors[:20],
        "parse_error_count": len(parse_errors),
    }


def count_reward_aspects(extra_info: Any) -> int:
    if not isinstance(extra_info, dict):
        return 1
    aspect_keys = [
        key
        for key in extra_info
        if isinstance(key, str)
        and key.startswith("reward/")
        and key.endswith("_score")
    ]
    return max(1, len(aspect_keys))


def iter_log_lines(
    path: Path,
    *,
    recent_lines: int | None = None,
    progress_desc: str | None = None,
) -> Iterable[str]:
    if recent_lines is not None:
        if recent_lines <= 0:
            return []
        lines = tail_lines(path, recent_lines)
        return progress(lines, desc=progress_desc or f"Parsing {path.name}", unit="line")
    return read_lines(path, progress_desc=progress_desc)


def read_lines(path: Path, *, progress_desc: str | None = None) -> Iterable[str]:
    byte_progress = None
    if tqdm is not None and progress_desc is not None:
        byte_progress = tqdm(
            total=path.stat().st_size,
            desc=progress_desc,
            unit="B",
            unit_scale=True,
            leave=False,
        )
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if byte_progress is not None:
                byte_progress.update(len(line.encode(errors="replace")))
            yield line
    if byte_progress is not None:
        byte_progress.close()


def tail_lines(path: Path, n_lines: int) -> list[str]:
    # Keep this simple and robust for shared filesystem logs. The router log can
    # be large, but reading bytes from the end avoids parsing old trainer runs.
    block_size = 1024 * 1024
    data = b""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        while position > 0 and data.count(b"\n") <= n_lines:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size) + data
    return data.decode(errors="replace").splitlines()[-n_lines:]


def filter_vllm_samples(
    samples: Sequence[VLLMSample],
    *,
    since: datetime | None,
    until: datetime | None,
) -> list[VLLMSample]:
    return [
        sample
        for sample in samples
        if timestamp_in_window(sample.timestamp, since=since, until=until)
    ]


def filter_router_samples(
    samples: Sequence[RouterSample],
    *,
    since: datetime | None,
    until: datetime | None,
) -> list[RouterSample]:
    return [
        sample
        for sample in samples
        if timestamp_in_window(sample.timestamp, since=since, until=until)
    ]


def timestamp_in_window(
    timestamp: datetime | None,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    if timestamp is None:
        return False
    if since is not None and timestamp < since:
        return False
    if until is not None and timestamp > until:
        return False
    return True


def parse_datetime_arg(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse datetime {value!r}. Use 'YYYY-MM-DD HH:MM:SS'."
        ) from exc


def progress(items: Sequence[Any], *, desc: str, unit: str = "file") -> Iterable[Any]:
    if tqdm is None or not items:
        return items
    return tqdm(items, desc=desc, unit=unit, leave=False)


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def parse_month_day_timestamp(month: int, day: int, time_value: str) -> datetime | None:
    # vLLM logs omit the year. Use the current year; these summaries are meant for
    # run-local ordering and durations, not cross-year archival timestamps.
    year = datetime.now().year
    try:
        return datetime.strptime(
            f"{year}-{month:02d}-{day:02d} {time_value}",
            "%Y-%m-%d %H:%M:%S",
        )
    except ValueError:
        return None


def parse_duration(value: str) -> float | None:
    value = value.strip()
    if value in {"?", ""}:
        return None

    parts = value.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None

    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return None


def describe(values: Sequence[float | int]) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values if math.isfinite(float(value))]
    if not cleaned:
        return {"count": 0}
    return {
        "count": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "min": min(cleaned),
        "p50": percentile(cleaned, 50),
        "p90": percentile(cleaned, 90),
        "p95": percentile(cleaned, 95),
        "p99": percentile(cleaned, 99),
        "max": max(cleaned),
    }


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values.")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile_value / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def fraction(values: Iterable[bool]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(1 for value in items if value) / len(items)


def multiply_or_none(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) * float(right)


def min_dt(values: Iterable[datetime | None]) -> datetime | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def max_dt(values: Iterable[datetime | None]) -> datetime | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def format_dt(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value is not None else None


def chunk_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def missing_ranges(values: Sequence[int]) -> list[str]:
    if not values:
        return []
    present = set(values)
    missing = [
        value
        for value in range(min(values), max(values) + 1)
        if value not in present
    ]
    return ranges(missing)


def ranges(values: Sequence[int]) -> list[str]:
    if not values:
        return []

    sorted_values = sorted(values)
    result: list[str] = []
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        result.append(format_range(start, prev))
        start = prev = value
    result.append(format_range(start, prev))
    return result


def format_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def print_text_report(report: dict[str, Any]) -> None:
    print("======================")
    print("QRPO Efficiency Report")
    print("======================")
    print()

    progress = report["trainer_progress"]
    print("Trainer Ref-Reward Progress")
    print("---------------------------")
    if progress.get("samples", 0) == 0:
        print("No tqdm progress samples found.")
    else:
        print(
            f"latest: {progress['latest_current']}/{progress['latest_total']} "
            f"({format_pct(progress['latest_fraction'])})"
        )
        print(
            "latest seconds/chunk: "
            f"{format_float(progress.get('latest_seconds_per_chunk'))}"
        )
        print(
            "generated chunks/hour, inferred from slow samples: "
            f"{format_float(progress.get('inferred_generated_chunks_per_hour'))}"
        )
    print()

    print_vllm_section("Trainer vLLM / Rollout Logs", report["trainer_vllm"])
    print_vllm_section("Judge vLLM Logs", report["judge_vllm"])

    router = report["judge_router"]
    print("Judge Router Requests")
    print("---------------------")
    if router.get("samples", 0) == 0:
        print("No router completion samples found.")
    else:
        print(f"samples: {router['samples']}")
        print(f"time: {router['time_start']} -> {router['time_end']}")
        print(f"requests/s: {format_float(router.get('requests_per_s'))}")
        print(f"status counts: {router['status_counts']}")
        print(f"error fraction: {format_pct(router.get('error_fraction'))}")
        print(f"latency_s: {format_description(router['latency_s'])}")
    print()

    store = report["ref_store"]
    print("Ref Reward Store")
    print("----------------")
    if not store.get("available"):
        print("No ref store data available.")
    elif store.get("chunks", 0) == 0:
        print(f"No chunks found in {store.get('chunks_dir')}.")
    else:
        print(
            f"chunks: {store['chunks']} "
            f"({store['first_chunk']}..{store['last_chunk']})"
        )
        print(f"missing ranges: {store['missing_ranges']}")
        print(f"last save: {store['last_mtime']}")
        print(
            "recent chunks/hour from mtimes: "
            f"{format_float(store.get('recent_chunks_per_hour'))}"
        )
        print(
            "recent prompts/hour from mtimes: "
            f"{format_float(store.get('recent_prompts_per_hour'))}"
        )
        print(
            "recent ref completions/hour from mtimes: "
            f"{format_float(store.get('recent_ref_completions_per_hour'))}"
        )
        print(
            "recent judge calls/hour estimate: "
            f"{format_float(store.get('recent_judge_calls_per_hour_estimate'))}"
        )
        print(
            "recent save interval_s: "
            f"{format_description(store['recent_save_interval_s'])}"
        )
        work = store["work"]
        print(
            "work per chunk: "
            f"prompts={format_float(work.get('prompts_per_chunk_mean'))}, "
            f"ref_completions={format_float(work.get('ref_completions_per_chunk_mean'))}, "
            f"aspects={format_float(work.get('aspect_count_estimate'))}"
        )
        rewards = store["rewards"]
        print(f"all-zero chunks: {rewards['all_zero_chunks']}")
        print(f"nonzero chunks: {rewards['nonzero_chunks'][:5]}")
        print(
            "chunk mean reward: "
            f"{format_description(rewards['chunk_mean_reward'])}"
        )


def print_vllm_section(title: str, data: dict[str, Any]) -> None:
    print(title)
    print("-" * len(title))
    if data.get("samples", 0) == 0:
        print("No vLLM metric samples found.")
        print()
        return

    aggregate = data["aggregate"]
    print(f"samples: {data['samples']}")
    print(f"time: {aggregate['time_start']} -> {aggregate['time_end']}")
    print(
        "saturation fraction waiting>0: "
        f"{format_pct(aggregate.get('saturation_fraction_waiting_gt_0'))}"
    )
    print(
        "heavy queue fraction waiting>=100: "
        f"{format_pct(aggregate.get('heavy_queue_fraction_waiting_ge_100'))}"
    )
    print(
        "idle fraction: "
        f"{format_pct(aggregate.get('idle_fraction_running_eq_0_waiting_eq_0'))}"
    )
    print(f"prompt_tps: {format_description(aggregate['prompt_tps'])}")
    print(f"generation_tps: {format_description(aggregate['generation_tps'])}")
    print(f"running reqs: {format_description(aggregate['running_reqs'])}")
    print(f"waiting reqs: {format_description(aggregate['waiting_reqs'])}")
    print(f"kv cache usage: {format_description(aggregate['kv_cache_usage_pct'])}")
    print()


def format_description(description: dict[str, Any]) -> str:
    if description.get("count", 0) == 0:
        return "n/a"
    return (
        f"mean={format_float(description['mean'])}, "
        f"p50={format_float(description['p50'])}, "
        f"p95={format_float(description['p95'])}, "
        f"max={format_float(description['max'])}"
    )


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


if __name__ == "__main__":
    main()
