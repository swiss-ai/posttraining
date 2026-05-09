from __future__ import annotations

import asyncio
import importlib.util
import math
import os
from functools import lru_cache
from typing import Any, Mapping

import httpx
from openai import AsyncOpenAI


TARGET_TOKENS = ("1", "2", "3", "4", "5")

_CLIENTS: dict[tuple[int, str, str], tuple[AsyncOpenAI, httpx.AsyncClient]] = {}
_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _env(name: str, *, required: bool = True, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"Environment variable {name} must be set.")
    return str(value)


@lru_cache(maxsize=1)
def _load_prompts():
    path = _env("ACTIVE_UF_PROMPTS_PATH")
    spec = importlib.util.spec_from_file_location("active_ultrafeedback_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prompts module from {path!r}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aspect_prompts() -> dict[str, str]:
    prompts = _load_prompts()

    if hasattr(prompts, "ASPECT2ANNOTATION_PROMPT"):
        aspect2prompt = dict(prompts.ASPECT2ANNOTATION_PROMPT)
    else:
        aspect2prompt = {
            "instruction_following": getattr(
                prompts,
                "INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT",
                None,
            ),
            "honesty": getattr(prompts, "HONESTY_ANNOTATION_PROMPT", None),
            "truthfulness": getattr(prompts, "TRUTHFULNESS_ANNOTATION_PROMPT", None),
            "helpfulness": getattr(prompts, "HELPFULNESS_ANNOTATION_PROMPT", None),
        }
        aspect2prompt = {k: v for k, v in aspect2prompt.items() if v is not None}

    selected = os.environ.get("ACTIVE_UF_ASPECTS")
    if selected:
        keep = {x.strip() for x in selected.split(",") if x.strip()}
        aspect2prompt = {k: v for k, v in aspect2prompt.items() if k in keep}

    if not aspect2prompt:
        raise RuntimeError("No active UltraFeedback aspect prompts configured.")

    return {str(k): str(v) for k, v in aspect2prompt.items()}


def _system_prompt() -> str:
    return str(_load_prompts().PREFERENCE_ANNOTATION_SYSTEM_PROMPT)


def _format_prompt_input(prompt_data: Any) -> str:
    if isinstance(prompt_data, str):
        return prompt_data.strip()

    if isinstance(prompt_data, tuple):
        prompt_data = list(prompt_data)

    if isinstance(prompt_data, list):
        messages = [_message_to_dict(message) for message in prompt_data]

        if len(messages) == 1:
            return str(messages[0].get("content", "")).strip()

        formatted = "### CONVERSATION HISTORY ###\n"
        for turn in messages[:-1]:
            role = str(turn.get("role", "")).upper()
            content = str(turn.get("content", "")).strip()
            formatted += f"[{role}]: {content}\n\n"

        formatted += "### FINAL INSTRUCTION ###\n"
        formatted += str(messages[-1].get("content", "")).strip()
        return formatted

    return str(prompt_data)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)

    if hasattr(message, "role") and hasattr(message, "content"):
        return {
            "role": getattr(message, "role"),
            "content": getattr(message, "content"),
        }

    raise TypeError(f"Unsupported message type: {type(message).__name__}.")


def _expected_score(probs: Mapping[str, float]) -> float:
    return float(
        sum(int(token) * float(probs.get(token, 0.0)) for token in TARGET_TOKENS)
    )


def _extract_probabilities(res: Any) -> dict[str, float]:
    try:
        first_token_logprobs = res.choices[0].logprobs.content[0].top_logprobs
    except Exception as exc:
        raise ValueError("Judge response does not contain first-token top_logprobs.") from exc

    best_logprob = {token: -float("inf") for token in TARGET_TOKENS}

    for lp in first_token_logprobs:
        token = str(lp.token).strip()
        if token in best_logprob:
            best_logprob[token] = max(best_logprob[token], float(lp.logprob))

    exp_values = {
        token: math.exp(logprob) if math.isfinite(logprob) else 0.0
        for token, logprob in best_logprob.items()
    }

    total = sum(exp_values.values())
    if total <= 0:
        raise ValueError(
            "Judge response top_logprobs contain no probability mass for target "
            f"tokens {TARGET_TOKENS}."
        )

    return {
        token: float(value / total)
        for token, value in exp_values.items()
    }


def _get_client() -> AsyncOpenAI:
    loop = asyncio.get_running_loop()
    base_url = _env("JUDGE_BASE_URL")
    api_key = _env("JUDGE_API_KEY")

    key = (id(loop), base_url, api_key)
    if key in _CLIENTS:
        return _CLIENTS[key][0]

    max_connections = int(os.environ.get("JUDGE_MAX_CONNECTIONS", "128"))
    timeout_s = float(os.environ.get("JUDGE_TIMEOUT_S", "60"))

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=min(max_connections, 100),
        ),
    )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )
    _CLIENTS[key] = (client, http_client)
    return client


def _get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)

    if loop_id not in _SEMAPHORES:
        _SEMAPHORES[loop_id] = asyncio.Semaphore(
            int(os.environ.get("JUDGE_MAX_CONCURRENCY_PER_WORKER", "64"))
        )

    return _SEMAPHORES[loop_id]


async def _judge_aspect(
    *,
    aspect: str,
    aspect_prompt: str,
    formatted_prompt: str,
    completion: str,
) -> float:
    client = _get_client()
    semaphore = _get_semaphore()
    model = _env("JUDGE_MODEL")

    user_prompt = aspect_prompt.format(
        prompt=formatted_prompt,
        completion=completion,
    )

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

    max_retries = int(os.environ.get("JUDGE_MAX_RETRIES", "2"))
    retry_base_sleep_s = float(os.environ.get("JUDGE_RETRY_BASE_SLEEP_S", "0.5"))

    for attempt in range(max_retries + 1):
        try:
            async with semaphore:
                res = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=20,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

            return _expected_score(_extract_probabilities(res))

        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    "Active UltraFeedback judge request failed after "
                    f"{max_retries + 1} attempts for aspect {aspect!r}."
                ) from exc
            await asyncio.sleep(retry_base_sleep_s * (2**attempt))

    raise RuntimeError(
        "Active UltraFeedback judge request failed unexpectedly for "
        f"aspect {aspect!r}."
    )


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
):
    if data_source != "activeultrafeedback":
        raise NotImplementedError(
            "active_ultrafeedback_reward only supports "
            f"data_source='activeultrafeedback', got {data_source!r}."
        )

    extra_info = extra_info or {}

    prompt = extra_info.get("prompt")
    if prompt is None and isinstance(ground_truth, Mapping):
        prompt = ground_truth.get("prompt")

    if prompt is None:
        raise ValueError(
            "active_ultrafeedback_reward requires extra_info['prompt'] "
            "or ground_truth['prompt']."
        )

    formatted_prompt = _format_prompt_input(prompt)
    completion = str(solution_str).strip()

    aspect2prompt = _aspect_prompts()

    tasks = [
        _judge_aspect(
            aspect=aspect,
            aspect_prompt=aspect_prompt,
            formatted_prompt=formatted_prompt,
            completion=completion,
        )
        for aspect, aspect_prompt in aspect2prompt.items()
    ]

    scores = await asyncio.gather(*tasks)

    aspect_scores = {
        aspect: float(score)
        for aspect, score in zip(aspect2prompt.keys(), scores)
    }

    reward = (
        sum(aspect_scores.values()) / len(aspect_scores)
        if aspect_scores
        else 0.0
    )

    return {
        "score": float(reward),
        **{
            f"reward/{aspect}_score": float(score)
            for aspect, score in aspect_scores.items()
        },
    }
