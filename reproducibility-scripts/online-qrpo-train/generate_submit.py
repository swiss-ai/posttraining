from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
import itertools
import shlex


"""Generate Slurm submit commands for online QRPO sweeps.

Run this file directly, edit the variables below to change the sweep.

Batch-size math:

total_gpus = num_nodes_per_job * num_devices_per_node
trajectories_per_prompt = n_online + n_offline
prompt_batch_size = global_train_batch_size / trajectories_per_prompt
train_batch_size_per_gpu = global_train_batch_size / total_gpus

actor_gradient_accumulation_steps =
    train_batch_size_per_gpu / train_micro_batch_size_per_gpu

ref_logprob_accumulation_steps =
    train_batch_size_per_gpu / log_prob_micro_batch_size_per_gpu

Therefore:
  - global_train_batch_size controls the final trajectory batch per optimizer update.
  - data.train_batch_size controls prompt batch size and is derived per grid job.
  - train_micro_batch_size_per_gpu controls actor memory.
  - log_prob_micro_batch_size_per_gpu controls ref-logprob memory.
  - global_train_batch_size must be divisible by:
      total_gpus,
      n_online + n_offline,
      total_gpus * train_micro_batch_size_per_gpu,
      total_gpus * log_prob_micro_batch_size_per_gpu.
"""


stdout_prefix = "init"
script_dir = Path(__file__).parent.resolve()
stdout_root = script_dir / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

job_name = "ap1p5-8b-64k-lc-stable-lr-ablate-mixed-adam-lr8e-5-linear-64n_mixed-sweep-true-off-rewards-long"

project_root = "/users/smatreno/projects/posttraining/dev"
submit_script = (
    f"{project_root}/src/post_training/qrpo-verl/scripts/submit_online_qrpo_ray.sh"
)

judge_base_url = "http://172.28.16.188:30000/v1"
judge_model = "Qwen/Qwen3.6-27B-smatrenok"
judge_max_concurrency_per_worker = 64
judge_max_connections = 2048
judge_timeout_s = 60

num_nodes_per_job = 4
num_devices_per_node = 4
global_train_batch_size = 256
train_micro_batch_size_per_gpu = 8
log_prob_micro_batch_size_per_gpu = 8

time_limit = "12:00:00"
account = "infra01"
partition = None
# reservation = "SD-69241-apertus-1-5"
reservation = None

data_path = (
    f"{project_root}/artifacts/private/datasets/"
    "MaxMin-Filtered-OnlineQRPOformat-30-ref-rewards-qwen-36-27B-judge-decontaminated-offline-rewards-recomputed/"
    "train_split"
)
model_path = (
    f"{project_root}/artifacts/private/baseline-checkpoints/"
    "ap1p5-8b-64k-lc-stable-lr-ablate-mixed-adam-lr8e-5-linear-64n"
)
reward_function_path = (
    f"{project_root}/src/post_training/qrpo-verl/rewards/"
    "active_ultrafeedback_reward.py"
)
ref_reward_store_dir = (
    f"{project_root}/artifacts/private/outputs/ref_reward_stores/"
    "ap1p5-8b-64k-lc-stable-lr-ablate-mixed-adam-lr8e-5-linear-64n"
)
output_root = f"{project_root}/artifacts/private/outputs/online-qrpo"

project_name = "ap1p5-8b-64k-lc-stable-lr-ablate-mixed-adam-lr8e-5-linear-64n_mixed-sweep-true-off-rewards-long"
wandb_entity = "apertus" # is overridden in setup.sh

# learning_rates = [1.5e-5, 2.5e-5, 3.5e-5]
# length-norm:
learning_rates = [1e-5, 1.5e-5, 2.5e-5] # <-- mixed
# learning_rates = [1.5e-5, 2e-5, 2.5e-5] # <-- online
# learning_rates = [2.5e-5]


# length_normalizations = [False]
# betas = [0.005, 0.01, 0.025]

length_normalizations = [True]
length_normalized_effective_beta_max = 0.1
betas = [1.25, 2.0, 2.5] # <-- mixed
# betas = [1.25, 2.5, 5.0] # <-- online
# # betas = [1.25]

nums_online = [1]
nums_offline = [1]
offline_selectors = ["random"]

candidate_selection_enableds = [False]
candidates_per_train_sample_values = [16]
candidate_selection_probability_values = [0.5]

# Useful extra dimensions to add later:
grad_clips = [20.0]
# warmup_ratios = [0.03, 0.1]
# weight_decays = [0.0, 0.01]

common_hydra_overrides = {
    "data.path": data_path,
    "actor_rollout_ref.model.path": model_path,
    "tokenizer.use_eos_as_pad": False,
    "offline_tokenization.require_assistant_mask": False,
    "actor_rollout_ref.rollout.n": 1,
    "reward.num_workers": 32,
    "reward.custom_reward_function.path": reward_function_path,
    "reward.custom_reward_function.name": "compute_score",
    "online_rollout.data_source": "activeultrafeedback",
    "trainer.save_freq": 100,
    "trainer.test_freq": 0,
    "trainer.val_before_train": False,
    "trainer.logger": '["console","wandb"]',
    "trainer.project_name": project_name,
    "online_rollout.completion_logging.enabled": False,
    "online_rollout.completion_logging.outputs": '["wandb"]',
    "online_rollout.completion_logging.selection": "all",
    # Sweeps should reuse a completed store. Do not let all jobs generate it.
    "ref_rewards.initial_source": "store",
    "ref_rewards.initial_version": "ref_step_000000",
    "ref_rewards.refresh_interval_epochs": None,
    "data.ref_rewards_key": "ref_rewards",
    "ref_rewards.store_dir": ref_reward_store_dir,
    "ref_rewards.generation_num_chunks": None,
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 4096,
    "actor_rollout_ref.rollout.agent.num_workers": 32,
    "actor_rollout_ref.rollout.max_num_seqs": 4096,
    "actor_rollout_ref.rollout.max_num_batched_tokens": 131072,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
    "actor_rollout_ref.rollout.disable_log_stats": False,
}


def validate_batch_math(*, trajectories_per_prompt: int) -> dict[str, int]:
    if trajectories_per_prompt <= 0:
        raise ValueError(
            "Each grid job must have n_online + n_offline > 0, got "
            f"{trajectories_per_prompt}."
        )

    total_gpus = num_nodes_per_job * num_devices_per_node
    if global_train_batch_size % total_gpus != 0:
        raise ValueError(
            "global_train_batch_size must be divisible by total GPUs: "
            f"{global_train_batch_size} % {total_gpus} != 0."
        )
    if global_train_batch_size % trajectories_per_prompt != 0:
        raise ValueError(
            "global_train_batch_size must be divisible by "
            "n_online + n_offline: "
            f"{global_train_batch_size} % {trajectories_per_prompt} != 0."
        )

    train_batch_size_per_gpu = global_train_batch_size // total_gpus
    for name, micro_batch_size in (
        ("train_micro_batch_size_per_gpu", train_micro_batch_size_per_gpu),
        ("log_prob_micro_batch_size_per_gpu", log_prob_micro_batch_size_per_gpu),
    ):
        if train_batch_size_per_gpu % micro_batch_size != 0:
            raise ValueError(
                f"train_batch_size_per_gpu={train_batch_size_per_gpu} must be "
                f"divisible by {name}={micro_batch_size}."
            )

    return {
        "total_gpus": total_gpus,
        "trajectories_per_prompt": trajectories_per_prompt,
        "prompt_batch_size": global_train_batch_size // trajectories_per_prompt,
        "train_batch_size_per_gpu": train_batch_size_per_gpu,
        "actor_gradient_accumulation_steps": (
            train_batch_size_per_gpu // train_micro_batch_size_per_gpu
        ),
        "ref_logprob_accumulation_steps": (
            train_batch_size_per_gpu // log_prob_micro_batch_size_per_gpu
        ),
    }


def validate_online_rollout_count(
    *,
    online_trajectory_count: int,
    agent_loop_num_workers: int,
    candidate_selection_enabled: bool,
    candidates_per_train_sample: int,
    candidate_selection_probability: float,
) -> None:
    if online_trajectory_count == 0:
        return

    if candidates_per_train_sample <= 0:
        raise ValueError(
            "candidates_per_train_sample_values must be positive."
        )
    if not 0.0 <= candidate_selection_probability <= 1.0:
        raise ValueError(
            "candidate_selection_probability_values must be in [0, 1]."
        )

    needs_base_divisibility = (
        not candidate_selection_enabled
        or candidates_per_train_sample == 1
        or candidate_selection_probability == 0.0
    )
    if needs_base_divisibility:
        if online_trajectory_count % agent_loop_num_workers != 0:
            raise ValueError(
                "Online rollout request count must be divisible by "
                "actor_rollout_ref.rollout.agent.num_workers because VERL chunks "
                "agent-loop input equally. Got "
                f"online_trajectory_count={online_trajectory_count}, "
                f"agent_loop_num_workers={agent_loop_num_workers}."
            )
        return

    extra_candidates_per_selected = candidates_per_train_sample - 1
    valid_expanded_count_exists = any(
        (
            online_trajectory_count
            + selected_count * extra_candidates_per_selected
        )
        % agent_loop_num_workers
        == 0
        for selected_count in range(online_trajectory_count + 1)
    )
    if not valid_expanded_count_exists:
        raise ValueError(
            "Cannot make candidate-expanded online rollout request count "
            "divisible by actor_rollout_ref.rollout.agent.num_workers. Got "
            f"online_trajectory_count={online_trajectory_count}, "
            f"candidates_per_train_sample={candidates_per_train_sample}, "
            f"agent_loop_num_workers={agent_loop_num_workers}."
        )


def hydra_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def slug(value) -> str:
    return (
        hydra_value(value)
        .replace("/", "_")
        .replace(" ", "")
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
        .replace("=", "")
    )


def env_assignments(items: dict[str, object]) -> str:
    return " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in items.items()
    )


def shell_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def hydra_overrides(overrides: dict[str, object]) -> list[str]:
    return [f"{key}={hydra_value(value)}" for key, value in overrides.items()]


batch_math_by_trajectories_per_prompt: dict[int, dict[str, int]] = {}

commands = []
total_nodes_needed = 0
for (
    lr,
    beta,
    grad_clip,
    length_normalization,
    candidate_selection_enabled,
    candidates_per_train_sample,
    candidate_selection_probability,
    num_online,
    num_offline,
    offline_selector,
) in itertools.product(
    learning_rates,
    betas,
    grad_clips,
    length_normalizations,
    candidate_selection_enableds,
    candidates_per_train_sample_values,
    candidate_selection_probability_values,
    nums_online,
    nums_offline,
    offline_selectors,
):
    trajectories_per_prompt = num_online + num_offline
    if trajectories_per_prompt not in batch_math_by_trajectories_per_prompt:
        batch_math_by_trajectories_per_prompt[trajectories_per_prompt] = (
            validate_batch_math(trajectories_per_prompt=trajectories_per_prompt)
        )
    batch_math = batch_math_by_trajectories_per_prompt[trajectories_per_prompt]
    prompt_batch_size = batch_math["prompt_batch_size"]

    agent_loop_num_workers = int(
        common_hydra_overrides["actor_rollout_ref.rollout.agent.num_workers"]
    )
    online_trajectory_count = prompt_batch_size * num_online
    validate_online_rollout_count(
        online_trajectory_count=online_trajectory_count,
        agent_loop_num_workers=agent_loop_num_workers,
        candidate_selection_enabled=candidate_selection_enabled,
        candidates_per_train_sample=candidates_per_train_sample,
        candidate_selection_probability=candidate_selection_probability,
    )

    candidate_selection_slug = (
        "cand-off"
        if not candidate_selection_enabled
        else (
            f"cand{slug(candidates_per_train_sample)}-"
            f"p{slug(candidate_selection_probability)}"
        )
    )

    jobid = (
        f"lr{slug(lr)}-beta{slug(beta)}-"
        f"ln{slug(length_normalization)}-"
        f"{candidate_selection_slug}-"
        f"on{slug(num_online)}-off{slug(num_offline)}-"
        f"sel{slug(offline_selector)}"
    )
    run_name = f"{job_name}/{jobid}"

    overrides = {
        **common_hydra_overrides,
        "data.train_batch_size": prompt_batch_size,
        "ref_rewards.generation_prompt_batch_size": global_train_batch_size,
        "qrpo_runtime.train_mini_batch_size": global_train_batch_size,
        "qrpo_runtime.train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "qrpo_runtime.log_prob_micro_batch_size_per_gpu": (
            log_prob_micro_batch_size_per_gpu
        ),
        "qrpo_runtime.lr": lr,
        "qrpo.beta": beta,
        "qrpo_runtime.grad_clip": grad_clip,
        "qrpo.length_normalization": length_normalization,
        "online_rollout.candidate_selection.enabled": candidate_selection_enabled,
        "source_schedule.n_online": num_online,
        "source_schedule.n_offline": num_offline,
        "offline_selector": offline_selector,
        "trainer.nnodes": num_nodes_per_job,
        "trainer.n_gpus_per_node": num_devices_per_node,
        "trainer.experiment_name": run_name,
        "trainer.default_local_dir": f"{output_root}/{job_name}/{jobid}",
    }
    if length_normalization:
        overrides["qrpo.effective_beta_max"] = length_normalized_effective_beta_max
    if candidate_selection_enabled:
        overrides["online_rollout.candidate_selection.candidates_per_train_sample"] = (
            candidates_per_train_sample
        )
        overrides["online_rollout.candidate_selection.probability"] = (
            candidate_selection_probability
        )

    env = {
        "JUDGE_BASE_URL": judge_base_url,
        "JUDGE_MODEL": judge_model,
        "JUDGE_MAX_CONCURRENCY_PER_WORKER": judge_max_concurrency_per_worker,
        "JUDGE_MAX_CONNECTIONS": judge_max_connections,
        "JUDGE_TIMEOUT_S": judge_timeout_s,
        "WANDB_ENTITY": wandb_entity,
        "TRAINER_NNODES": num_nodes_per_job,
        "N_GPUS_PER_NODE": num_devices_per_node,
        "GLOBAL_TRAIN_BATCH_SIZE": global_train_batch_size,
        "TRAJECTORIES_PER_PROMPT": trajectories_per_prompt,
        "TRAIN_MICRO_BATCH_SIZE_PER_GPU": train_micro_batch_size_per_gpu,
        "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": log_prob_micro_batch_size_per_gpu,
    }

    sbatch = [
        "sbatch",
        "-J",
        f"{job_name}-{jobid}",
        "-N",
        num_nodes_per_job,
        "-t",
        time_limit,
        "-o",
        stdout_root / "out" / f"{jobid}.out",
        "-e",
        stdout_root / "out" / f"{jobid}.err",
    ]
    if account:
        sbatch.extend(["-A", account])
    if partition:
        sbatch.extend(["-p", partition])
    if reservation:
        sbatch.extend(["--reservation", reservation])
    sbatch.append(submit_script)
    sbatch.extend(hydra_overrides(overrides))

    commands.append(f"{env_assignments(env)} {shell_join(sbatch)}")
    total_nodes_needed += num_nodes_per_job


submit_dir = stdout_root
(submit_dir / "out").mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
    for command in commands:
        f.write(command + "\n")
submit_file.chmod(0o755)

readme_file = submit_dir / "README.md"
batch_math_lines = [
    f"total_gpus = {num_nodes_per_job} * {num_devices_per_node} = "
    f"{next(iter(batch_math_by_trajectories_per_prompt.values()))['total_gpus']}",
    f"global_train_batch_size = {global_train_batch_size}",
    "train_batch_size_per_gpu = global_train_batch_size / total_gpus = "
    f"{next(iter(batch_math_by_trajectories_per_prompt.values()))['train_batch_size_per_gpu']}",
    "actor_gradient_accumulation_steps = "
    "train_batch_size_per_gpu / train_micro_batch_size_per_gpu = "
    f"{next(iter(batch_math_by_trajectories_per_prompt.values()))['actor_gradient_accumulation_steps']}",
    "ref_logprob_accumulation_steps = "
    "train_batch_size_per_gpu / log_prob_micro_batch_size_per_gpu = "
    f"{next(iter(batch_math_by_trajectories_per_prompt.values()))['ref_logprob_accumulation_steps']}",
    "",
    "Per source schedule:",
]
for trajectories_per_prompt, math in sorted(
    batch_math_by_trajectories_per_prompt.items()
):
    batch_math_lines.append(
        "  trajectories_per_prompt = n_online + n_offline = "
        f"{trajectories_per_prompt}; "
        "prompt_batch_size = global_train_batch_size / trajectories_per_prompt = "
        f"{math['prompt_batch_size']}"
    )

with open(readme_file, "w") as f:
    f.write(
        "# Online QRPO Sweep\n\n"
        f"Jobs: {len(commands)}\n\n"
        "Batch-size math:\n\n"
        "```text\n"
        + "\n".join(batch_math_lines)
        + "\n"
        "```\n\n"
        f"Judge URL: `{judge_base_url}`\n\n"
        f"Run with: `bash {submit_file}`\n"
    )

print("Batch math:", batch_math_by_trajectories_per_prompt)
print("Total nodes needed:", total_nodes_needed)
print(f"README: {readme_file}")
