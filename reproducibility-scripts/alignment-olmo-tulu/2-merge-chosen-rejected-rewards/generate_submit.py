from datetime import datetime
from pathlib import Path

"""Nomenclature:

dataset = f"{dataset}"
reward_model = f"{reward_model}"

dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
"""

stdout_prefix = "run"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)


datasets = ["olmo2-32b-preference"]
reward_models = ["skywork-llama3-8b", "skywork-qwen3-8b", "armorm-llama3-8b"]
is_partitioned = False
dataset_type = "datasets-with-chosen-rewards"
num_nodes_per_job = 1

commands = []
total_nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
        jobid = dataset_with_chosen_rewards
        commands.append(
            (
                "sbatch "
                f"-N {num_nodes_per_job} "
                f"-o {stdout_root}/out/{jobid}.out "
                f"-e {stdout_root}/out/{jobid}.err "
                "./cscs-shared-submit-scripts/unattended.sh "
                f"python -m swiss_alignment.data_alignment.merge_partitions "
                f"dataset={dataset_with_chosen_rewards} "
                f"dataset_id={dataset_with_chosen_rewards} "
                f"dataset_type={dataset_type} "
                f"is_partitioned={is_partitioned} "
                f"job_subdir={dataset_type}/{jobid} "
                "artifacts_subdir=shared "
                "resuming.resume=True "
            )
        )
        total_nodes_needed += num_nodes_per_job

# Write th submit commands to a new directory where this batch of experiments will be managed)
# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Total nodes needed:", total_nodes_needed)
