from datetime import datetime
from pathlib import Path

"""Nomenclature:

dataset = f"{dataset}"
model = f"{sft_model}"
reward_model = f"{reward_model}"

dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
dataset_with_chosen_rewards_for_model = f"{dataset}-{reward_model}-{model}"

model_sftid = f"{model}-(sftid)"
               = f"{model}-(sftid)"

dataset_with_ref_completions = f"{dataset_with_chosen_rewards_for_model}-(sftid)-Nref{NRefDataset}"
                             = f"{dataset}-{reward_model}-{model}-(sftid)-Nref{NRefDataset}"

dataset_with_ref_rewards = f"{dataset_with_ref_completions}-(offline|offpolicy|mix)"
                         = f"{dataset}-{reward_model}-{model}-(sftid)-Nref{NRefDataset}-(offline|offpolicy|mix)"
"""

stdout_prefix = "offlinepatch"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

datasets = ["olmo2-32b-preference"]

models = ["olmo2-32b-sft"]
sftids = ["default"]

reward_models = ["skywork-llama3-8b", "skywork-qwen3-8b", "armorm-llama3-8b"]

dataset_num_ref_reward = 10
modes = ["offlinepatch"]

num_nodes_per_job = 1
commands = []
total_nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            for sftid in sftids:
                for mode in modes:
                    dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                    dataset_with_chosen_rewards_for_model = (
                        f"{dataset_with_chosen_rewards}-{model}"
                    )
                    model_sftid = f"{model}-{sftid}"
                    dataset_with_ref_completions = f"{dataset_with_chosen_rewards}-{model_sftid}-Nref{dataset_num_ref_reward}"
                    dataset_with_ref_rewards = f"{dataset_with_ref_completions}-offline"
                    new_dataset_with_ref_rewards = (
                        f"{dataset_with_ref_completions}-{mode}"
                    )
                    jobid = new_dataset_with_ref_rewards
                    commands.append(
                        (
                            "sbatch "
                            f"-t 30:00 "
                            f"-N {num_nodes_per_job} "
                            f"-o {stdout_root}/out/{jobid}.out "
                            f"-e {stdout_root}/out/{jobid}.err "
                            "./cscs-shared-submit-scripts/unattended.sh "
                            f"python -m swiss_alignment.data_alignment.prepare_offpolicy_dataset "
                            f"dataset={dataset_with_chosen_rewards_for_model} "
                            f"dataset_id={dataset_with_ref_rewards} "
                            f"new_dataset_id={new_dataset_with_ref_rewards} "
                            f"mode={mode}"
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
