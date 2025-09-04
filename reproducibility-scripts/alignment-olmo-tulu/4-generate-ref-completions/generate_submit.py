from datetime import datetime
from pathlib import Path

import yaml

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

stdout_prefix = "run"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "datasets-with-ref-completions"

datasets = ["olmo2-32b-preference"]
splits = ["train_split", "eval_split"]
models = ["olmo2-32b-sft"]
reward_models = ["skywork-llama3-8b", "skywork-qwen3-8b", "armorm-llama3-8b"]
sftids = ["default"]

model_sftid_path_prefix = "\${artifacts_dir}/shared/"
model_sftid_paths = {
    "olmo2-32b-sft-default": "models/olmo2-32b-sft",
}

dataset_num_ref_reward = 10

# Reference numbers for 8B
## ~6000 tokens per second
# 128 prompts with 100 completions each take 15 minutes on 1 GPU
# 1024 prompts with 50 completions each take 1h on 1 GPU

# 32B is 3x slower.
# And we have 2x longer context length.

# We want
# 400k prompts
# 1024 prompts with 10 completions each take 2h on 1 GPU


# completions
# subpartition size = 1024
# partition size = 1024 * 4 = 4096
# nb partitions = 400000 / 4096 <= 100

partition_size = 4096  # 4 GPUs per node, 2048 prompts per GPU
save_interval = 1024  # 2h per GPU
num_nodes_per_job = 1  # 1 node per job

commands = []
total_nodes_needed = 0
for dataset in datasets:
    for reward_model in reward_models:
        for model in models:
            for sftid in sftids:
                for split in splits:
                    dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                    dataset_with_chosen_rewards_for_model = (
                        f"{dataset_with_chosen_rewards}-{model}"
                    )
                    model_sftid = f"{model}-{sftid}"
                    dataset_with_ref_completions = f"{dataset_with_chosen_rewards}-{model_sftid}-Nref{dataset_num_ref_reward}"

                    with open(
                        f"src/post_training/configs/dataset/{dataset_with_chosen_rewards_for_model}.yaml",
                        "r",
                    ) as file:
                        dataset_config = yaml.safe_load(file)
                    split_config = dataset_config["dataset_args"][split]
                    split_name = split_config["name"]
                    if split_name is None:
                        continue
                    split_size = split_config["end"] - split_config["start"]

                    for partition_start_idx in range(0, split_size, partition_size):
                        partition_end_idx = min(
                            partition_start_idx + partition_size, split_size
                        )
                        jobid = f"{dataset_with_ref_completions}/{split}/{partition_start_idx}-{partition_end_idx}"
                        model_path = f"{model_sftid_path_prefix}/{model_sftid_paths[model_sftid]}"

                        commands.append(
                            (
                                "sbatch "
                                f"-N {num_nodes_per_job} "
                                f"-o {stdout_root}/out/{jobid}.out "
                                f"-e {stdout_root}/out/{jobid}.err "
                                "./cscs-shared-submit-scripts/unattended-generate-ref-completions-with-vllm.sh "
                                f"model={model} "
                                f"model_args.model_name_or_path='{model_path}' "
                                f"dataset={dataset_with_chosen_rewards_for_model} "
                                f"split={split_name} "
                                f"n_completions={dataset_num_ref_reward} "
                                f"partition_start_idx={partition_start_idx} "
                                f"partition_end_idx={partition_end_idx} "
                                f"save_interval={save_interval} "
                                "artifacts_subdir=shared "
                                f"job_subdir_prefix={job_name}/{jobid} "
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
