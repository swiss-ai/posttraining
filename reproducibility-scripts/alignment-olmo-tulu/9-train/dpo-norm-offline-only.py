from collections import defaultdict
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

stdout_prefix = "dpo-offline-only-norm-beta"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "olmo2-qrpo"

dataset_with_ref_rewards_path_prefix = "\${artifacts_dir}/shared/datasets/alignment-pipeline/datasets-with-ref-rewards/merged"
model_sftid_path_prefix = "\${artifacts_dir}/shared/"
model_sftid_paths = {
    "olmo2-32b-sft-default": "models/olmo2-32b-sft",
}

datasets = ["olmo2-32b-preference"]
distributions = [
    "offlinepatch",
]


models = ["olmo2-32b-sft"]
sftids = ["default"]

# reward_models = ["skywork-llama3-8b", "skywork-qwen3-8b", "armorm-llama3-8b"]
reward_models = ["skywork-llama3-8b"]

dataset_num_ref_reward = 10
train_num_ref_rewards_list = [10]

losses = ["dpo"]
betas = {
    "dpo": [5.0, 10.0],
}
learning_rates = [1e-7, 1e-6]
max_grad_norm = 1e8  # Disable but still log.

batch_size = 128
num_nodes_per_job = 16
num_devices_per_node = 4
per_device_train_batch_size = 2
accumulation_steps = batch_size // (
    num_nodes_per_job * num_devices_per_node * per_device_train_batch_size
)


commands = []
total_nodes_needed = 0
for dataset in datasets:
    for distribution in distributions:
        for reward_model in reward_models:
            for model in models:
                for sftid in sftids:
                    dataset_with_chosen_rewards = f"{dataset}-{reward_model}"
                    dataset_with_chosen_rewards_for_model = (
                        f"{dataset_with_chosen_rewards}-{model}"
                    )
                    model_sftid = f"{model}-{sftid}"
                    dataset_with_ref_completions = f"{dataset_with_chosen_rewards}-{model_sftid}-Nref{dataset_num_ref_reward}"
                    dataset_with_ref_rewards = (
                        f"{dataset_with_ref_completions}-{distribution}"
                    )

                    sft_model_path = (
                        f"{model_sftid_path_prefix}/{model_sftid_paths[model_sftid]}"
                    )
                    dataset_with_ref_rewards_path = f"{dataset_with_ref_rewards_path_prefix}/{dataset_with_ref_rewards}"

                    for loss in losses:
                        for lr in learning_rates:
                            for beta in betas[loss]:
                                # max_grad_norm = 1e3 * beta
                                for train_num_ref_rewards in train_num_ref_rewards_list:
                                    jobid = f"{dataset_with_ref_rewards}-{loss}-numref{train_num_ref_rewards}-lr{lr}-beta{beta}-normbeta"
                                    run_name = f"{job_name}/{jobid}"
                                    commands.append(
                                        (
                                            "sbatch "
                                            f"-p large512 "
                                            f"-t 48:00:00 "
                                            f"-N {num_nodes_per_job} "
                                            f"-o {stdout_root}/out/{jobid}.out "
                                            f"-e {stdout_root}/out/{jobid}.err "
                                            "./cscs-shared-submit-scripts/recursive-unattended-accelerate.sh "
                                            f"-m post_training.train_preference "
                                            f"accelerate_config=src/post_training/configs/accelerate/ds-zero3.yaml "
                                            f"dataset={dataset_with_chosen_rewards} "
                                            f"dataset_args.dataset_name='{dataset_with_ref_rewards_path}' "
                                            f"model={model} "
                                            f"model_args.model_name_or_path='{sft_model_path}' "
                                            f"training_args.max_grad_norm={max_grad_norm} "
                                            f"training_args.gradient_accumulation_steps={accumulation_steps} "
                                            f"training_args.per_device_train_batch_size={per_device_train_batch_size} "
                                            f"training_args.learning_rate={lr} "
                                            f"training_args.loss_type={loss} "
                                            f"training_args.normalize_beta_by_length=true "
                                            f"training_args.num_ref_rewards={train_num_ref_rewards} "
                                            f"training_args.beta={beta} "
                                            f"job_subdir={run_name} "
                                            f"wandb.run_name={run_name} "
                                            f"'wandb.tags=[prod,{job_name}]' "
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
