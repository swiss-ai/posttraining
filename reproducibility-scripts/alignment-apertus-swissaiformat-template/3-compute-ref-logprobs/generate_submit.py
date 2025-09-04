from datetime import datetime
from pathlib import Path

import yaml
from datasets import load_from_disk

"""Nomenclature:

dataset = f"{dataset}"
model = f"{sft_model}"
model_sftid = f"{model}-(sftid)"
reward_model = f"{reward_model}"

dataset_for_model = f"{dataset}-{model}-(sft_id)-maxlen{max_seq_len}"
# dataset_for_model depends on the SFTid through the tokenizer and chat template to filter by max_seq_len

dataset_with_ref_completions = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}"

dataset_with_ref_logprobs = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs"

dataset_with_ref_rewards = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs-{reward_model}"

train_datasets = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs-{reward_model}-(train_id)"
"""

stdout_prefix = "70b"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

dataset_for_model_path_prefix = "\${artifacts_dir}/shared/datasets/alignment-pipeline-swissaiformat/datasets-for-ref-models"
dataset_with_ref_completions_path_prefix = "\${artifacts_dir}/shared/datasets/alignment-pipeline-swissaiformat/datasets-with-ref-completions/merged"
datasets = ["swissai-olmo2-32b-preference"]
splits = ["train_split"]

max_seq_len = 4096

models = ["apertus-70b-sft"]
sftids = {
    "apertus-70b-sft": [
        (
            "mixture-7-d0012600a8854237",
            "\${artifacts_dir}/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-v2-bs1024-lr2e-06-maxgnorm1-epochs1-ademamix/checkpoints/d0012600a8854237/checkpoint-4462",
        )
    ],
    "apertus-8b-sft": [
        (
            "10T-mixture-7-7fea1f8c44336360",
            "\${artifacts_dir}/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-7-ln-v2-ademamix/checkpoints/7fea1f8c44336360/checkpoint-8925",
        )
    ],
}

dataset_num_ref_reward = 10

# Reference numbers for 10 reference completions per prompt:

# 8B
# 2048 prompts with 20 (10 off-policy + 10 offline) completions each take 1h on 1 GPU

# 70B
# 256 prompts with 20 completions each take 1h on 1 node.

# We need N nodes (with 4 GPUs per node) for X prompts in H hours where:
# 8B:  N = X / (8192 · H)
# 70B: N = X / (256 · H)

is_partitioned = True
partition_size = 1024  # N prompts per node (1024 for 70b, 8192 for 8b)
save_interval = 256  # Try to keep it to a reasonable number (like 1-2h), e.g. 256 for 70b and 2048 for 8b
num_nodes_per_job = 1  # 1 node per job
num_gpus_per_node = 4  # 4 GPUs per node

commands = []
total_nodes_needed = 0
for dataset in datasets:
    for model in models:
        for sftid, sftid_path in sftids[model]:
            model_sftid = f"{model}-{sftid}"
            dataset_for_model = f"{dataset}-{model_sftid}-maxlen{max_seq_len}"

            commands.append(f"# Dataset-model: {dataset_for_model}")

            commands.append(
                "# Step 3.1 Commands to generate the ref logprobs in parallel"
            )

            dataset_with_ref_completions = (
                f"{dataset_for_model}-Nref{dataset_num_ref_reward}"
            )

            with open(f"src/post_training/configs/model/{model}.yaml", "r") as file:
                model_config = yaml.safe_load(file)
            tensor_parallel_size = model_config["model_vllm_config"][
                "tensor_parallel_size"
            ]

            num_subpartitions = num_gpus_per_node // tensor_parallel_size

            with open(
                f"src/post_training/configs/dataset/{dataset}.yaml", "r"
            ) as file:
                dataset_config = yaml.safe_load(file)

            dataset_for_model_path = (
                f"{dataset_for_model_path_prefix}/{dataset_for_model}"
            )
            dataset_for_model_path_local = f"{dataset_for_model_path_prefix.replace('\${artifacts_dir}', 'artifacts')}/{dataset_for_model}"
            d = load_from_disk(dataset_for_model_path_local)

            dataset_with_ref_completions_path = f"{dataset_with_ref_completions_path_prefix}/{dataset_with_ref_completions}"

            dataset_with_ref_logprobs = f"{dataset_with_ref_completions}-logprobs"
            dataset_type = "datasets-with-ref-logprobs"

            for split in splits:
                split_name = dataset_config["dataset_args"][split]["name"]
                d_split = d[split_name]
                split_size = len(d_split)
                for partition_start_idx in range(0, split_size, partition_size):
                    partition_end_idx = min(
                        partition_start_idx + partition_size, split_size
                    )
                    jobid = f"{dataset_with_ref_logprobs}/{split}/{partition_start_idx}-{partition_end_idx}"

                    commands.append(
                        (
                            "sbatch "
                            f"-N {num_nodes_per_job} "
                            f"-p large512 "
                            f"-t 24:00:00 "
                            f"--ntasks-per-node {num_subpartitions} "
                            f"-o {stdout_root}/out/{jobid}.out "
                            f"-e {stdout_root}/out/{jobid}.err "
                            "./cscs-shared-submit-scripts/unattended-compute-ref-logprobs-swissaiformat.sh "
                            f"model={model} "
                            f"model_args.model_name_or_path='{sftid_path}' "
                            f"dataset={dataset} "
                            f"dataset_args.dataset_name='{dataset_with_ref_completions_path}' "
                            f"split={split_name} "
                            f"num_gpus_per_node={num_gpus_per_node} "
                            f"max_seq_len={max_seq_len} "
                            f"partition_start_idx={partition_start_idx} "
                            f"partition_end_idx={partition_end_idx} "
                            f"save_interval={save_interval} "
                            "artifacts_subdir=shared "
                            f"job_subdir_prefix={dataset_type}/{jobid} "
                            "resuming.resume=True "
                        )
                    )
                    total_nodes_needed += num_nodes_per_job

            # 3.2 Merge command: To use at the end.
            jobid = dataset_with_ref_logprobs
            commands.append("# Step 3.2 Command to merge the ref logprobs.")
            commands.append(
                (
                    "sbatch "
                    f"-N {num_nodes_per_job} "
                    f"-p large512 "
                    f"-o {stdout_root}/out/{jobid}.out "
                    f"-e {stdout_root}/out/{jobid}.err "
                    "./cscs-shared-submit-scripts/unattended.sh "
                    f"python -m post_training.data_alignment.merge_partitions_swissaiformat "
                    f"dataset={dataset} "
                    f"dataset_args.dataset_name='{dataset_for_model_path}' "
                    f"dataset_id={dataset_with_ref_logprobs} "
                    f"dataset_type={dataset_type} "
                    f"is_partitioned={is_partitioned} "
                    f"num_subpartitions={num_subpartitions} "
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
