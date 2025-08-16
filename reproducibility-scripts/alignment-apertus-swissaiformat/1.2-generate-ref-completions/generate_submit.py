from datetime import datetime
from pathlib import Path

import yaml
from datasets import load_from_disk

"""Nomenclature:

dataset = f"{dataset}"
model = f"{sft_model}"
model_sftid = f"{model}-(sftid)"
reward_model = f"{reward_model}"

dataset_for_model = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}"
# dataset_for_model depends on the SFTid through the tokenizer and chat template to filter by max_seq_len

dataset_with_ref_completions = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}"

dataset_with_ref_completions_and_logprobs = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs"

dataset_with_ref_rewards = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs-{reward_model}"
"""

stdout_prefix = "run"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "datasets-with-ref-completions-swissaiformat"

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
            "default",
            "\${artifacts_dir}/shared/models/apertus-8b-sft",
        )
    ],
}

dataset_num_ref_reward = 15
dataset_with_for_model_path_prefix = "\${artifacts_dir}/shared/datasets/alignment-pipeline-swissaiformat/datasets-for-ref-models"
dataset_with_for_model_path_prefix_local = (
    "artifacts/shared/datasets/alignment-pipeline-swissaiformat/datasets-for-ref-models"
)


# Reference numbers for 10 reference completions per prompt:

# 8B
# ~6000 tokens per second per GPU.
# 4096 prompts with 10 completions each take 1h on 1 GPU

# 32B is 4x slower (actually 3x, but 4x to be conservative)
# ~2000 tokens per second per GPU.
# 1024 prompts with 10 completions each take 1h on 1 GPU

# 70B is 4x slower and needs one node with Tensor Parallelism.
# ~X tokens per second per node.
# 1024 prompts with 10 completions each take 1h on 1 node.

# We need N nodes (with 4 GPUs per node) for X prompts in H hours where:
# 8B:  N = X / (16,384 · H)
# 32B: N = X / (4,096 · H)
# 70B: N = X / (1,024 · H)

partition_size = 4096  # N prompts per node
save_interval = (
    1024  # Try to keep it to a reasonable number (like 1-4h), e.g. 1024 for 32B
)
num_nodes_per_job = 1  # 1 node per job
num_gpus_per_node = 4  # 4 GPUs per node

commands = []
total_nodes_needed = 0
for dataset in datasets:
    for split in splits:
        for model in models:
            for sftid, sftid_path in sftids[model]:
                model_sftid = f"{model}-{sftid}"
                dataset_for_model = f"{dataset}-{model_sftid}-maxlen{max_seq_len}"
                dataset_with_ref_completions = (
                    f"{dataset_for_model}-Nref{dataset_num_ref_reward}"
                )

                with open(
                    f"src/swiss_alignment/configs/model/{model}.yaml", "r"
                ) as file:
                    model_config = yaml.safe_load(file)
                tensor_parallel_size = model_config["model_vllm_config"][
                    "tensor_parallel_size"
                ]

                with open(
                    f"src/swiss_alignment/configs/dataset/{dataset}.yaml", "r"
                ) as file:
                    dataset_config = yaml.safe_load(file)

                dataset_for_model_path = (
                    f"{dataset_with_for_model_path_prefix}/{dataset_for_model}"
                )
                dataset_for_model_path_local = (
                    f"{dataset_with_for_model_path_prefix_local}/{dataset_for_model}"
                )
                d = load_from_disk(dataset_for_model_path_local)
                split_name = dataset_config["dataset_args"][split]["name"]

                d_split = d[split_name]
                split_size = len(d_split)

                for partition_start_idx in range(0, split_size, partition_size):
                    partition_end_idx = min(
                        partition_start_idx + partition_size, split_size
                    )
                    jobid = f"{dataset_with_ref_completions}/{split}/{partition_start_idx}-{partition_end_idx}"

                    commands.append(
                        (
                            "sbatch "
                            f"-N {num_nodes_per_job} "
                            f"-p large512 "
                            f"-t 48:00:00 "
                            f"--ntasks-per-node {num_gpus_per_node // tensor_parallel_size} "
                            f"-o {stdout_root}/out/{jobid}.out "
                            f"-e {stdout_root}/out/{jobid}.err "
                            "./cscs-shared-submit-scripts/unattended-generate-ref-completions-with-vllm-swissaiformat.sh "
                            f"model={model} "
                            f"model_args.model_name_or_path='{sftid_path}' "
                            f"dataset={dataset} "
                            f"dataset_args.dataset_name='{dataset_for_model_path}' "
                            f"split={split_name} "
                            f"num_gpus_per_node={num_gpus_per_node} "
                            f"n_completions={dataset_num_ref_reward} "
                            f"max_seq_len={max_seq_len} "
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
