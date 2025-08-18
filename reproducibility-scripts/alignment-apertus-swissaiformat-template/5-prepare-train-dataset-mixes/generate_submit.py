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

train_dataset = f"{dataset}-{model}-(sftid)-maxlen{max_seq_len}-Nref{NRefDataset}-logprobs-{reward_model}-(train_id)"
"""

stdout_prefix = "8b-70b"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

dataset_with_ref_rewards_path_prefix = "\${artifacts_dir}/shared/datasets/alignment-pipeline-swissaiformat/datasets-with-ref-rewards/merged"

datasets = ["swissai-olmo2-32b-preference"]
splits = ["train_split"]

max_seq_len = 4096

models = ["apertus-8b-sft", "apertus-70b-sft"]
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

dataset_num_ref_reward = 30

reward_models = ["skywork-llama3-8b"]

num_train_pairs_per_prompt = 1

num_nodes_per_job = 1
commands = []
extra_commands = 0
total_nodes_needed = 0
for dataset in datasets:
    for model in models:
        for sftid, sftid_path in sftids[model]:
            for reward_model in reward_models:
                model_sftid = f"{model}-{sftid}"
                dataset_for_model = f"{dataset}-{model_sftid}-maxlen{max_seq_len}"

                commands.append(f"# Dataset-model: {dataset_for_model}")
                commands.append("# Step 5. Prepare train dataset")
                extra_commands += 2

                dataset_with_ref_completions = (
                    f"{dataset_for_model}-Nref{dataset_num_ref_reward}"
                )
                dataset_with_ref_logprobs = f"{dataset_with_ref_completions}-logprobs"
                dataset_with_ref_rewards = f"{dataset_with_ref_logprobs}-{reward_model}"

                dataset_with_ref_rewards_path = (
                    f"{dataset_with_ref_rewards_path_prefix}/{dataset_with_ref_rewards}"
                )

                dataset_type = "train-datasets"
                train_id = f"Npairs{num_train_pairs_per_prompt}"
                train_dataset = f"{dataset_with_ref_rewards}-{train_id}"

                jobid = train_dataset

                commands.append(
                    (
                        "sbatch "
                        f"-N {num_nodes_per_job} "
                        f"-p large512 "
                        f"-t 1:00:00 "
                        f"-o {stdout_root}/out/{jobid}.out "
                        f"-e {stdout_root}/out/{jobid}.err "
                        "./cscs-shared-submit-scripts/unattended.sh "
                        f"python -m swiss_alignment.data_alignment.prepare_train_dataset_swissaiformat "
                        f"dataset={dataset} "
                        f"dataset_args.dataset_name='{dataset_with_ref_rewards_path}' "
                        f"num_pairs_per_prompt={num_train_pairs_per_prompt} "
                        f"dataset_type={dataset_type} "
                        f"dataset_id={train_dataset} "
                        "artifacts_subdir=shared "
                        f"job_subdir={jobid} "
                        "resuming.resume=True "
                    )
                )
                total_nodes_needed += num_nodes_per_job


# Write th submit commands to a new directory where this batch of experiments will be managed)
# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands) - extra_commands} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Total nodes needed:", total_nodes_needed)
