from datetime import datetime
from pathlib import Path

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

datasets = ["swissai-olmo2-32b-preference"]

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

num_nodes_per_job = 1

commands = []
total_nodes_needed = 0
for dataset in datasets:
    for model in models:
        for sftid, sftid_path in sftids[model]:
            model_sftid = f"{model}-{sftid}"
            dataset_for_model = f"{dataset}-{model_sftid}-maxlen{max_seq_len}"
            jobid = f"{dataset_for_model}"
            commands.append(
                (
                    "sbatch "
                    f"-N {num_nodes_per_job} "
                    f"-o {stdout_root}/out/{jobid}.out "
                    f"-e {stdout_root}/out/{jobid}.err "
                    "./cscs-shared-submit-scripts/unattended.sh "
                    f"python -m swiss_alignment.data_alignment.filter_dataset_for_ref_model_swissaiformat "
                    f"dataset={dataset} "
                    f"model={model} "
                    f"model_args.model_name_or_path='{sftid_path}' "
                    f"max_seq_len={max_seq_len} "
                    f"dataset_id={dataset_for_model} "
                    f"job_subdir={jobid} "
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
