from datetime import datetime
from pathlib import Path

stdout_prefix = "submit-compute-logprobs"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

trained_models_names = ["dpo", "qrpo"]
ref_model_name = "sft"

trained_models_paths = [
    "/users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/swissai-olmo2-32b-preference-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref30-logprobs-skywork-llama3-8b-Npairs2-dpo-adamw_torch-r5e-07-beta5.0/checkpoints/41e2dc9d0e98a4b3/checkpoint-1439",
    "/users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/swissai-olmo2-32b-preference-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref30-logprobs-skywork-llama3-8b-Npairs2-qrpo-adamw_torch-r5e-07-beta5.0/checkpoints/772acd08c3ee9ee6/checkpoint-1439"
]
ref_model_path = "/users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/apertus-8b-sft-10T-mixture-7-7fea1f8c44336360"

datasets_with_completions_for_trained_models = [
    "/users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_dpo_num_completions_1000_temp_1.0_top_p_1.0",
    "/users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_qrpo_num_completions_1000_temp_1.0_top_p_1.0"
]
dataset_with_completions_for_ref_model = "/users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0"

batch_size = 50
beta = 5.0

commands = []
for model_name, model_path, dataset_path in zip(trained_models_names, trained_models_paths, datasets_with_completions_for_trained_models):
    # Compute logprobs for trained model
    jobid = f"compute-logprobs-{model_name}-beta{beta}-on-{model_name}-completions"
    save_path = dataset_path + f"_with_logps_{model_name}_model"
    commands.append(
        (
            "sbatch "
            f"-t 6:00:00 "
            f"-N 1 "
            f"-o {stdout_root}/out/{jobid}.out "
            f"-e {stdout_root}/out/{jobid}.err "
            "./cscs-shared-submit-scripts/unattended-compute-logprobs.sh "
            f"--model_path {model_path} "
            f"--dataset_path {dataset_path} "
            f"--batch_size {batch_size} "
            f"--beta {beta} "
            f"--save_path {save_path} "
        )
    )
    # Compute logprobs for reference model
    jobid = f"compute-logprobs-{ref_model_name}-beta{beta}-on-{model_name}-completions"
    save_path = dataset_path + f"_with_logps_{ref_model_name}_model"
    commands.append(
        (
            "sbatch "
            f"-t 6:00:00 "
            f"-N 1 "
            f"-o {stdout_root}/out/{jobid}.out "
            f"-e {stdout_root}/out/{jobid}.err "
            "./cscs-shared-submit-scripts/unattended-compute-logprobs.sh "
            f"--model_path {ref_model_path} "
            f"--dataset_path {dataset_path} "
            f"--batch_size {batch_size} "
            f"--beta {beta} "
            f"--save_path {save_path} "
        )
    )

# Compute logprobs for reference model on its own dataset
for model_name, model_path in zip(trained_models_names, trained_models_paths):
    # Compute logprobs for trained model
    jobid = f"compute-logprobs-{model_name}-beta{beta}-on-{ref_model_name}-completions"
    save_path = dataset_with_completions_for_ref_model + f"_with_logps_{model_name}_model"
    commands.append(
        (
            "sbatch "
            f"-t 6:00:00 "
            f"-N 1 "
            f"-o {stdout_root}/out/{jobid}.out "
            f"-e {stdout_root}/out/{jobid}.err "
            "./cscs-shared-submit-scripts/unattended-compute-logprobs.sh "
            f"--model_path {model_path} "
            f"--dataset_path {dataset_with_completions_for_ref_model} "
            f"--batch_size {batch_size} "
            f"--beta {beta} "
            f"--save_path {save_path} "
        )
    )
jobid = f"compute-logprobs-{ref_model_name}-beta{beta}-on-{ref_model_name}-completions"
save_path = dataset_with_completions_for_ref_model + f"_with_logps_{ref_model_name}_model"
commands.append(
    (
        "sbatch "
        f"-t 6:00:00 "
        f"-N 1 "
        f"-o {stdout_root}/out/{jobid}.out "
        f"-e {stdout_root}/out/{jobid}.err "
        "./cscs-shared-submit-scripts/unattended-compute-logprobs.sh "
        f"--model_path {ref_model_path} "
        f"--dataset_path {dataset_with_completions_for_ref_model} "
        f"--batch_size {batch_size} "
        f"--beta {beta} "
        f"--save_path {save_path} "
    )
)

# Path from the project root
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")