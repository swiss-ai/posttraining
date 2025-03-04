import json
import os
import subprocess

command_template = (
    "cd ../../ && python3.10 -m accelerate.commands.launch "
    "--config-file src/swiss_alignment/configs/accelerate/{ds_config} "
    "--num_processes {num_processes} "
    "-m tests.loss.run "
    "model=meta-llama-3-2-1b "
    "dataset=tulu-3-sft-mixture-split-debug "
    "outputs_subdir=dev "
    "job_subdir=sft-dev "
    "wandb.run_name=dev-sft "
    "resuming.resume=False "
    "training_args.gradient_accumulation_steps=1 "
    "training_args.logging_steps=1 "
    "training_args.eval_on_start=false "
    "training_args.average_tokens_across_devices=true "
    "training_args.per_device_train_batch_size={batch_size} "
    "+train_loss_dir={train_loss_dir}"
)

commands = [
    command_template.format(
        ds_config="ds1-acc1-4xN.yaml", num_processes=2, batch_size=2, train_loss_dir="out/exp_np_2_bs_2"
    ),
    # command_template.format(
    #     ds_config="ds1-acc1-4xN.yaml", num_processes=2, batch_size=4, train_loss_dir="out/exp_np_2_bs_4"
    # ),
    # command_template.format(
    #     ds_config="ds1-acc1-4xN.yaml", num_processes=4, batch_size=2, train_loss_dir="out/exp_np_4_bs_2"
    # ),
    # command_template.format(
    #     ds_config="ds1-acc1-4xN.yaml", num_processes=4, batch_size=4, train_loss_dir="out/exp_np_4_bs_4"
    # )
]

for cmd in commands:
    print(f"Executing: {cmd}")
    process = subprocess.run(cmd, shell=True)

    if process.returncode != 0:
        assert False, f"Command failed: {cmd}"

print("All commands executed.")

out_dir_path = os.path.join(os.path.dirname(__file__), "out")

log_history_files = []

# Walk through the directory
for root, dirs, files in os.walk(out_dir_path):
    for file in files:
        # Check if the file name matches
        if file == "log_history.json":
            # Add the full path of the file to the list
            log_history_files.append(os.path.join(root, file))

for log_path1, log_path2 in zip(log_history_files, log_history_files[1:]):
    with open(log_path1, "r") as f1, open(log_path2, "r") as f2:
        logs1 = json.load(f1)
        logs2 = json.load(f2)

    if len(logs1) != len(logs2):
        continue

    losses1 = [h['loss'] for h in logs1 if 'loss' in h]
    losses2 = [h['loss'] for h in logs2 if 'loss' in h]

    grad_norm1 = [h['grad_norm'] for h in logs1 if 'grad_norm' in h]
    grad_norm2 = [h['grad_norm'] for h in logs2 if 'grad_norm' in h]

    loss_ratio = [l1 / l2 for l1, l2 in zip(losses1, losses2)]
    grad_norm_ration = [g1 / g2 for g1, g2 in zip(grad_norm1, grad_norm2)]

    exp1 = os.path.basename(os.path.dirname(log_path1))
    exp2 = os.path.basename(os.path.dirname(log_path2))

    print("Comparing ", exp1, " and ", exp2)

    print(f"Loss ratio statistics: min = {min(loss_ratio):.4f}, max = {max(loss_ratio):.4f}, avg = {sum(loss_ratio) / len(loss_ratio):.4f}")
    print(f"Grad Norm ratio statistics: min = {min(grad_norm_ration):.4f}, max = {max(grad_norm_ration):.4f}, avg = {sum(grad_norm_ration) / len(grad_norm_ration):.4f}")

    assert max(loss_ratio) < 2.0
    assert min(loss_ratio) > 0.5

    assert max(grad_norm_ration) < 2.0
    assert min(grad_norm_ration) > 0.5
