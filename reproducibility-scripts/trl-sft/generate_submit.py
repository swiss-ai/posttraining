from datetime import datetime

models = ["meta-llama-3-1-8b.yaml"]
datasets = ["tulu-3-sft-mixture-split"]

num_epochs = 2
learning_rates = [5e-6]
lr_scheduler_type = "linear"
lr_warmup_ratio = 0.03
batch_size = 128
num_nodes = 2
num_proc_per_node = 4
proc_train_batch_size = 2
accumulation_steps = batch_size // (
    num_nodes * num_proc_per_node * proc_train_batch_size
)
proc_eval_batch_size = 2

logging_steps = 200
eval_steps = 400
save_steps = 800

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"tulu-3-sft"
nruns = 0
for dataset in datasets:
    for model in models:
        for lr in learning_rates:
            model_config = f"{model}-{dataset}"
            hp_config = f"lr-{lr}"
            command = (
                f"sbatch "
                f"--nodes {num_nodes} "
                f"--output=reproducibility-scripts/trl-sft/out-{current_time}/{model_config}/{hp_config}.out "
                "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                f"-m swiss_alignment.trl.train_sft "
                f"dataset={dataset} "
                f"model={model} "
                f"dataset_args.debug_oom=True "
                f"training_args.num_train_epochs={num_epochs} "
                f"training_args.gradient_accumulation_steps={accumulation_steps} "
                f"training_args.per_device_train_batch_size={proc_train_batch_size} "
                f"training_args.per_device_eval_batch_size={proc_eval_batch_size} "
                f"training_args.learning_rate={lr} "
                f"training_args.lr_scheduler_type={lr_scheduler_type} "
                f"training_args.warmup_ratio={lr_warmup_ratio} "
                f"training_args.logging_steps={logging_steps} "
                f"training_args.eval_steps={eval_steps} "
                f"training_args.save_steps={save_steps} "
                "outputs_subdir=shared "
                f"job_subdir={run_name}/{model_config} "
                f"wandb.run_name={model_config}-sft-{run_name} "
                "'wandb.tags=[prod,sft]' "
                "resuming.resume=True "
            )
            print(command)
            nruns += 1

print(nruns)
