from datetime import datetime

models = ["meta-llama-3-1-8b.yaml"]
datasets = ["tulu-3-sft-mixture-split"]

num_epochs = 5
batch_size = 64
max_seq_length = 4096
num_nodes = 2
num_proc_per_node = 4
proc_train_batch_size = 1
accumulation_steps = batch_size // (
    num_nodes * num_proc_per_node * proc_train_batch_size
)
proc_eval_batch_size = 2

learning_rates = [5e-6]
lr_scheduler_type = "linear"
lr_warmup_ratio = 0.03

prompt_loss_weight = [0.0, 0.01, 0.1, 0.5, 1.0]

logging_steps = 200
eval_steps = 400
save_steps = 800

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"plw-sweep"
nruns = 0
for dataset in datasets:
    for model in models:
        for lr in learning_rates:
            for plw in prompt_loss_weight:
                model_config = f"{model}-{dataset}"
                hp_config = f"plw-{plw}-lr-{lr}"
                command = (
                    f"sbatch "
                    f"--nodes {num_nodes} "
                    f"--output=reproducibility-scripts/trl-plw/out-{current_time}/{model_config}/{hp_config}.out "
                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                    f"-m swiss_alignment.trl.plw.train_plw "
                    f"dataset={dataset} "
                    f"model={model} "
                    f"plw_args.prompt_loss_weight={plw} "
                    f"dataset_args.debug_oom=True "
                    f"dataset_args.debug_subsample.train=10_000 "
                    f"dataset_args.debug_subsample.eval=1_000 "
                    f"training_args.max_seq_length={max_seq_length} "
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
                    f"tokenizer_args.chat_template_name=tulu "
                    "outputs_subdir=dev "
                    f"job_subdir={run_name}/{model_config} "
                    f"wandb.run_name={model_config}-plw-{run_name} "
                    "'wandb.tags=[prod,plw]' "
                    "resuming.resume=True "
                )
                print(command)
                nruns += 1

print(nruns)
