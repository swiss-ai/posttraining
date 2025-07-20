from datetime import datetime

models = ["apertus-8b"]
datasets = ["swissai-tulu-3-sft-0225"]

ds_config = "ds-zero1"  # ds-zero1, ds-zero2, ds-zero3

num_epochs = 2
batch_size = 128
max_seq_length = 4096
num_nodes = 8
num_proc_per_node = 4
proc_train_batch_size = 1
accumulation_steps = batch_size // (
    num_nodes * num_proc_per_node * proc_train_batch_size
)
proc_eval_batch_size = 2

learning_rates = [5e-6]  # 8b: 5e-6; 70b: 2e-6
lr_scheduler_type = "linear"
lr_warmup_ratio = 0.03

trainer = "plw"  # can only take values: sft, plw, ln-plw, irl
prompt_loss_weight = [
    0.0,
]  # where sft -> plw=1.0

logging_steps = 1
save_steps = 1000

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"apertus3-8b-sweep"
nruns = 0
for dataset in datasets:
    for model in models:
        for iter in [
            # "Apertus8B-tokens2T-it478000",
            # "Apertus8B-tokens3T-it716000",
            # "Apertus8B-tokens4T-it954000",
            # "Apertus8B-tokens5T-it1194000",
            # "Apertus8B-tokens6T-it1432000",
            # "Apertus8B-tokens7T-it1670000",
            "Apertus8B-tokens7.04T-it1678000",
            # "Apertus8B-tokens7.09T-it1690000-phase3",
            # "Apertus8B-tokens7.09T-it1690000-phase4",
            # "Apertus8B-tokens7.09T-it1690000-phase4-provenance-flan-short",
            # "Apertus8B-tokens7.09T-it1690000-phase4-megamath-pro-short",
        ]:
            for lr in learning_rates:
                for plw in prompt_loss_weight:
                    model_config = f"{iter}-{dataset}"
                    hp_config = f"{trainer}-{plw}-lr-{lr}"
                    command = (
                        f"sbatch "
                        f"--nodes {num_nodes} "
                        f"--output=reproducibility-scripts/trl-plw/out-{current_time}/{model_config}/{hp_config}.out "
                        f"./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                        f"-m swiss_alignment.trl.plw.train_plw "
                        f"dataset={dataset} "
                        f"model={model}.yaml "
                        f"model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{iter} "
                        f"tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/{iter} "
                        f"trainer={trainer} "
                        f"plw_args.prompt_loss_weight={plw} "
                        f"training_args.max_seq_length={max_seq_length} "
                        # f"training_args.max_grad_norm=1 "
                        f"training_args.num_train_epochs={num_epochs} "
                        f"training_args.gradient_accumulation_steps={accumulation_steps} "
                        f"training_args.per_device_train_batch_size={proc_train_batch_size} "
                        f"training_args.per_device_eval_batch_size={proc_eval_batch_size} "
                        f"training_args.learning_rate={lr} "
                        f"training_args.lr_scheduler_type={lr_scheduler_type} "
                        f"training_args.warmup_ratio={lr_warmup_ratio} "
                        f"training_args.logging_steps={logging_steps} "
                        f"training_args.eval_strategy=no "
                        f"training_args.eval_on_start=false "
                        f"training_args.save_strategy=steps "
                        f"training_args.save_steps={save_steps} "
                        f"tokenizer_args.chat_template_name=tulu "
                        "artifacts_subdir=private "
                        f"job_subdir={run_name}/{model_config} "
                        f"wandb.run_name={model_config} "
                        f"wandb.tags=[prod,{trainer}] "
                        "resuming.resume=True "
                    )
                    print(command)
                    nruns += 1

print(nruns)
