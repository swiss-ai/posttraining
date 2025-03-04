from datetime import datetime

models = ["meta-llama-3-1-8b.yaml"]
datasets = ["tulu-3-sft-mixture-split"]

prompt_loss_weight = [0.0, 0.01, 0.1, 0.2, 0.5, 1.0]
learning_rates = [5e-6, 1e-5, 2e-5]
ds_accumulation_steps = 8

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"promp_loss_weight-sweep"
nruns = 0
for dataset in datasets:
    for model in models:
        for lr in learning_rates:
            for plw in prompt_loss_weight:
                model_config = f"{model}-{dataset}"
                hp_config = f"plw-{plw}-lr-{lr}"
                command = (
                    f"sbatch --nodes 4 "
                    f"--output=reproducibility-scripts/trl-plw/out-{current_time}/{model_config}/{hp_config}.out "
                    "./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                    f"-m swiss_alignment.trl.train_plw "
                    f"dataset={dataset} "
                    f"model={model} "
                    f"plw_args.prompt_loss_weight={plw} "
                    f"training_args.gradient_accumulation_steps={ds_accumulation_steps} "
                    f"training_args.learning_rate={lr} "
                    "outputs_subdir=shared "
                    f"job_subdir={run_name}/{model_config} "
                    f"wandb.run_name={model_config}-{run_name} "
                    "'wandb.tags=[prod,plw]' "
                    "resuming.resume=True "
                )
                print(command)
                nruns += 1

print(nruns)
