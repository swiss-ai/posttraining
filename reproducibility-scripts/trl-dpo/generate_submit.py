from datetime import datetime

models = ["apertus3-8b-sft.yaml"]
datasets = ["tulu-3-8b-preference-mixture"]

num_nodes = 4

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = f"swissai-3-dpo-v0-b_exp"
nruns = 0
with open("dpo_submit.sh", "w") as file:
    for beta in [0.1, 0.01, 0.5]:
        for dataset in datasets:
            for model in models:
                model_config = f"{model}-{dataset}"
                command = (
                    f"sbatch "
                    f"--nodes {num_nodes} "
                    f"--output=$HOME/projects/swiss-alignment/dev/reproducibility-scripts/trl-dpo/out-{current_time}/{model_config}/dpo.out "
                    "$HOME/projects/swiss-alignment/dev/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-ds.sh "
                    f"-m swiss_alignment.trl.dpo.train_dpo "
                    f"model={model} "
                    f"training_args.beta={beta} "
                    "outputs_subdir=shared "
                    f"job_subdir={run_name}/{model_config} "
                    f"wandb.run_name={model_config}-{run_name}-{beta} "
                    f"wandb.tags=[prod,dpo] "
                    "resuming.resume=True "
                )
                print(command, file=file)
                nruns += 1

print(nruns)
