# SwissAI Alignment Codebase

Docs for SwissAI users. To update when open-sourcing, if we want to support other users. Editing the template docs should help for this.

## Getting Started

### Preliminaries

- You can log in to Clariden and know the basic setup (https://github.com/swiss-ai/documentation/blob/main/pages/setup_clariden.md).
- You understand Clariden uses SLURM for job scheduling (https://github.com/swiss-ai/documentation/blob/main/pages/slurm_intro.md).
- You can authenticate to GitHub on Clariden through SSH agent forwarding (keys from your laptop follow your SSH connections) (`ssh -T git@github.com`). Check the [troubleshooting section](#github-authentication-through-ssh-agent-forwarding) if you have issues.

### Understand the storage structure

#### Scratch storage: shared and private

Use it to read and write during jobs.

In `iopstor/scratch` (deletion policy: 30 days):  
Shared: `/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared`  
Private: `/iopsstor/scratch/cscs/$USER/projects/swiss-alignment/artifacts/private`

#### Permanent storage: shared

Move valuable permanent assets there, like good final checkpoints, checkpoints to keep for the release, container images...  

In `capstor/store`: `/capstor/store/cscs/swissai/infra01/swiss-alignment/artifacts`.

#### Storage summary:

These paths are symlinked  under the git root and all you have to remember are the following relative paths from the project root.  
Use these relative paths in the code so that it works for everyone.

```
PROJECT_ROOT/                  # In your home directory.
├── artifacts/                 # Anything that lives on other storage scratch and store.
│   ├── shared/                # Shared  (smoalla's iopsstor/scratch).
│   ├── private/               # Private (your iopsstor/scratch).
│   └── store/                 # Store   (infra01's capstor/store).
├── src/                       # Source code.
│   └── swiss_alignment/
...
```

#### Note

Everything in `store` and `shared` should have group write permissions for `infra01` by default.
If at any point this is not the case in some subdirectory you created, run the following:
```bash
cd the-problemetic-subdirectory
chown -R :infra01 .
chmod -R g+wrxs .
setfacl -R -m group::rwx,default:group::rwx,group:infra01:rwx,default:group:infra01:rwx .

# If you want to give another group read access:
GROUP_NAME=some-other-group
setfacl -R -m group:$GROUP_NAME:r-x,default:group:$GROUP_NAME:r-x .
```

### Clone the code and set up the storage

It will look like this:

```bash
$HOME/projects/swiss_alignment
├── dev/    # A dev instance of the repo for development jobs, which can be broken at any time.
└── run/    # A run instance of the repo from where experiments in the queue will read the code.
# Each has the same storage structure above.
```

commands:

```bash
mkdir -p $HOME/projects/swiss-alignment && cd $HOME/projects/swiss-alignment

git clone git@github.com:swiss-ai/swiss-alignment.git dev
git clone git@github.com:swiss-ai/swiss-alignment.git run

for INSTANCE in dev run; do
  INSTANCE=$INSTANCE ./$INSTANCE/artifacts/setup-symlinks.sh
done
```

### Set up credentials

They will be mounted in the containers at runtime.

```bash
# Weights & Biases
echo <my-wandb-api-key> > $HOME/.wandb-api-key
chmod 600 $HOME/.wandb-api-key

# Hugging Face  
echo <my-huggingface-api-key> > $HOME/.hf-token
chmod 600 $HOME/.hf-token
```

## What's next?

Check the sections below:
- [Running the typical training experiment](#running-the-typical-training-experiment): how to run a training experiment with the codebase.
- [Where to change what to do what](#where-to-change-what-to-do-what): how to change the codebase to do different things.
- [Repository structure](#repository-structure): where to find what in the codebase.
- [Contributing](#contributing): good practices and guides for changing specific parts of the codebase.
- [Remote development](#remote-development): every user does this differently. We give you some docs but won't support you for this.
  We however have comprehensive documentation in TODO.
- [Rebuilding the container image](#rebuilding-the-container-image): how to rebuild the container image if you change the dependencies.

## Running the typical training experiment
The `reproducibility-scripts` directory includes scripts to generate SLURM jobs for training models like Apertus 8B and 70B.
Below are example `sbatch` scripts for training Apertus 8B and 70B checkpoints, generated using [generate_submit_apertus_8b.py](https://github.com/swiss-ai/swiss-alignment/blob/main/reproducibility-scripts/trl-plw/apertus/generate_submit_apertus_8b.py)
and [generate_submit_apertus_70b.py](https://github.com/swiss-ai/swiss-alignment/blob/refactor-and-guidelienes/reproducibility-scripts/trl-plw/apertus/generate_submit_apertus_70b.py)

Run these scripts from the dev project root and their outputs will be saved in its
`reproducibility-scripts/trl-plw/out-test` directory.
```
cd $HOME/projects/swiss_alignment/dev
```
In both cases the code used to run will be the one in the stable `$HOME/projects/swiss_alignment/run` repo.

These are foolproof examples that use hardcoded checkpoints from `store`, please don't manually edit them to run
scaled experiments, use the script generators instead.

<details>
<summary>Apertus 8B sbatch command</summary>

```bash
sbatch \
  --nodes 8 \
  --output=reproducibility-scripts/trl-plw/out-test/Apertus8B-tokens7.04T-it1678000-swissai-tulu-3-sft-0225/plw-0.0-lr-5e-06.out \
  ./cscs-shared-submit-scripts/unattended-accelerate.sh \
  -m swiss_alignment.trl.plw.train_sft \
  dataset=swissai-tulu-3-sft-0225 \
  model=apertus-8b \
  model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.04T-it1678000 \
  tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus8B-tokens7.04T-it1678000 \
  trainer=plw \
  accelerate_config=src/swiss_alignment/configs/accelerate/ds-zero1.yaml \
  plw_args.prompt_loss_weight=0.0 \
  training_args.max_seq_length=4096 \
  training_args.num_train_epochs=2 \
  training_args.gradient_accumulation_steps=4 \
  training_args.per_device_train_batch_size=1 \
  training_args.per_device_eval_batch_size=2 \
  training_args.learning_rate=5e-06 \
  training_args.lr_scheduler_type=linear \
  training_args.warmup_ratio=0.03 \
  training_args.eval_strategy=no \
  training_args.eval_on_start=false \
  training_args.save_strategy=steps \
  training_args.save_steps=1000 \
  tokenizer_args.chat_template_name=tulu \
  artifacts_subdir=private \
  job_subdir=apertus3-8b-sweep/Apertus8B-tokens7.04T-it1678000-swissai-tulu-3-sft-0225 \
  wandb.run_name=Apertus8B-tokens7.04T-it1678000-swissai-tulu-3-sft-0225 \
  wandb.tags=[dev,plw] \
  resuming.resume=True
```

</details>

<details>
<summary>Apertus 70B sbatch command</summary>

```bash
sbatch \
  --nodes 32 \
  --output=reproducibility-scripts/trl-plw/out-test/Apertus70B-tokens8T-it739000-swissai-tulu-3-sft-0225/plw-0.0-lr-2e-06.out \
  ./installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/unattended-accelerate.sh \
   -m swiss_alignment.trl.plw.train_sft \
   dataset=swissai-tulu-3-sft-0225 \
   model=apertus-70b \
   model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens8T-it739000 \
   tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens8T-it739000 \
   trainer=plw \
   accelerate_config=src/swiss_alignment/configs/accelerate/ds-zero3.yaml \
   plw_args.prompt_loss_weight=0.0 \
   training_args.max_seq_length=4096 \
   training_args.num_train_epochs=2 \
   training_args.gradient_accumulation_steps=1 \
   training_args.per_device_train_batch_size=1 \
   training_args.per_device_eval_batch_size=2 \
   training_args.learning_rate=2e-06 \
   training_args.lr_scheduler_type=linear \
   training_args.warmup_ratio=0.03 \
   training_args.eval_strategy=no \
   training_args.eval_on_start=false \
   training_args.save_strategy=steps \
   training_args.save_steps=1000 \
   tokenizer_args.chat_template_name=tulu \
   artifacts_subdir=private \
   job_subdir=apertus3-70b-sweep/Apertus70B-tokens8T-it739000-swissai-tulu-3-sft-0225 \
   wandb.run_name=Apertus70B-tokens8T-it739000-swissai-tulu-3-sft-0225 \
   wandb.tags=[dev,plw] \
   resuming.resume=True
```

</details>

These scripts configure the model, dataset, and override training parameters from `src/swiss-alignment/configs/trl-plw.yaml`.
Training outputs are saved to `artifacts/private/outputs/train_sft/{job_subdir}`.

> [!NOTE]
> **Artifact directories**:
> Artifact directories for input and outputs need to be specified in the config file:
> ```bash
> artifacts_subdir: private  # Can be private/shared/store
> input_artifacts_subdir: ${artifacts_subdir}
> output_artifacts_subdir: ${artifacts_subdir}
> ```

> [!NOTE]
> **Debugging**:
> For debugging purposes, defining the following in `trl-plw.yaml` might help:
> ```bash
> dataset_args:
>   debug_oom=True # Sorts dataset by sequence length (largest first) to identify memory issues.
>   debug_subsample.train=10_000 # Subsample size for train split.
>   debug_subsample.eval=100 # Subsample size for eval split.
> ```

> [!IMPORTANT]
> **Checkpointing**:
> Checkpointing only works if `resuming.resume` is set to True in the config.

### Customizing Training Runs

#### Adding a Dataset
To add a new dataset, create a YAML file in `src/swiss-alignment/configs/dataset` and update the `dataset` field in `src/swiss-alignment/configs/trl-plw.yaml` to reference the new file.
We also recommend caching datasets to `artifacts/{artifacts_subdir}/datasets` instead of loading from HF directly.

**Example: `swissai-tulu-3-sft-0225.yaml`**
```yaml
# @package _global_
dataset_name: allenai/tulu-3-sft-olmo-2-mixture-0225
dataset_args:
  dataset_name: ${artifacts_dir}/shared/datasets/swissai/tulu-3-sft-mixture-0225-swissai-hardcoded-15x-no-openai
  train_split:
    name: train
  eval_split:
    name: null
training_args:
  eval_strategy: "no"
  eval_steps: null
```

#### Configuring Loss Functions
The PLW trainer supports four modes: SFT, PLW, LN-PLW, and IRL. Set the desired mode in `src/swiss-alignment/configs/trl-plw.yaml`:
- **SFT**: Trains on full sequence (prompt + completion).
- **PLW/LN-PLW**: Applies `prompt_loss_weight` to prompt loss.
- **IRL**: Uses LN-PLW as backbone for inverse reinforcement learning.

**Example Configuration:**
```yaml
trainer: ln-plw # Options: sft, plw, ln-plw, irl
plw_args:
  prompt_loss_weight: 0.1
irl_args:
  lambda_td: 0.5
  gamma: 1.0
```

To add new trainers, update `src/swiss-alignment/trl/trainers.py` following the project standard.



## Where to change what to do what:
#### Creating an eval split
The `swiss_alignment.trl.dataset_split` module facilitates subsampling a training dataset from a single
source (either a HF dataset or a local path) defined in `config/dataset_split.yaml` into distinct training and evaluation sets.
For instance, the following command was used to subsample the [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)
dataset, generating a split version saved as `tulu-3-sft-mixture-split`.
```bash
exec python -m swiss_alignment.trl.dataset_split
  dataset_args.dataset_name=${artifacts_dir}/shared/datasets/swissai/tulu-3-sft-mixture-0225-swissai-harcoded-15x-no-openai
  dataset_args.output_path=${artifacts_dir}/shared/datasets/swissai/tulu-3-sft-mixture-0225-swissai-harcoded-15x-no-openai-split
  dataset_args.train_split.name=train
  dataset_args.eval_split.name=validation
  dataset_args.eval_split.ratio=0.01
  dataset_args.stratify_by_column=source
```

#### Data Mixtures
A dataset mixture is a tailored combination of datasets, configured in the `config/dataset_mixture.yaml` file.
The `dataset_mixer` attribute is defined as a list of objects, each specifying details for a dataset to include
in the mixture. These objects contain the following fields:
- **`dataset_name`**: The name of the dataset (e.g., `allenai/tulu-3-sft-mixture-split`), not correctly used.
- **`dataset_path`**: The path to the dataset, which can point to a HF db or a local directory (e.g., `${artifacts_dir}/shared/datasets/tulu-3-sft-mixture-plw`).
- **`train_splits`**: A list of split names to include in the training set (e.g., `[train]`).
- **`eval_splits`**: A list of split names to include in the evaluation set (e.g., `[validation]`).
- **`subsample_factor`**: (Optional) Controls the amount of datapoints to save. This can be either a fraction of the dataset (e.g., `0.1` for 10%) or an exact number of samples (e.g., `1_000`). Defaults to the full dataset.
- **`duplication_factor`**: (Optional) Duplicates the dataset x number of times. Defaults to 1.

```bash
exec python -m swiss_alignment.trl.dataset_mixture
```

> [!NOTE]
> Note: This script can also generate the Swiss AI hardcoded prompts dataset. It reads a JSON file containing the prompts and creates a dataset object.
> ```bash
> dataset_mixer:
>   - dataset_name: swissai/hardcoded-prompts
>     dataset_path: ${artifacts_dir}/shared/datasets/swissai/swissai_hardcoded_prompts.jsonl
>     train_splits: [train]
>     eval_splits: []
>     duplication_factor: 10
> columns_to_keep: [messages]
> need_columns: null
> keep_ids: true
> shuffle: false
> save_data_dir: ${artifacts_dir}/shared/datasets/swissai/swissai-harcoded-prompts-10x


#### Model Merging
The `src/swiss_alignment/trl/model_merging/model_merging.py` file initializes the model merging process by calling
`run_merge` from [mergekit](https://github.com/swiss-ai/mergekit). Configure the merge by specifying a technique in
`src/swiss_alignment/configs/model_merging.yaml`, setting the `config_yml` field to, for example,
`src/swiss_alignment/configs/model_merging/linear.yaml`.


## Repository structure
```
└── src/                               # Source code directory
    └── swiss_alignment/                  # Core package for alignment tasks
        ├── configs/                          # Hydra configuration files for flexible experiment setup
        │   ├── accelerate/                       # DeepSpeed configs (stages 1/2/3).
        │   ├── dataset/                          # Dataset definitions (local paths or HF datasets).
        │   ├── model/                            # Model/tokenizer configs.
        │   ├── model-merging/                    # Model merging techniques configs.
        │   ├── dataset-mixture.yaml              # Defines dataset mixtures for training.
        │   ├── dataset-split.yaml                # Configures (stratified) evaluation splits.
        │   ├── model-merging.yaml                # Hydra config for model merging (requires a model_merging/ config).
        │   ├── setup.yaml                        # Base Hydra template inherited by other configs.
        │   └── train-sft.yaml                    # Hydra config for SFT (e.g. training arguments).
        ├── model_merging/                    # Model merging module.
        │   └── model_merging.py                  # Implements model merging using mergekit.
        ├── data_sft/                         # SFT Data related logic.
        │   ├── dataset_mixture.py                # Creates SFT dataset mixtures (logic in utils_for_dataset.py).
        │   ├── dataset_split.py                  # Generates (stratified) evaluation splits.
        │   ├── prompt_generation_ratios.py       # Calculates generation ratio (=completion length / prompt length) for datasets.
        │   ├── tokenization.py                   # Configures tokenizer, BOS/PAD/EOS tokens, and chat templates.
        │   ├── utils_for_dataset.py              # Handles dataset loading, dataset mixing, message format conversion, and tokenization.
        │   └── utils_for_gen_ratio.py            # Contains all generation ratio logic.
        ├── trainers/                         # Core training logic.
        │   └── sft.py                            # SFT and Prompt-Loss-Weight training modules.
        ├── utils/                            # Infra utility functions.  
        └── train_sft.py                  # SFT training entrypoint
```



## Contributing

### Code quality

We use [`pre-commit`](https://pre-commit.com) hooks to ensure high-quality code (it will run when you do `git commit`).

It's installed in the container image with the rest of the dependencies, but if you're developing locally you should install it as well (typically in your Python environment with `pip install pre-commit`).

```bash
# When in the PROJECT_ROOT.
pre-commit install --install-hooks
```

Then every time you commit, the pre-commit hooks will be triggered.
You can also trigger them manually with `pre-commit run --all-files`.

## Remote development

Every user does this differently. We give you some docs but won't support you for this.
Refer to `installation/docker-arm64/CSCS-Clariden-setup/remote-development.md`.

## Rebuilding the container image

Follow the instructions in `installation/docker-arm64/CSCS-Clariden-setup/building-the-image.md`.

## Licenses and acknowledgements

This project is licensed under the LICENSE file in the root directory of the project.

The initial code of this repository has been initiated by the [Python Machine Learning Research Project Template](https://github.com/CLAIRE-Labo/python-ml-research-template)
with the LICENSE.ml-template file.

Additional LICENSE files may be present in subdirectories of the project.

## Getting started - Troubleshooting and extra information (for SwissAI users)

### GitHub authentication through SSH Agent Forwarding

A quick checklist for troubleshooting SSH agent forwarding.
- You can authenticate to GitHub on your laptop: `ssh -T git@github.com`. If not, you don't have an SSH key set up, or it's not added to GitHub.
    ```bash
    # generate new ssh key
    ssh-keygen -t ed25519 -C "your_email@example.com"

    # Add the following to .zshrc
    ssh-add ~/.ssh/id_ed25519

    # add public key id_ed25519.pub to your github account
    # Follow https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

    # Try again
    ssh -T git@github.com
    ```
- The key is forwarded by your SSH agent: `ssh-add -l # List the keys added to your SSH agent.` If not there,
    ```bash
    ssh-add <the path to your SSH key, e.g., ~/.ssh/id_rsa>
    # And add that to your bashrc or equivalent to automatically add it on login.
    ```
- Agent forwarding is enabled on you SSH connection to Clariden: on Clariden `echo "$SSH_AUTH_SOCK"`. Otherwise, you are missing a field `ForwardAgent yes` in your `~/.ssh/config` config for clariden.
- The key is visible on Clariden `ssh-add -l # List the keys added to your SSH agent.` If not, go back 2 steps.
