# SwissAI Post-Training Codebase

## Repository structure
```
└── reproducibility-scripts            # scripts to reproduce SFT and alignment experiments
└── src/                               # Source code directory
    └── post_training/                  # Core package for post-training tasks
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
        |   └── train-preference.yaml             # Hydra config for preference (e.g. training arguments).
        ├── model_merging/                    # Model merging module.
        │   └── model_merging.py                  # Implements model merging using mergekit.
        ├── data_sft/                         # SFT Data related logic.
        │   ├── dataset_mixture.py                # Creates SFT dataset mixtures (logic in utils_for_dataset.py).
        │   ├── dataset_split.py                  # Generates (stratified) evaluation splits.
        │   ├── prompt_generation_ratios.py       # Calculates generation ratio (=completion length / prompt length)
        │   ├── tokenization.py                   # Configures tokenizer, BOS/PAD/EOS tokens, and chat templates.
        │   ├── utils_for_dataset.py              # Handles dataset loading, dataset mixing, message format conversion, and tokenization.
        │   └── utils_for_gen_ratio.py            # Contains all generation ratio logic.
        ├── data_alignment/                   # Preference Data related logic.
        ├── trainers/                         # Core training logic.
        |   ├── dpo.py                            # DPO experiments code.
        |   ├── preference.py                     # QRPO experiments code.
        │   └── sft.py                            # SFT and Prompt-Loss-Weight training modules.
        ├── utils/                            # Infra utility functions.  
        └── train_sft.py                  # SFT training entrypoint
        └── train_preference.py           # Preference training entrypoint
```

## Reproducibility

The `reproducibility-scripts` directory includes scripts to generate SLURM jobs for training models like Apertus 8B and 70B.
Below are example `sbatch` scripts for training Apertus 8B and 70B checkpoints, generated using a template in
[reproducibility-scripts/sft/0-apertus-template/generate_submit.py](https://github.com/swiss-ai/posttraining/tree/main/reproducibility-scripts/sft/0-apertus-template/generate_submit.py).

To create new experiments copy the directory `reproducibility-scripts/sft/0-apertus-template/` and rename `0-apertus-template` to your new experiment.
Then run the submit script inside to create sub-experiments whose submit script and SLURM logs will be recorded under.

The same for alignment in `/reproducibility-scripts/alignment-apertus-swissaiformat`.

For example  `reproducibility-scripts/sft/0-apertus-template/generate_submit.py`
creates `reproducibility-scripts/sft/0-apertus-template/some-sub-experiment-2025-08-07-17-14/submit.sh`

<details>
<summary>Apertus 70B sbatch command</summary>

```bash
sbatch -N 64 -p large512 -t 48:00:00 \
 -o reproducibility-scripts/sft/0-apertus-template/some-sub-experiment-2025-08-07-17-14/out/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch.out \
 -e reproducibility-scripts/sft/0-apertus-template/some-sub-experiment-2025-08-07-17-14/out/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch.err \
 ./cscs-shared-submit-scripts/recursive-unattended-accelerate.sh -m post_training.train_sft \
 dataset=apertus-sft-mixture-1 \
 model=apertus-70b \
 model_args.model_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828 \
 tokenizer_args.tokenizer_name_or_path=/capstor/store/cscs/swissai/infra01/pretrain-checkpoints/apertus/Apertus70B-tokens15T-it1155828 \
 trainer=plw \
 accelerate_config=src/post_training/configs/accelerate/ds-zero3.yaml \
 plw_args.prompt_loss_weight=0.0 \
 training_args.gradient_accumulation_steps=1 \
 training_args.per_device_train_batch_size=2 \
 training_args.optim=adamw_torch \
 training_args.learning_rate=2e-06
 tokenizer_args.chat_template_name=tulu \
 training_args.num_train_epochs=1 \
 artifacts_subdir=private \
 job_subdir=0-apertus-template/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch \
 wandb.run_name=0-apertus-template/Apertus70B-tokens15T-it1155828-apertus-sft-mixture-1-bs512-lr2e-06-epochs1-adamw_torch \
 wandb.tags=[prod,plw,default,0-apertus-template] \
 resuming.resume=True
```
</details>

These scripts configure the model, dataset, and override training parameters from `src/post_training/configs/train-sft.yaml`.
Training outputs are saved to `artifacts/private/outputs/train_sft/{job_subdir}`.

> [!IMPORTANT]
> **Checkpointing**:
> Checkpointing only works if `resuming.resume` is set to True in the config.

### Customizing Training Runs

#### Adding a Dataset
To add a new dataset, create a YAML file in `src/post_training/configs/dataset` and update the `dataset` field in `src/post_training/configs/train-sft.yaml` (e.g. for SFT) to reference the new file.
We also recommend caching datasets to `artifacts/{artifacts_subdir}/datasets` instead of loading from HF directly.

**Example: `apertus-sft-mixture-1.yaml`**
```yaml
# @package _global_
dataset_name: apertus-sft-mixture-1
dataset_args:
  dataset_name: /capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures/apartus-sft-mixture-1
  train_split: # Only required for DatasetDict objects
    name: null
  eval_split:
    name: null
training_args:
  eval_strategy: "no"
  eval_steps: null
```

#### Configuring Loss Functions
The SFT trainer supports: PLW, LN-PLW, and IRL. Set the desired mode in `src/post_training/configs/train-sft.yaml`:
- **PLW/LN-PLW**: Applies `prompt_loss_weight` to the prompt token loss and can normalize everything together or separate the
normalization for prompt and completion (LN).

<details>
<summary>Definitions</summary>
For a sequence of tokens $x = [x_{1}, ..., x_{n}]$, with a prompt $[x_1, \dots, x_m]$ and a completion $[x_{m+1}, \dots, x_n]$.

- **PLW**: Performs a weighted MLE objective where the prompt tokens are weighted by $w_i \in [0,1]$ and $w_i = 1$ if $x_i$ is in the completion.
$$\mathcal{L}_{\text{PLW}} = - \frac{\sum_{i=1}^{n} w_i \cdot \log p(x_i | x_{\text{<}i}) }{\sum_{i=1}^{N} w_i}$$

- **LN-PLW**: Separates the prompt/completion by weighting their average losses independently and summing them.
$$\mathcal{L}_{\text{LN-PLW}} = \frac{w}{m} \sum_{i=1}^{m} \log p(x_i | x_{\text{<}i}) + \frac{1}{n-m} \sum_{i=m+1}^{n} \log p(x_i | x_{\text{<}i} )$$
</details>

**Example Configuration:**
```yaml
trainer: ln-plw # Options: sft, plw, ln-plw, irl
plw_args:
  prompt_loss_weight: 0.1
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


