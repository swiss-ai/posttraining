# Steps for alignments

The steps follow the numbered directories. You can reuse this directory and generate new commands or if you want to change the pipeline a bit
copy the whole directory `alignment-apertus-swissaiformat`.

## 0. Preliminaries

Make these symlinks (will symlink `.../datasets/X` to generated datasets in `.../outputs/X`).
```
cd artifacts/shared/datasets/alignment-pipeline-swissaiformat
ln -s ../../outputs/generate_ref_completions_vllm_swissaiformat/datasets-with-ref-completions datasets-with-ref-completions
ln -s ../../outputs/compute_ref_logprobs_swissaiformat/datasets-with-ref-logprobs datasets-with-ref-logprobs
ln -s ../../outputs/compute_ref_rewards_swissaiformat/datasets-with-ref-rewards datasets-with-ref-rewards  
```

## 1. Filter the dataset prompts and completions

Filter the completions in the dataset that are longer than `max_sequence_length`, and remove the prompts with no completions lefts.

These completions are problematic for generation, annotation, and training (in DPO and QRPO we need the full sequence logprob as a signal).

### 2. generate reference completion

Are performed in a embarrassingly data parallel way. The dataset is split into chunks, and each chunk is processed in parallel.
Then it's merged after the generation is done. This can be scaled infinitely with no communication overhead to use all GPUs available.

Done with vllm and is pretty fast.

We recommend sampling around 10 samples per prompt.

### 3. precompute the reference logprobs for the dataset

The logprobs of the reference model (the policy to train) are computed on all the samples in the dataset. This is also done in the same parallel way as the generation.

They are precomputed to avoid having 2 copies of the model during training. This is prohibitive for 70B.

This step technically doesn't have to be done on all the completions in the dataset so far, only the ones which will make it to the training dataset. However, having it here allows to later iterate
faster on selecting the training samples without recomputing the logprobs.

At the moment this is the slowest step, as done with plain huggingface which for 70B uses inefficient pipeline parallel, so feel free to move it to the end before training which will significantly reduce the samples it has to process.

### 4. Annotate the reference completions with rewards

Same as the two previous steps using a reward model. Quite efficient at the moment, as we have an 8B reward model.

### 5. choose the samples for the training datasets and mix training datasets

Select the training samples for each prompt from all the annotated completions for that prompt (e.g. best and worst, from the middle quantiles, etc.). This is an open question.
Atm we use a heuristic (it seems like the most important is to have bad completions, it's not clear how good the rest of the completions should be).

### 6. Train

A final training dataset for a model is identified by all the models and hyperparameters used in the previous steps.
