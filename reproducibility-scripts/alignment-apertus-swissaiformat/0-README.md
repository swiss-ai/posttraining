# Steps for alignments


## 0. Preliminaries


Make these symlinks
```
cd artifacts/shared/datasets/alignment-pipeline-swissaiformat
ln -s ../../outputs/generate_ref_completions_vllm_swissaiformat/datasets-with-ref-completions datasets-with-ref-completions
ln -s ../../outputs/compute_ref_logprobs_swissaiformat/datasets-with-ref-logprobs datasets-with-ref-logprobs
ln -s ../../outputs/compute_ref_rewards_swissaiformat/datasets-with-ref-rewards datasets-with-ref-rewards  
```

## 1. Filter the dataset prompts and completions

Filter the completions in the dataset that are longer than max_sequence_length, and remove the prompts with no completions lefts.

These completions are problematic for generation, annotation, and training (in DPO and QRPO we need the full sequence logprob as a signal)

### 2. generate reference completion

Are performed in a embarrassingly data parallel way. The dataset is split into chunks, and each chunk is processed in parallel.
Then it's merged after the two steps are done.

### 3. Annotate the reference completions with rewards


### 4. choose the samples for the training datasets


### 5. precompute the reference logprobs for the dataset
