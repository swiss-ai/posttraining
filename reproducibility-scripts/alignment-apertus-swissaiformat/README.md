# Steps for alignments

## 1. Filter, generate reference off-policy ref samples, and compute logprobs

### 1.1 Filter the dataset

```
for each prompt
    for each completion
        linearize
        apply chat template
        compute length and add tokenized text.
        if too long
            remove the completion
        else
            keep the completion
    if no completions left
        remove the prompt
    else
        keep the prompt

=> result: a dataset with prompts and completions that fit within the max_seq_length
```

### 1.2 and 1.3

Are performed in a embarassignly data parallel way. The dataset is split into chunks, and each chunk is processed in parallel.
Then it's merged after the two steps are done.

### 1.2 Generate reference off-policy ref samples

```
for each prompt
    linearize prompt only
    apply chat template
    compute length and add tokenized text.
    generate completions
```


### 1.3 Compute logprobs



### 1.4 Merge (The pre)
