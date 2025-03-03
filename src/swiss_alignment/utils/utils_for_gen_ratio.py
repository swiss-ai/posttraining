import bisect

import numpy as np
from datasets import DatasetDict


# Adapted from https://github.com/davidsvaughn/prompt-loss-weight/blob/main/gen_ratios.py
def compute_generation_ratios(dataset, tokenizer):
    # print splits and number of samples
    dataset_keys = list(dataset.keys())

    # tokenize and encode batch of samples
    def tokenize_batch(batch):
        tokenized_text = tokenizer(
            batch["text"],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        data = {k: tokenized_text[k] for k in tokenized_text.keys()}

        # use offset_mappings to find the index of the last token of the prompt
        gen_ratio = []
        for offset_mapping, idx in zip(data["offset_mapping"], batch["idx"]):
            num_prompt_tokens = bisect.bisect_right(offset_mapping, (idx,)) - 1
            gen_ratio += [
                (len(offset_mapping) - num_prompt_tokens) / num_prompt_tokens
            ]  # compute Rg with token counts

        data["gen_ratio"] = gen_ratio
        del data["offset_mapping"]
        return data

    # apply instruction template and chat template to each sample
    def format_sample(sample):
        # print(sample)
        # Apply chat template to full conversation
        sample["text"] = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        prompt_txt = tokenizer.apply_chat_template(
            sample["messages"][:1], tokenize=False, add_generation_prompt=True
        )

        sample["idx"] = num_prompt_chars = len(prompt_txt)
        sample["gen_ratio"] = (
            len(sample["text"]) - num_prompt_chars
        ) / num_prompt_chars  # compute Rg with character counts
        return sample

    # format each sample
    dataset = DatasetDict({k: dataset[k].map(format_sample) for k in dataset_keys})
    dataset = DatasetDict(
        {k: dataset[k].map(tokenize_batch, batched=True) for k in dataset_keys}
    )

    # collect generation ratios over all splits
    gen_ratios = np.sort(
        np.concatenate([dataset[k]["gen_ratio"] for k in dataset_keys])
    )

    # remove top and bottom q quintiles
    q = 0.0025
    gen_ratios = gen_ratios[int(q * len(gen_ratios)) : int((1 - q) * len(gen_ratios))]
    return gen_ratios
