import bisect

import numpy as np
from datasets import DatasetDict


# All adapted from: https://github.com/davidsvaughn/prompt-loss-weight/blob/main/gen_ratios.py
def compute_generation_ratios(dataset, tokenizer):
    """
    Computes Rg - generation ratios based on  prompt and completion lengths for a single dataset.

    Args:
        dataset (DatasetDict): Dictionary of datasets (e.g., train, validation) with 'messages' field.
        tokenizer: Tokenizer object to encode the text.

    Returns:
        np.ndarray: Sorted array of generation ratios with top and bottom 0.25% trimmed.
    """
    # print splits and number of samples
    dataset_keys = list(dataset.keys())

    # apply instruction template and chat template to each sample
    def format_sample(sample):
        """
        Formats a sample by applying chat templates and computing generation ratio with character counts.
        """
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

    # collect generation ratios over all splits
    gen_ratios = np.sort(
        np.concatenate([dataset[k]["gen_ratio"] for k in dataset_keys])
    )

    # remove top and bottom q quintiles
    q = 0.0025
    gen_ratios = gen_ratios[int(q * len(gen_ratios)) : int((1 - q) * len(gen_ratios))]
    return gen_ratios
