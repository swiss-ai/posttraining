import logging
import random
from functools import partial
from itertools import chain
from queue import PriorityQueue

import numpy as np
from accelerate.logging import get_logger
from accelerate.state import PartialState
from datasets import DatasetDict, Sequence, Value

from swiss_alignment import utils

utils.config.register_resolvers()
acc_state = PartialState()
acc_logger = get_logger(__name__)
hydra_logger = logging.getLogger(__name__)


# Adapted from: https://github.com/davidsvaughn/prompt-loss-weight/blob/main/run_plw.py
# shortest pack first histogram packing
def spfhp(seq_lens, chunk_length=2048):
    q = PriorityQueue()
    q.put((0, []))
    idx = seq_lens.argsort()[::-1]
    for i in idx:
        n, pack = seq_lens[i], q.get()
        if n + pack[0] > chunk_length:
            q.put(pack)
            pack = (0, [])
        q.put((n + pack[0], pack[1] + [i]))
    return list(q.queue)


# pack sequences into chunks
def pack(sample, chunk_length=2048, pad_token_id=0):
    # compute packing arrangement
    seq_lens = np.array([len(t) for t in sample["input_ids"]])
    chunks = spfhp(seq_lens, chunk_length=chunk_length)
    random.shuffle(chunks)

    # pack sequences according to arrangement
    result = {}
    for k in sample.keys():
        result[k] = []
        pad_id = pad_token_id if k == "input_ids" else 0
        for chunk in chunks:
            item = list(
                chain(
                    *[sample[k][i] for i in chunk[1]],
                    [pad_id] * (chunk_length - chunk[0]),
                )
            )
            result[k].append(item)

    # add labels (same as input_ids!)
    result["labels"] = result["input_ids"].copy()
    return result


# Function to tokenize and encode a batch of samples, and creates prompt/completion masks.
def tokenize_batch(batch, tokenizer):
    # tokenize and encode text
    tokenized_text = tokenizer(
        batch["text"],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    data = {k: tokenized_text[k] for k in tokenized_text.keys()}

    # use offset_mappings to make prompt/completion masks (idx marks the start of each completion)
    prompt_masks, completion_masks = [], []
    for offset_mapping, idx in zip(data["offset_mapping"], batch["idx"]):
        prompt_masks.append([1 if o[1] < idx else 0 for o in offset_mapping])
        completion_masks.append([0 if o[1] < idx else 1 for o in offset_mapping])

    data["prompt_mask"] = prompt_masks
    data["completion_mask"] = completion_masks
    del data["offset_mapping"]
    return data


# tokenize and pack dataset
def tokenize_and_pack(dataset, tokenizer, config):
    # Tokenize dataset, remove original columns
    tokenized = dataset.map(
        partial(tokenize_batch, tokenizer=tokenizer),
        batched=True,
        remove_columns=list(dataset.features),
    )

    # Cast mask columns to int8
    for col in ["prompt_mask", "completion_mask"]:
        tokenized = tokenized.cast_column(col, Sequence(Value("int8")))

    # Filter sequences longer than max_seq_length
    tokenized = tokenized.filter(
        lambda x: len(x["input_ids"]) <= config.training_args.max_seq_length
    )

    # Print sample count
    acc_logger.info(f"Number of samples: {len(tokenized)}")

    # Pack sequences
    packed = tokenized.map(
        lambda x: pack(
            x,
            chunk_length=config.training_args.max_seq_length,
            pad_token_id=tokenizer.pad_token_id,
        ),
        batched=True,
    )
    packed = packed.cast_column("labels", Sequence(Value("int32")))

    # Calculate and print packing stats
    seq_lens = [len(x) for x in tokenized["input_ids"]]
    total_packed = len(packed) * config.training_args.max_seq_length
    acc_logger.info(f"Packing density:     {100 * sum(seq_lens) / total_packed:.1f}%")
    acc_logger.info(f"Packing compression: {100 * len(packed) / len(tokenized):.1f}%")
    return packed


def prepare_dataset(dataset, tokenizer, config):
    # Helper function to format each sample
    def format_sample(sample):
        # Apply chat template to full conversation
        sample["text"] = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        # Get prompt length for completion index
        prompt_text = tokenizer.apply_chat_template(
            sample["messages"][:1], tokenize=False, add_generation_prompt=True
        )
        sample["idx"] = len(prompt_text)
        return sample

    processed = DatasetDict({})
    for k in dataset.keys():
        split = dataset[k].map(format_sample)
        split = tokenize_and_pack(split, tokenizer, config)
        processed[k] = split
        acc_logger.info(f"Total count of {k} packed sequences: {len(split)}")

    return processed
