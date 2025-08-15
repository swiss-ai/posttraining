import copy
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from swiss_alignment import utils
from swiss_alignment.data_alignment.linearize_swissaiformat import (
    linearise_sample_for_sft,
)

utils.config.register_resolvers()
logger = logging.getLogger(__name__)


# Prepare a function to generate completions in batches
def generate_completions_batch(llm, batch, tokenizer, config):
    prompts = []
    for row in batch:
        # TODO: hardcoded format.
        # TODO: Will not trigger for tools and reasoning by default.
        # TODO: for some reason row is changed but not batch (good for us.)
        row["conversation_branches"] = [{"messages": []}]
        linear_row_prompt_only = linearise_sample_for_sft(row)
        prompt_text = tokenizer.apply_chat_template(
            linear_row_prompt_only, tokenize=False, add_generation_prompt=True
        )
        # TODO: not a robust behaviour and may depend on the tokenizer:
        # vllm adds an extra eos in llm.generate. We remove it here if the chat template add it.
        if prompt_text.startswith(tokenizer.bos_token):
            prompt_text = prompt_text[len(tokenizer.bos_token) :]
        prompts.append(prompt_text)

    sampling_params = SamplingParams(
        temperature=config.model_generation_config.temperature,
        top_p=config.model_generation_config.top_p,
        n=config.n_completions,
        max_tokens=config.max_new_tokens,
    )

    # Generate completions
    outputs = llm.generate(prompts, sampling_params)

    # TODO: the output is assumed to be a test response, no tool use, reasoning separation, etc.
    rows_result = []
    for i, (output, row) in enumerate(zip(outputs, batch)):
        completions = [completion.text.strip() for completion in output.outputs]
        conv_branch_row = copy.deepcopy(row)
        for completion in completions:
            new_conv_branch = {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "response",
                                "content": completion,
                                "metadata": {
                                    "is_reference_completion": True,
                                    "model": str(
                                        Path(
                                            config.model_args.model_name_or_path
                                        ).resolve()
                                    ),
                                },
                            }
                        ],
                    }
                ]
            }
            conv_branch_row["conversation_branches"] = [new_conv_branch]
            linear_chat = linearise_sample_for_sft(conv_branch_row)
            chat_tokens = tokenizer.apply_chat_template(linear_chat, tokenize=True)
            context_tokens = tokenizer.apply_chat_template(
                linear_chat[:-1], tokenize=True, add_generation_prompt=True
            )
            chat_tokens_len = len(chat_tokens)
            context_tokens_len = len(context_tokens)
            if chat_tokens_len <= config.max_seq_len:
                new_conv_branch["messages"][0]["parts"][-1]["metadata"][
                    "chat_num_tokens"
                ] = chat_tokens_len
                new_conv_branch["messages"][0]["parts"][-1]["metadata"][
                    "context_num_tokens"
                ] = context_tokens_len
                row["conversation_branches"].append(new_conv_branch)
        rows_result.append(row)

    return datasets.Dataset.from_list(rows_result)


def compute_subpartition_start_end_indices(
    partition_start_idx, partition_end_idx, subpartition_number, num_subpartitions
):
    subpartition_size = math.ceil(
        (partition_end_idx - partition_start_idx) / num_subpartitions
    )
    start_idx = partition_start_idx + subpartition_number * subpartition_size
    end_idx = partition_start_idx + (subpartition_number + 1) * subpartition_size
    end_idx = min(end_idx, partition_end_idx)

    return start_idx, end_idx


@hydra.main(
    version_base=None, config_path="../configs", config_name="generate-ref-completions"
)
def main(config: DictConfig) -> None:
    config = utils.config.setup_config_and_resuming(config)
    random.seed(config.seed)

    tp_size = config.model_vllm_config.tensor_parallel_size
    cuda_devices = [config.subpartition_number * tp_size + i for i in range(tp_size)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_devices))
    logger.info(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Model
    llm = LLM(
        model=config.model_args.model_name_or_path,
        dtype=config.model_args.torch_dtype,
        model_impl=config.model_vllm_config.model_impl,
        tensor_parallel_size=config.model_vllm_config.tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_args.model_name_or_path)

    # End index is exclusive
    num_subpartitions = (
        config.num_gpus_per_node // config.model_vllm_config.tensor_parallel_size
    )
    (
        subpartition_start_idx,
        subpartition_end_idx,
    ) = compute_subpartition_start_end_indices(
        config.partition_start_idx,
        config.partition_end_idx,
        config.subpartition_number,
        num_subpartitions,
    )

    if subpartition_start_idx >= subpartition_end_idx:
        logger.info("Subpartition is empty. Exiting.")
        return

    subpartition_data = datasets.load_from_disk(config.dataset_args.dataset_name)[
        config.split
    ].select(range(subpartition_start_idx, subpartition_end_idx))

    # Handle resuming.
    resuming_dir = Path.cwd()
    # Checkpoints are saved as `checkpoint-{last-relative-index-processed-in-the-subpartition}`.
    already_processed_samples = max(
        (
            int(item.name.split("-")[-1])
            for item in resuming_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint-")
        ),
        default=0,
    )
    if already_processed_samples == len(subpartition_data):
        logger.info(
            "All samples in the subpartition have already been processed. Exiting."
        )
        return

    local_start_idx = already_processed_samples  # 64, 128, ...
    if local_start_idx > 0:
        logger.info(
            f"Resuming from checkpoint-{local_start_idx}. Processing from sample {local_start_idx}."
        )

    pbar = tqdm(total=len(subpartition_data), desc="Generating completions")
    pbar.update(local_start_idx)
    while local_start_idx < len(subpartition_data):
        current_slice = (
            local_start_idx,
            min(local_start_idx + config.save_interval, len(subpartition_data)),
        )
        current_slice_data = subpartition_data.select(range(*current_slice))
        local_end_idx = local_start_idx + len(current_slice_data)

        current_slice_data = generate_completions_batch(
            llm, current_slice_data, tokenizer, config
        )

        # current_slice_data = subpartition_data.select(range(*current_slice))
        # current_slice_data = current_slice_data.map(
        #     lambda _, idx: {"ref_completions": processed_chunk[idx]}, with_indices=True
        # )
        save_path = resuming_dir / f"checkpoint-{local_end_idx}"
        current_slice_data.save_to_disk(save_path)
        logger.info(f"Saved checkpoint-{local_end_idx} successfully!")

        pbar.update(len(current_slice_data))

        local_start_idx = local_end_idx  # Update start index for the next chunk

    logger.info("Completions generated and saved successfully!")


if __name__ == "__main__":
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_NET"] = "Socket"
    # os.environ["NCCL_DEBUG"] = "INFO"
    main()
