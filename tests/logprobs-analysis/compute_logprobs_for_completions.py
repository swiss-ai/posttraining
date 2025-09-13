import argparse
import json
import os
import logging

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

def compute_logprobs(
    data,
    model,
    tokenizer,
    beta,
    sample_idx,
    batch_size=75,
):
    ref_completions = json.loads(data[sample_idx]["ref_completions"][1]["content"])
    prompt = data[sample_idx]["chosen"][0]["content"]
    prompt_length = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )

    inputs = [
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ref_completion},
        ]
        for ref_completion in ref_completions
    ]

    tokenized_inputs = tokenizer.apply_chat_template(
        inputs, return_dict=True, return_tensors="pt", padding=True, truncation=True, max_length=2048
    )

    # all_logits = []
    all_logps = []
    all_lengths = []

    batch_size = min(batch_size, len(ref_completions))
    total_batches = (
        len(ref_completions) + batch_size - 1
    ) // batch_size  # Calculate total number of batches

    with tqdm(total=total_batches, desc="Computing logprobs") as pbar:
        for batch_i in range(total_batches):
            batch_input_ids = tokenized_inputs["input_ids"][
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]
            batch_attention_mask = tokenized_inputs["attention_mask"][
                batch_i * batch_size : (batch_i + 1) * batch_size
            ]

            with torch.no_grad():
                outputs = model(
                    input_ids=batch_input_ids.to("cuda"),
                    attention_mask=batch_attention_mask.to("cuda"),
                )

            logits = outputs.logits[:, :-1, :].cpu()

            labels = batch_input_ids[:, 1:].clone()
            loss_mask = batch_attention_mask
            loss_mask[:, :prompt_length] = 0
            loss_mask = loss_mask[:, 1:].bool()
            labels[
                ~loss_mask
            ] = 0  # dummy token; we'll ignore the losses on these tokens later

            per_token_logps = torch.gather(
                logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)

            per_token_logps[~loss_mask] = 0
            all_logps.extend(per_token_logps.sum(-1).float().tolist())
            all_lengths.extend(loss_mask.sum(-1).tolist())
            # all_logits.extend([logits_sample[loss_mask_sample].float().tolist() for logits_sample, loss_mask_sample in zip(logits, loss_mask)])
            pbar.update(1)

    all_logps = np.array(all_logps)
    all_lengths = np.array(all_lengths)
    lengthnorm_beta = np.clip(beta / (all_lengths + 1e-10), a_min=0, a_max=0.1)
    beta_scaled_logps = beta * all_logps
    lengthnorm_beta_scaled_logps = lengthnorm_beta * all_logps

    return all_logps.tolist(), beta_scaled_logps.tolist(), lengthnorm_beta_scaled_logps.tolist(), all_lengths.tolist()# , all_logits


###################
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Compute rewards and DPO rewards for model completions"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to compute logprobs with"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset with completions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        default=128,
        help="Batch size for vllm inference",
    )
    parser.add_argument(
        "--partition",
        type=int,
        required=True,
        default=0,
        help="Partition index for distributed processing (0, 1, 2 or 3)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        default=5.0,
        help="Batch size for vllm inference",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the dataset with completions",
    )
    parser.add_argument(
        "--device", type=str, required=False, default="0", help="Cuda device index to use"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True
    )

    beta = args.beta
    batch_size = args.batch_size
    partition = args.partition

    ###################

    data_with_completions = load_from_disk(args.dataset_path)

    ###################
    # logits_col = []
    logprobs_col = []
    beta_scaled_logprobs_col = []
    lengthnorm_beta_scaled_logprobs_col = []
    lengths_col = []

    start_idx = (len(data_with_completions) // 4) * partition
    end_idx = (len(data_with_completions) // 4) * (partition + 1) if partition < 3 else len(data_with_completions)

    for sample_idx in tqdm(range(start_idx, end_idx), desc=f"Processing partition {args.partition}, samples {start_idx} to {end_idx}"):
        (
            all_logps,
            beta_scaled_logps,
            lengthnorm_beta_scaled_logprobs,
            all_lengths,
        ) = compute_logprobs(
            data=data_with_completions,
            model=model,
            tokenizer=tokenizer,
            beta=beta,
            sample_idx=sample_idx,
            batch_size=batch_size,
        )
        logprobs_col.append(all_logps)
        beta_scaled_logprobs_col.append(beta_scaled_logps)
        lengthnorm_beta_scaled_logprobs_col.append(lengthnorm_beta_scaled_logprobs)
        lengths_col.append(all_lengths)

    data_with_completions = data_with_completions.select(range(start_idx, end_idx))
    data_with_completions = data_with_completions.add_column("logprobs", logprobs_col)
    data_with_completions = data_with_completions.add_column("beta_scaled_logprobs", beta_scaled_logprobs_col)
    data_with_completions = data_with_completions.add_column("lengthnorm_beta_scaled_logprobs", lengthnorm_beta_scaled_logprobs_col)
    data_with_completions = data_with_completions.add_column("lengths", lengths_col)

    data_with_completions.save_to_disk(args.save_path + f"_partition{args.partition}")
    logger.info(f"Saved updated dataset to {args.save_path}_partition{args.partition}")

if __name__ == "__main__":
    main()
