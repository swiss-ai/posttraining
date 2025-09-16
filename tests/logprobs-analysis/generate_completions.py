import argparse
import json
import os

from datasets import load_from_disk
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Prepare a function to generate completions in batches
def generate_completions_batch(prompts, tokenizer, llm, args):
    # Prepare prompts
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, n=args.n_completions, max_tokens=4096
    )

    # Generate completions
    outputs = llm.generate(inputs, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        user_content = output.prompt
        completions = [completion.text.strip() for completion in output.outputs]
        results.append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(completions)},
            ]
        )

    return results

def extract_prompt(sample):
    prompt = None
    for item in sample['chosen']:
        if item['role'] == 'user':
            if prompt is None:
                if isinstance(item['content'], str):
                    prompt = item['content']
                elif isinstance(item['content'], dict) and ('parts' in item['content']) and (len(item['content']['parts']) > 0):
                    prompt = item['content']['parts'][0]['text']
                else:
                    raise ValueError("Unexpected content format in user prompt.")
            else:
                raise ValueError("Multiple user prompts found in sample.")
    if prompt is None:
        raise ValueError("No user prompt found in sample.")
    return prompt


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Generate completions for dataset partitions"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model to generate with"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset to use prompts from",
    )
    parser.add_argument(
        "--partition",
        type=str,
        required=True,
        help="Partition of the dataset to use (e.g., 'train', 'validation', 'test')",
    )
    parser.add_argument(
        "--sample_indexes",
        type=int,
        nargs="+",
        required=True,
        help="Indexes of prompts to use",
    )
    parser.add_argument(
        "--n_completions",
        type=int,
        required=True,
        help="Number of completions from reference model for each sample",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the dataset with completions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=128,
        help="Batch size for vllm inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_p", type=float, required=False, default=0.9, help="Top-p for generation"
    )
    parser.add_argument(
        "--device", type=str, required=False, default="0", help="Cuda device number to use"
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    dataset_path = args.dataset_path
    partition = args.partition
    sample_indexes = args.sample_indexes
    batch_size = min(args.batch_size, len(sample_indexes))

    # Load your dataset from the datasets library
    data = load_from_disk(dataset_path)[partition].select(sample_indexes)

    # Initialize the LLM model
    llm = LLM(model=args.model_path,
              dtype=torch.bfloat16,
              model_impl="vllm",
              tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Apply the batch generation function to your dataset
    batched_data = []
    batch = []
    total_batches = (len(data) + batch_size - 1) // batch_size  # Calculate total number of batches

    with tqdm(total=total_batches, desc="Generating completions") as pbar:
        for sample in data:
            batch.append(sample)
            if len(batch) == batch_size:
                # Process the batch
                prompts = [extract_prompt(sample) for sample in batch]
                batched_data.extend(generate_completions_batch(prompts, tokenizer, llm, args))
                batch = []
                pbar.update(1)

    data = data.map(
        lambda _, idx: {"completions": batched_data[idx]}, with_indices=True
    )
    data.save_to_disk(args.save_path)
    print(f"Completions generated and saved {args.save_path} successfully!")

if __name__ == "__main__":
    main()
