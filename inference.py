# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto",
#                                              torch_dtype="bfloat16",
#                                              low_cpu_mem_usage=True)
# model.eval()

# prompt = "Write a python function that takes a list of numbers and returns their sum."
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# outputs = model.generate(**inputs, max_new_tokens=256)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM

checkpoint = "/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-bs1024-ademamix/checkpoints/a4f7e2d9c8b16a53/checkpoint-4462"


def main():
    # Create an LLM
    llm = LLM(model=checkpoint, dtype="bfloat16", tensor_parallel_size=4)

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    conversation = [
        {"role": "user", "content": "tell me a joke"},
    ]
    outputs = llm.chat(conversation, use_tqdm=False)
    print_outputs(outputs)


if __name__ == "__main__":
    main()