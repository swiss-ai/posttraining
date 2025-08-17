model_name = "../../../artifacts/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-v2-bs1024-lr2e-06-maxgnorm1-epochs1-ademamix/checkpoints/d0012600a8854237/checkpoint-4462"

# model_name = "artifacts/shared/outputs/train_sft/final-run/Apertus70B-tokens15T-longcontext64k-apertus-sft-mixture-7-ln-v2-bs1024-lr2e-06-maxgnorm1-epochs1-ademamix/checkpoints/d0012600a8854237/checkpoint-4462"

import os

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


m = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(model_name).from_pretrained(
    model_name,
    use_fast=True,
)


import time

time.sleep(1000)
