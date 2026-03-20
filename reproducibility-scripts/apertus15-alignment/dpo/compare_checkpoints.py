from transformers import AutoModelForCausalLM
import torch

FINAL = "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/cdb3e4f4ad41e0cc394bb92c302ac2eed57e9586"
SFT = "/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364"

print("Loading final model...")
final = AutoModelForCausalLM.from_pretrained(FINAL, torch_dtype=torch.bfloat16)
print("Loading SFT model...")
sft = AutoModelForCausalLM.from_pretrained(SFT, torch_dtype=torch.bfloat16)

final_sd = final.state_dict()
sft_sd = sft.state_dict()

diff_count = 0
for name in final_sd:
    diff = (final_sd[name].float() - sft_sd[name].float()).abs().sum().item()
    if diff > 0:
        diff_count += 1
        print(f"DIFF  {name}: abs_sum={diff:.6f}")

if diff_count == 0:
    print("\nModels are IDENTICAL.")
else:
    print(f"\n{diff_count}/{len(final_sd)} parameters differ.")
