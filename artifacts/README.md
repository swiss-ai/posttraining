```bash
# Datasets
python -c "import datasets; datasets.load_dataset('allenai/tulu-3-sft-olmo-2-mixture-0225').save_to_disk('artifacts/shared/datasets/sft/olmo2-32b-sft')"
python -c "import datasets; datasets.load_dataset('allenai/olmo-2-0325-32b-preference-mix').save_to_disk('artifacts/shared/datasets/preference/olmo2-32b-preference')"

# Models
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir artifacts/shared/models/llama3-1b

# Reward Models
huggingface-cli download Skywork/Skywork-Reward-V2-Llama-3.1-8B --local-dir artifacts/shared/reward-models/skywork-llama3-8b
huggingface-cli download Skywork/Skywork-Reward-V2-Qwen3-8B --local-dir artifacts/shared/reward-models/skywork-qwen3-8b
huggingface-cli download RLHFlow/ArmoRM-Llama3-8B-v0.1 --local-dir artifacts/shared/reward-models/armorm-llama3-8b
# Remote code is outdated for ArmoRM. Needs a patch.
# Delete the lines containing "LLAMA_INPUTS_DOCSTRING" in modeling_custom.py
```
