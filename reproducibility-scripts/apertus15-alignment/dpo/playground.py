from datasets import load_from_disk

dataset = load_from_disk("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/Qwen3-32B_vs_0.6B")

print("Splits:", dataset.keys())
print("Train split columns:", dataset["train_split"].column_names)
print("Train split features:", dataset["train_split"].features)
print("First train sample:", dataset["train_split"][0])