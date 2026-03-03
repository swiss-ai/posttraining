from datasets import load_from_disk

# dataset = load_from_disk("/users/dmelikidze/projects/posttraining/run/artifacts/shared/datasets/alignment-pipeline-swissaiformat/datasets-with-ref-completions/merged/dolci-instruct-dpo-regenerated-qwen06-qwen32-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref10")
# print(dataset)
# print(dataset["conversation_branches"])

# dataset = load_from_disk("/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-0325-32b-preference-mix-newCompletions")
# print(dataset)
# print(dataset["chosen"])

dataset = load_from_disk("/users/dmelikidze/projects/posttraining/run/artifacts/shared/datasets/alignment-pipeline-swissaiformat/train-datasets/hfformat/dolci-instruct-dpo-regenerated-qwen06-qwen32-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref10-logprobs-dpo-format")
print(dataset)
index = 7
print(dataset["train_split"]["rejected"][index])
print("\n --- \n")
dataset2 = load_from_disk("/users/dmelikidze/projects/posttraining/run/artifacts/shared/datasets/alignment-pipeline-swissaiformat/datasets-for-ref-models/dolci-instruct-dpo-regenerated-qwen06-qwen32-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen40962")
print(dataset2["train"]["rejected"][index])
