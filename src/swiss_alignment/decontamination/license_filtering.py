import argparse
import datasets

def main(args):
    assert "02_standardised" in args.dataset_path, "Dataset must be in the 02_standardised directory!"
    dataset = datasets.load_from_disk(args.dataset_path)
    print("Dataset loaded from: ", args.dataset_path)
    print(dataset)
    dataset_name = args.dataset_path.split("/")[-1]
    output_path = args.dataset_path.replace("02_standardised", "03_license_filtered")

    if dataset_name == "tulu-3-sft-mixture":
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["source"] not in ["ai2-adapt-dev/tulu_hard_coded_repeated_10", "ai2-adapt-dev/no_robots_converted"],
            cache_file_name="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter.arrow",
        )
    elif dataset_name == "EuroBlocks-SFT-Synthetic-1124":
        dataset_filtered = dataset
    elif dataset_name == "smoltalk":
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["source"] not in [
                "openhermes-100k",
                "longalign",
                "explore-instruct-rewriting",
            ],
            cache_file_name="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter.arrow",
        )
    elif dataset_name == "The-Tome":
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["dataset"] not in [
                "infini-instruct-top-500k",
                "ultrainteract_trajectories_sharegpt",
                "qwen2-72b-magpie-en",
            ],
            cache_file_name="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter.arrow",
        )
    elif dataset_name == "AceReason-1.1-SFT":
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["source"] not in ["leetcode"],
            cache_file_name="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter.arrow",
        )
    elif dataset_name == "Llama-Nemotron-Post-Training-Dataset":
        print("Processing Llama-Nemotron-Post-Training-Dataset")
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["license"] in ["cc-by-4.0", "odc-by"],
            cache_file_name="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter.arrow",
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    print("Number of samples removed", len(dataset) - len(dataset_filtered))
    print("Saving filtered dataset to: ", output_path)
    dataset_filtered.save_to_disk(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='The dataset to process')
    args = parser.parse_args()
    main(args)