import json

import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None,
            config_path="./configs",
            config_name="plot-completions-length-hist")
def main(cfg: DictConfig):
    dataset_with_completions = load_from_disk(cfg.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    fig, ax = plt.subplots(cfg.num_rows, cfg.num_cols, figsize=(cfg.num_rows * 5, cfg.num_cols * 5))
    ax = ax.flatten() if cfg.num_rows * cfg.num_cols > 1 else [ax]

    for sample_idx in cfg.sample_indexes:
        sample = dataset_with_completions[sample_idx]
        prompt = sample["chosen"][0]["content"]
        completions = json.loads(sample["ref_completions"][1]["content"])

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
                {"role": "assistant", "content": completion},
            ]
            for completion in completions
        ]

        tokenized_inputs = tokenizer.apply_chat_template(
            inputs, return_dict=True, return_tensors="pt", padding=True
        )

        completion_lengths = tokenized_inputs["attention_mask"].sum(dim=1).numpy() - prompt_length
        ax[sample_idx].hist(completion_lengths, bins=cfg.num_bins, color=cfg.color, alpha=cfg.alpha)
        ax[sample_idx].set_xlim([cfg.min_length, cfg.max_length])
        ax[sample_idx].set_title(f"Sample Index: {sample_idx}")
        ax[sample_idx].set_xlabel(cfg.x_label)
        ax[sample_idx].set_ylabel(cfg.y_label)
        ax[sample_idx].grid(True)

    plt.tight_layout()
    plt.title(cfg.title)
    plt.show()

if __name__ == "__main__":
    main()
