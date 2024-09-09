trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat

trl sft --model_name_or_path facebook/opt-125m --dataset_name imdb --dataset_text_field text  --output_dir opt-sft-imdb

trl dpo --model_name_or_path facebook/opt-125m --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style --output_dir opt-sft-hh-rlhf
