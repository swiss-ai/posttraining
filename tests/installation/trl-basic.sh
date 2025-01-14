trl chat --model_name_or_path Qwen/Qwen1.5-0.5B-Chat

trl sft --model_name_or_path facebook/opt-125m --dataset_name imdb --dataset_text_field text  --output_dir opt-sft-imdb
# 974/2346 [01:12<01:32, 14.85it/s]

trl dpo --model_name_or_path facebook/opt-125m --dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style --output_dir opt-sft-hh-rlhf
# 66/4110 [00:08<07:26,  10.05it/s]
