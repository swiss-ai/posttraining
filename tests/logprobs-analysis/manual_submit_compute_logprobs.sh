# DPO completions
python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/swissai-olmo2-32b-preference-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref30-logprobs-skywork-llama3-8b-Npairs2-dpo-adamw_torch-r5e-07-beta5.0/checkpoints/41e2dc9d0e98a4b3/checkpoint-1439 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_dpo_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_dpo_num_completions_1000_temp_1.0_top_p_1.0_with_logps_dpo_model \
    --device 0

python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/apertus-8b-sft-10T-mixture-7-7fea1f8c44336360 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_dpo_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_dpo_num_completions_1000_temp_1.0_top_p_1.0_with_logps_sft_model \
    --device 0

# QRPO completions
python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/swissai-olmo2-32b-preference-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref30-logprobs-skywork-llama3-8b-Npairs2-qrpo-adamw_torch-r5e-07-beta5.0/checkpoints/772acd08c3ee9ee6/checkpoint-1439 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_qrpo_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_qrpo_num_completions_1000_temp_1.0_top_p_1.0_with_logps_qrpo_model \
    --device 0

python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/apertus-8b-sft-10T-mixture-7-7fea1f8c44336360 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_qrpo_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_qrpo_num_completions_1000_temp_1.0_top_p_1.0_with_logps_sft_model \
    --device 0

# ref completions
python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/swissai-olmo2-32b-preference-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref30-logprobs-skywork-llama3-8b-Npairs2-dpo-adamw_torch-r5e-07-beta5.0/checkpoints/41e2dc9d0e98a4b3/checkpoint-1439 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0_with_logps_dpo_model \
    --device 0

python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/swissai-olmo2-32b-preference-apertus-8b-sft-10T-mixture-7-7fea1f8c44336360-maxlen4096-Nref30-logprobs-skywork-llama3-8b-Npairs2-qrpo-adamw_torch-r5e-07-beta5.0/checkpoints/772acd08c3ee9ee6/checkpoint-1439 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0_with_logps_qrpo_model \
    --device 0

python /users/smatreno/projects/swiss-alignment/dev/tests/logprobs-analysis/compute_logprobs_for_completions.py \
    --model_path /users/smatreno/projects/swiss-alignment/dev/artifacts/shared/outputs/train_preference/apertus-first-sweep/apertus-8b-sft-10T-mixture-7-7fea1f8c44336360 \
    --dataset_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0 \
    --batch_size 50 \
    --beta 5.0 \
    --save_path /users/smatreno/projects/swiss-alignment/dev/artifacts/private/outputs/logprobs-analysis/magpieair_apertus8b_sft_num_completions_1000_temp_1.0_top_p_1.0_with_logps_sft_model \
    --device 0