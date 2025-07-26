#!/bin/bash

export PROJECT_ROOT_AT=$HOME/projects/swiss-alignment/dev
data_root_folder="/capstor/store/cscs/swissai/infra01/posttrain_data/"
decontamination_prompts_path="${data_root_folder}/04_decontaminated/decontamination_prompts"

# if decontamination_prompts_path does not exist, then create it
if [ ! -d "${decontamination_prompts_path}" ]; then
  echo "Decontamination prompts not found at: ${decontamination_prompts_path}"
  echo "Running Python command..."

  python gather_decontamination_prompts --save_path "${decontamination_prompts_path}"
else
  echo "Decontamination prompts already exists: ${decontamination_prompts_path}"
fi

#dataset_name="Llama-Nemotron-Post-Training-Dataset"
#dataset_name="tulu-3-sft-mixture"
#dataset_name="EuroBlocks-SFT-Synthetic-1124"
#dataset_name="smoltalk"
#dataset_name="The-Tome"
dataset_name="AceReason-1.1-SFT"
python $PROJECT_ROOT_AT/src/swiss_alignment/decontamination/decontamination.py \
--decontamination_prompts "${decontamination_prompts_path}" \
--train_dataset "${data_root_folder}/03_license_filtered/${dataset_name}" \
--tokenizer_name "alehc/swissai-tokenizer" \
--report_path "${data_root_folder}/03_license_filtered/${dataset_name}/contamination_reports" \
--ngram_length 8 \
--diff_threshold 0.5 \
--num_proc 10

#--train_dataset_split None \

