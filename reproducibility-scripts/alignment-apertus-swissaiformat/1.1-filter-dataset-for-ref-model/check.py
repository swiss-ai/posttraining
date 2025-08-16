import datasets

# d = datasets.load_from_disk(
#     "/users/smoalla/projects/swiss-alignment/dev/artifacts/private/datasets/alignment-pipeline-v2format/datasets-for-ref-models/swissai-olmo2-32b-preference.yaml-apertus-70b-sft-mixture-7-d0012600a8854237-maxlen4096"
# )

do = datasets.load_from_disk(
    "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-0325-32b-preference-mix-newCompletions"
)

a = 1
# def get_num_conversations(row):
#     row["num_convsersations"] = len(row["conversation_branches"])
#     return row
#
# d = d.map(get_num_conversations, num_proc=260, desc="Adding number of conversations")

import numpy as np

# n = np.array(d['train']['num_convsersations'])
