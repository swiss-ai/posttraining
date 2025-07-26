#!/bin/bash

dataset_path="/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/smoltalk"
python license_filtering.py --dataset_path="${dataset_path}"
