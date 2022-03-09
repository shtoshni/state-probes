#!/bin/bash

env | grep CUDA
venv=state-probes
base_dir=/share/data/speech/shtoshni/research/state-probes/shtoshni_lm_multitask
args="$@"

echo $args
source /share/data/speech/shtoshni/Development/anaconda3/etc/profile.d/conda.sh
export TORCH_HOME=/share/data/speech/hackathon_2019/.cache/torch
export TRANSFORMERS_CACHE=/share/data/speech/shtoshni/.cache/huggingface
export HF_DATASETS_CACHE=/share/data/speech/shtoshni/.cache/huggingface/datasets
export PYTHONPATH=/share/data/speech/shtoshni/research/state-probes:$PYTHONPATH

gpu_name="$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader)"
if [[ "$gpu_name" == "RTX A6000" ]]; then
    venv=state-probes
else
    venv=state-probes
fi
echo $venv
conda activate "${venv}"
echo "Using environment ${venv}"


echo "Host: $(hostname)"
echo "GPU: $gpu_name"
echo "PYTHONPATH: $PYTHONPATH"
echo "--------------------"

echo "Starting experiment."

python ${base_dir%/}/main.py ${args} --base_model_dir ~/research/state-probes/models/ --base_dir ~/research/state-probes/

conda deactivate
