import sys

sys.path.append("/private/home/shtoshni/research/state-probes")
sys.path.append("/private/home/shtoshni/research/state-probes/shtoshni_lm")
from shtoshni_lm.main import main as job_fn
from itertools import product
import submitit


common_options = [
    "--seed 10 --epochs 100 --base_model_dir /private/home/shtoshni/research/state-probes/models --base_data_dir /private/home/shtoshni/research/state-probes/ --use_wandb --num_dev 50"
]

rap_prob_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
models_desc = (
    [f"--add_state ras --rap_prob {rap_prob}" for rap_prob in rap_prob_list]  # RAP models
    + [f"--add_state multitask --rap_prob {rap_prob}" for rap_prob in rap_prob_list]  # Multitask models
    + [f"--add_state explanation"]  # Explanation models
    + [""]  # Baseline
)

raw_comb_list = [common_options, models_desc]
parsed_list = []
for comb_list in product(*raw_comb_list):
    command_line_list = []
    for elem in comb_list:
        if elem:
            command_line_list.extend(elem.split(" "))

    # print(command_line_list)
    parsed_list.append(command_line_list)

executor = submitit.AutoExecutor(
    folder="/private/home/shtoshni/research/state-probes/slurm_scripts/outputs/aug_comparison"
)
executor.update_parameters(
    timeout_min=4320,
    slurm_partition="learnlab,learnfair,scavenge,devlab",
    gpus_per_node=1,
    cpus_per_task=2,
    nodes=1,
    mem=450,
    exclude="learnfair7491,learnfair7477,learnfair7487,learnfair0725,learnfair0866",
    constraint="volta32gb",
)

jobs = executor.map_array(job_fn, parsed_list)
for job in jobs:
    print(job.job_id)
