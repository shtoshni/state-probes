from itertools import product
from os import path
import os
import subprocess

JOB_NAME = f'base_probing'

out_dir = path.join(os.getcwd(), f'slurm_scripts/outputs/{JOB_NAME}')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/state-probes"


model_names = [
    'oracle_size_base_epochs_100_patience_10_state_1.0_all_text_seed_60',
    'oracle_size_base_epochs_100_patience_10_state_1.0_random_text_seed_60',
    'oracle_size_base_epochs_100_patience_10_state_1.0_targeted_text_seed_60',

    'size_base_epochs_100_patience_10_state_0.1_targeted_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.25_targeted_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.5_targeted_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.75_targeted_text_seed_60',

    'size_base_epochs_100_patience_10_state_0.1_all_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.25_all_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.5_all_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.75_all_text_seed_60',

    'size_base_epochs_100_patience_10_state_0.1_random_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.25_random_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.5_random_text_seed_60',
    'size_base_epochs_100_patience_10_state_0.75_random_text_seed_60',
]


base_model_path = '/share/data/speech/shtoshni/research/state-probes/models/'
common_options = [
    [path.join(path.join(base_model_path, model_name), 'best/doc_encoder') for model_name in model_names]
]


with open(out_file, 'w') as out_f:
    for option_comb in product(*common_options):
        # print(option_comb)
        base = '{}/slurm_scripts/probing/run.sh '.format(base_dir)
        cur_command = base
        for value in option_comb:
            cur_command += (value + " ")

        cur_command = cur_command.strip()
        out_f.write(cur_command + '\n')

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {} --constraint 2080ti".format(out_dir, out_file, JOB_NAME), shell=True)
