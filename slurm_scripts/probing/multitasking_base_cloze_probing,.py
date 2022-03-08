from itertools import product
from os import path
import os
import subprocess

JOB_NAME = f'multitasking_base_cloze_probing'

out_dir = path.join(os.getcwd(), f'slurm_scripts/outputs/{JOB_NAME}')
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, 'commands.txt')
base_dir = "/share/data/speech/shtoshni/research/state-probes"


model_names = [
    'size_base_epochs_100_patience_10_seed_100',

    "size_base_epochs_100_patience_10_state_0.1_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.1_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.1_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.2_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.2_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.2_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.3_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.3_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.3_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.4_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.4_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.4_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.5_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.5_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.5_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.6_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.6_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.6_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.7_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.7_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.7_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.8_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.8_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.8_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_0.9_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.9_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_0.9_targeted_text_seed_100",

    "size_base_epochs_100_patience_10_state_1.0_all_text_seed_100",
    "size_base_epochs_100_patience_10_state_1.0_random_text_seed_100",
    "size_base_epochs_100_patience_10_state_1.0_targeted_text_seed_100",
]

model_names = ['multitasking_' + model_name for model_name in model_names]


base_model_path = '/share/data/speech/shtoshni/research/state-probes/models/'
common_options = [
    [path.join(path.join(base_model_path, model_name), 'best/doc_encoder') for model_name in model_names]
]


with open(out_file, 'w') as out_f:
    for option_comb in product(*common_options):
        base = '{}/slurm_scripts/probing/cloze_run.sh '.format(base_dir)
        cur_command = base
        for value in option_comb:
            cur_command += (value + " ")

        cur_command = cur_command.strip()
        out_f.write(cur_command + '\n')

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {} --constraint 2080ti".format(out_dir, out_file, JOB_NAME), shell=True)
