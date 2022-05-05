from itertools import product
from os import path
import os
import subprocess

JOB_NAME = f"probing_2"

out_dir = path.join(os.getcwd(), f"slurm_scripts/outputs/{JOB_NAME}")
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, "commands.txt")
base_dir = "/share/data/speech/shtoshni/research/state-probes"

model_paths = [
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.1_all_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.25_all_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.5_all_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.75_all_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.1_targeted_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.25_targeted_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.5_targeted_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.75_targeted_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.1_random_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.25_random_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.5_random_text_seed_10/best/doc_encoder",
    "/share/data/speech/shtoshni/research/state-probes/models/epochs_100_patience_10_state_0.75_random_text_seed_10/best/doc_encoder",
]

common_options = [model_paths]

with open(out_file, "w") as out_f:
    for option_comb in product(*common_options):
        # print(option_comb)
        base = "{}/slurm_scripts/probing/run.sh ".format(base_dir)
        cur_command = base
        for value in option_comb:
            cur_command += value + " "

        cur_command = cur_command.strip()
        out_f.write(cur_command + "\n")

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {}".format(out_dir, out_file, JOB_NAME),
    shell=True,
)
