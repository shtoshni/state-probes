from itertools import product
from os import path
import os
import subprocess

JOB_NAME = f"rap_large"

out_dir = path.join(os.getcwd(), f"slurm_scripts/outputs/{JOB_NAME}")
if not path.isdir(out_dir):
    os.makedirs(out_dir)

# Write commands to output file
out_file = path.join(out_dir, "commands.txt")
base_dir = "/share/data/speech/shtoshni/research/state-probes"

fixed = ["--epochs 100 --patience 10 --use_wandb --model_size large --seed 60"]
state = ["--add_state " + state_type for state_type in ["all", "targeted"]]
rap_prob = [f"--rap_prob {rap_prob}" for rap_prob in [0.1, 0.25, 0.5, 0.75]]

common_options = [fixed, state, rap_prob]


with open(out_file, "w") as out_f:
    for fixed_conf in fixed:
        vanilla_comb = "{}/slurm_scripts/lm/run.sh ".format(base_dir) + fixed_conf
        out_f.write(vanilla_comb + "\n")

    for option_comb in product(*common_options):
        # print(option_comb)
        base = "{}/slurm_scripts/lm/run.sh ".format(base_dir)
        cur_command = base
        for value in option_comb:
            cur_command += value + " "

        cur_command = cur_command.strip()
        out_f.write(cur_command + "\n")

subprocess.call(
    "cd {}; python ~/slurm_batch.py {} -J {} --constraint a6000".format(
        out_dir, out_file, JOB_NAME
    ),
    shell=True,
)
