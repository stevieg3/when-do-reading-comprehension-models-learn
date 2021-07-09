"""
Script for generating .job files. Adapted version of script written by Max Bartolo.
"""

SEEDS = [28, 29, 30]
SAVE_STEPS_SCHEDULE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 44, 52, 60, 68, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 300, 316, 332, 348, 364, 380, 396, 428, 460, 492, 524, 556, 588, 620, 652, 684, 716, 748, 780, 812, 844, 876, 908, 940, 972, 1004, 1036, 1100, 1164, 1228, 1292, 1356, 1420, 1484, 1548, 1612, 1676, 1804, 1932, 2060, 2188, 2316, 2444, 2572, 2700, 2828, 2956, 3084, 3212, 3340, 3468, 3596, 3724, 3852, 3980, 4108, 4236, 4364, 4492, 4620, 4748, 4876, 5004, 5132, 5260, 5388, 5516, 5644, 5772, 5900, 6028, 6156, 6284, 6412, 6540, 6668, 6796, 6924, 7052, 7180, 7308, 7436, 7564, 7692, 7820, 7948]

# Generate the commands
commands = []

for seed in SEEDS:
    for checkpoint in SAVE_STEPS_SCHEDULE:
        this_commands = [
            f"python -m torch.distributed.launch --nproc_per_node=2 src/models/run_qa.py "
            f"--model_name_or_path /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/models/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}/checkpoint-{checkpoint} "
            f"--do_predict "
            f"--test_file /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/src/analysis/checklist/squad_ans.json "
            f"--per_device_eval_batch_size 64 "
            f"--output_dir /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/predictions/checklist/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}/checkpoint-{checkpoint} "
            f"--overwrite_output_dir "
            f"--overwrite_cache "
            f"--report_to none "
            f"> logs/checklist/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed={seed}-checkpoint={checkpoint}.log 2>&1"
        ]

        commands += this_commands


if __name__ == '__main__':
    headers = f"""#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -l tmem=16G
#$ -t 1-{len(commands)}
#$ -l h_rt=48:00:00
#$ -o /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/array.out
#$ -e /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/array.err
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y

hostname
date

# Activate conda environment
conda activate rclearn

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
export HF_HOME=/SAN/intelsys/rclearn/cache

cd /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn
"""

    job_file = headers + '\n\n'
    for job_id, cmd in enumerate(commands):
        job_file += "test $SGE_TASK_ID -eq {} && sleep 10 && {}".format(job_id+1, cmd)
        job_file += '\n'

    job_file += '\ndate\n'

    # Save the job file
    with open(f'jobs/checklist/make-predictions.job', 'w') as f:
        f.write(job_file)
