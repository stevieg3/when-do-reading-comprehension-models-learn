"""
Script for generating .job files. Adapted version of script written by Max Bartolo.
"""

SAVE_STEPS_SCHEDULE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 44, 52, 60, 68, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 300, 316, 332, 348, 364, 380, 396, 428, 460, 492, 524, 556, 588, 620, 652, 684, 716, 748, 780, 812, 844, 876, 908, 940, 972, 1004, 1036, 1100, 1164, 1228, 1292, 1356, 1420, 1484, 1548, 1612, 1676, 1804, 1932, 2060, 2188, 2316, 2444, 2572, 2700, 2828, 2956, 3084, 3212, 3340, 3468, 3596, 3724, 3852, 3980, 4108, 4236, 4364, 4492, 4620, 4748, 4876, 5004, 5132, 5260, 5388, 5516, 5644, 5772, 5900, 6028, 6156, 6284, 6412, 6540, 6668, 6796, 6924, 7052, 7180, 7308, 7436, 7564, 7692, 7820, 7948]
MODEL_PATH = '/SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/models/{}/'

experiments = [
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=28',
        'dataset_config_name': 'dbidaf',
        'version_2_with_negative': ''
    },
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=29',
        'dataset_config_name': 'dbidaf',
        'version_2_with_negative': ''
    },
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=30',
        'dataset_config_name': 'dbidaf',
        'version_2_with_negative': ''
    },
        {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=28',
        'dataset_config_name': 'dbert',
        'version_2_with_negative': ''
    },
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=29',
        'dataset_config_name': 'dbert',
        'version_2_with_negative': ''
    },
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=30',
        'dataset_config_name': 'dbert',
        'version_2_with_negative': ''
    },
        {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=28',
        'dataset_config_name': 'droberta',
        'version_2_with_negative': ''
    },
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=29',
        'dataset_config_name': 'droberta',
        'version_2_with_negative': ''
    },
    {
        'name': 'albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=30',
        'dataset_config_name': 'droberta',
        'version_2_with_negative': ''
    }
]


# Generate the commands
commands = []

for checkpoint in SAVE_STEPS_SCHEDULE:
    this_commands = [
        f"python src/models/run_qa.py "
        f"--model_name_or_path {MODEL_PATH.format(e['name']) + f'checkpoint-{checkpoint}'} "
        f"--dataset_name adversarial_qa "
        f"--dataset_config_name {e['dataset_config_name']} "
        f"{e['version_2_with_negative']}"
        f"--do_eval "
        f"--per_device_eval_batch_size 128 "
        f"--output_dir /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/predictions/{e['name']}-{e['dataset_config_name']}/checkpoint-{checkpoint} "
        f"--overwrite_output_dir "
        f"--overwrite_cache "
        f"--report_to none "
        f"> logs/predictions/{e['name'] + '-' + e['dataset_config_name'] + '-checkpoint={}'.format(checkpoint) + '.log'} 2>&1"
        for e in experiments
    ]

    commands += this_commands


if __name__ == '__main__':
    headers = f"""#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -l tmem=16G
#$ -t 1-{len(commands)}
#$ -l h_rt=24:00:00
#$ -o /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/array.out
#$ -e /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/array.err
#$ -l gpu=true

hostname
date

# Activate conda environment
conda activate rclearn

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn
"""

    job_file = headers + '\n\n'
    for job_id, cmd in enumerate(commands):
        job_file += "test $SGE_TASK_ID -eq {} && sleep 10 && {}".format(job_id+1, cmd)
        job_file += '\n'

    job_file += '\ndate\n'

    # Save the job file
    with open(f'jobs/eval/albert-xlarge-v2-squadv1-eval-adversarialqa.job', 'w') as f:
        f.write(job_file)
