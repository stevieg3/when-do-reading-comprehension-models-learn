"""
Script for generating .job files. Adapted version of script written by Max Bartolo.
"""

MODEL_NAME_OR_PATH = 'albert-xlarge-v2'
PER_DEVICE_BATCH_SIZE = 2
SAVE_STEPS_SCHEDULE = "1 2 3 4 5 6 8 10 12 14 16 20 24 28 32 36 44 52 60 68 76 92 108 124 140 156 172 188 204 220 236 252 268 284 300 316 332 348 364 380 396 428 460 492 524 556 588 620 652 684 716 748 780 812 844 876 908 940 972 1004 1036 1100 1164 1228 1292 1356 1420 1484 1548 1612 1676 1804 1932 2060 2188 2316 2444 2572 2700 2828 2956 3084 3212 3340 3468 3596 3724 3852 3980 4108 4236 4364 4492 4620 4748 4876 5004 5132 5260 5388 5516 5644 5772 5900 6028 6156 6284 6412 6540 6668 6796 6924 7052 7180 7308 7436 7564 7692 7820 7948"
MAX_STEPS = 8200
FP16 = '--fp16 True'
LOGGING_STEPS = 1640
SEEDS = [27, 28, 29, 30]
OUTPUT_DIR_ROOT = '/SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/models/'
LEARNING_RATE = 3e-5
ACCUMULATION_STEPS = 16
MAX_SEQ_LENGTH = 384
WARMUP_STEPS = 100

experiments = [
    {
        'run_name': 'mrqa-wu=100-lr=3e5-bs=32-msl=384-seed={}',
        'version_2_with_negative': ''
    }
]


# Generate the commands
commands = []

for seed in SEEDS:
    this_commands = [
        f"python src/models/run_qa.py "
        f"--model_name_or_path {MODEL_NAME_OR_PATH} "
        f"--train_file /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/data/external/mrqa/train/mrqa_train_squad_format.json "
	    f"--validation_file /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/data/external/mrqa/dev/mrqa_dev_squad_format.json "
        f"{e['version_2_with_negative']}"   
        f"--do_train "
        f"--do_eval "
        f"--per_device_train_batch_size {PER_DEVICE_BATCH_SIZE} "
        f"--gradient_accumulation_steps {ACCUMULATION_STEPS} "
        f"--learning_rate {LEARNING_RATE} "
        f"--max_seq_length {MAX_SEQ_LENGTH} "
        f"--output_dir {OUTPUT_DIR_ROOT + MODEL_NAME_OR_PATH + '-' + e['run_name'].format(seed)} "
        f"--overwrite_output_dir "
        f"--overwrite_cache "
        f"--evaluation_strategy steps "
        f"--save_steps_schedule {SAVE_STEPS_SCHEDULE} "
        f"--save_strategy steps "
        f"--report_to wandb "
        f"--seed {seed} "
        f"--run_name {e['run_name'].format(seed)} "
        f"--max_steps {MAX_STEPS} "
        f"--warmup_steps {WARMUP_STEPS} "
        f"{FP16} "
        f"--logging_steps {LOGGING_STEPS} "
        f"> logs/{MODEL_NAME_OR_PATH + '-' + e['run_name'].format(seed) +'.log'} 2>&1"
        for e in experiments
    ]

    commands += this_commands


if __name__ == '__main__':
    headers = f"""#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -l tmem=16G
#$ -t 1-{len(commands)}
#$ -l h_rt=72:00:00
#$ -o /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/array.out
#$ -e /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn/array.err
#$ -l gpu=true

hostname
date

# Activate conda environment
conda activate rclearn

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
export WANDB_PROJECT="albert-xlarge-v2"
export HF_HOME=/SAN/intelsys/rclearn/cache

cd /SAN/intelsys/rclearn/when-do-reading-comprehension-models-learn
"""

    job_file = headers + '\n\n'
    for job_id, cmd in enumerate(commands):
        job_file += "test $SGE_TASK_ID -eq {} && sleep 10 && {}".format(job_id+1, cmd)
        job_file += '\n'

    job_file += '\ndate\n'

    # Save the job file
    with open(f'jobs/train/albert-xlarge-v2-mrqa.job', 'w') as f:
        f.write(job_file)
