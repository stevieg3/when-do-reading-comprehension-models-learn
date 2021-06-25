"""
Script for generating .job files. Adapted version of script written by Max Bartolo.
"""

MODEL_NAME_OR_PATH = 'albert-xlarge-v2'
PER_DEVICE_BATCH_SIZE = 2
SAVE_STEPS_SCHEDULE = "1 2 4 8 16 32 64 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096 4224 4352 4480 4608 4736 4864 4992 5120 5248 5376 5504 5632 5760 5888 6016 6144 6272 6400 6528 6656 6784 6912 7040 7168 7296 7424 7552 7680 7808 7936 8064 8192"
MAX_STEPS = 8200
FP16 = '--fp16 True'
LOGGING_STEPS = 896
SEEDS = [27, 28, 29]
OUTPUT_DIR_ROOT = '/cluster/project7/max_harderqs/projects/sgeorge/when-do-reading-comprehension-models-learn/models/'

experiments = [
    {
        'run_name': 'squadv2-wu=100-lr=3e5-bs=32-msl=384-seed={}',
        'warmup_steps': 100,
        'learning_rate': 3e-5,
        'accumulation_steps': 16,
        'max_seq_length': 384,
    },
    {
        'run_name': 'squadv2-wu=0-lr=3e5-bs=32-msl=384-seed={}',
        'warmup_steps': 0,
        'learning_rate': 3e-5,
        'accumulation_steps': 16,
        'max_seq_length': 384,
    },
    {
        'run_name': 'squadv2-wu=100-lr=3e5-bs=48-msl=512-seed={}',
        'warmup_steps': 100,
        'learning_rate': 3e-5,
        'accumulation_steps': 24,
        'max_seq_length': 512,
    },
    {
        'run_name': 'squadv2-wu=0-lr=3e5-bs=48-msl=512-seed={}',
        'warmup_steps': 0,
        'learning_rate': 3e-5,
        'accumulation_steps': 24,
        'max_seq_length': 512,
    }
]


# Generate the commands
commands = []

for seed in SEEDS:
    this_commands = [
        f"python src/models/run_qa.py "
        f"--model_name_or_path {MODEL_NAME_OR_PATH} "
        f"--dataset_name squad_v2 "
        f"--version_2_with_negative "
        f"--do_train "
        f"--do_eval "
        f"--per_device_train_batch_size {PER_DEVICE_BATCH_SIZE} "
        f"--gradient_accumulation_steps {e['accumulation_steps']} "
        f"--learning_rate {e['learning_rate']} "
        f"--max_seq_length {e['max_seq_length']} "
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
        f"--warmup_steps {e['warmup_steps']} "
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
#$ -l tmem=32G
#$ -t 1-{len(commands)}
#$ -l h_rt=72:00:00
#$ -o /cluster/project7/max_harderqs/projects/sgeorge/when-do-reading-comprehension-models-learn/array.out
#$ -e /cluster/project7/max_harderqs/projects/sgeorge/when-do-reading-comprehension-models-learn/array.err
#$ -l gpu=true

hostname
date

# Activate conda environment
conda activate rclearn

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
export WANDB_PROJECT="albert-xlarge-v2"

cd /cluster/project7/max_harderqs/projects/sgeorge/when-do-reading-comprehension-models-learn
"""

    job_file = headers + '\n\n'
    for job_id, cmd in enumerate(commands):
        job_file += "test $SGE_TASK_ID -eq {} && sleep 10 && {}".format(job_id+1, cmd)
        job_file += '\n'

    job_file += '\ndate\n'

    # Save the job file
    with open(f'jobs/train/albert-xlarge-v2-squadv2.job', 'w') as f:
        f.write(job_file)
