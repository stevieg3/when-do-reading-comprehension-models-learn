#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=3600
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -l gpu=true
#$ -l gpu_p100=yes

# Activate conda environment
conda activate rclearn

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/Github/when-do-reading-comprehension-models-learn

test $SGE_TASK_ID -eq 1 && sleep 10 && python src/models/practice_finetune/run_qa.py --model_name_or_path bert-base-uncased --dataset_name squad --do_train --do_eval --per_device_train_batch_size 2 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --output_dir /tmp/debug_squad/ --max_train_samples 1024 --max_eval_samples 1024
