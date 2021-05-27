# Practice fine-tune

Files for practicing model fine-tuning and gaining familiarity with UCL cluster.

`run_qa.py`, `trainer_qa.py`, `utils_qa.py` originally from Huggingface examples.

```bash
python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $HOME/Github/when-do-reading-comprehension-models-learn/outputs \
  --overwrite_output_dir \
  --max_train_samples 512 \
  --max_eval_samples 1024
```