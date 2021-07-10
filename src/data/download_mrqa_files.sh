# Links from https://github.com/huggingface/datasets/blob/master/datasets/mrqa/mrqa.py#L50

# Train sub-datasets

wget -P data/external/mrqa/train "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz"
wget -P data/external/mrqa/train "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz"
wget -P data/external/mrqa/train "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz"
wget -P data/external/mrqa/train "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz"
wget -P data/external/mrqa/train "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz"
wget -P data/external/mrqa/train "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz"

# Validation sub-datasets
wget -P data/external/mrqa/dev "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz"
wget -P data/external/mrqa/dev "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz"
wget -P data/external/mrqa/dev "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz"
wget -P data/external/mrqa/dev "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz"
wget -P data/external/mrqa/dev "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz"
wget -P data/external/mrqa/dev "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz"

# Combine and convert into SQuAD format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/train/HotpotQA.jsonl.gz, data/external/mrqa/train/NaturalQuestionsShort.jsonl.gz, data/external/mrqa/train/NewsQA.jsonl.gz, data/external/mrqa/train/SearchQA.jsonl.gz, data/external/mrqa/train/SQuAD.jsonl.gz, data/external/mrqa/train/TriviaQA-web.jsonl.gz' --save_name mrqa_train_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/HotpotQA.jsonl.gz, data/external/mrqa/dev/NaturalQuestionsShort.jsonl.gz, data/external/mrqa/dev/NewsQA.jsonl.gz, data/external/mrqa/dev/SearchQA.jsonl.gz, data/external/mrqa/dev/SQuAD.jsonl.gz, data/external/mrqa/dev/TriviaQA-web.jsonl.gz' --save_name mrqa_dev_squad_format
