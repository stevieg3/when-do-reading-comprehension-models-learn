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
