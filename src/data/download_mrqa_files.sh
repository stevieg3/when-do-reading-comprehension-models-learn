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

# Test (out-domain validation) sub-datasets
wget -P data/external/mrqa/test "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz"
wget -P data/external/mrqa/test "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz"
wget -P data/external/mrqa/test "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz"
wget -P data/external/mrqa/test "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz"
wget -P data/external/mrqa/test "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz"
#wget -P data/external/mrqa/test "http://participants-area.bioasq.org/MRQA2019/"  # BioASQ.jsonl.gz
cd data/external/mrqa/test && { curl -JLO "http://participants-area.bioasq.org/MRQA2019/" ; cd -; }

# Combine and convert into SQuAD format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/train/HotpotQA.jsonl.gz, data/external/mrqa/train/NaturalQuestionsShort.jsonl.gz, data/external/mrqa/train/NewsQA.jsonl.gz, data/external/mrqa/train/SearchQA.jsonl.gz, data/external/mrqa/train/SQuAD.jsonl.gz, data/external/mrqa/train/TriviaQA-web.jsonl.gz' --save_name mrqa_train_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/HotpotQA.jsonl.gz, data/external/mrqa/dev/NaturalQuestionsShort.jsonl.gz, data/external/mrqa/dev/NewsQA.jsonl.gz, data/external/mrqa/dev/SearchQA.jsonl.gz, data/external/mrqa/dev/SQuAD.jsonl.gz, data/external/mrqa/dev/TriviaQA-web.jsonl.gz' --save_name mrqa_dev_squad_format
# Save evaluation files separately so they can be run on separate jobs
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/HotpotQA.jsonl.gz' --save_name hotpotqa_dev_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/NaturalQuestionsShort.jsonl.gz' --save_name naturalquestions_dev_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/NewsQA.jsonl.gz' --save_name newsqa_dev_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/SearchQA.jsonl.gz' --save_name searchqa_dev_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/SQuAD.jsonl.gz' --save_name squad_dev_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/dev/TriviaQA-web.jsonl.gz' --save_name triviaqa_dev_squad_format

python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/test/BioASQ.jsonl.gz' --save_name bioasq_test_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/test/DROP.jsonl.gz' --save_name drop_test_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/test/DuoRC.ParaphraseRC.jsonl.gz' --save_name duorc_test_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/test/RACE.jsonl.gz' --save_name race_test_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/test/RelationExtraction.jsonl.gz' --save_name relationextraction_test_squad_format
python src/data/convert_mrqa_to_squad.py --file_paths 'data/external/mrqa/test/TextbookQA.jsonl.gz' --save_name textbookqa_test_squad_format
