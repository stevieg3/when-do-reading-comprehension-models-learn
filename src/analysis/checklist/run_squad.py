"""
Code for running CheckList test suite on SQuAD predictions

Adapted version of script written by Max Bartolo
"""

import os
import argparse
import checklist
from checklist.test_suite import TestSuite


"""
Example Usage:
python run_squad.py '/private/home/maxbartolo/_code/checklist/release_data/squad/predictions/bert' > results_bert_checklist.txt
python run_squad.py '/private/home/maxbartolo/_code/checklist_max/predictions_bert_round1_flat.txt' > results_bert_round1.txt
python run_squad.py '/private/home/maxbartolo/_code/checklist_max/predictions_roberta_round1_flat.txt' > results_roberta_round1.txt
python run_squad.py '/private/home/maxbartolo/_code/checklist_max/predictions_bert_round2_flat.txt' > results_bert_round2.txt
python run_squad.py '/private/home/maxbartolo/_code/checklist_max/predictions_roberta_round2_1_flat.txt' > results_roberta_round2.txt
"""

parser = argparse.ArgumentParser()
parser.add_argument("prediction_file", help="Path to the predictions", type=str)
args = parser.parse_args()

suite_path = '/home/sgeorge/Github/checklist/release_data/squad/squad_suite.pkl'
suite = TestSuite.from_file(suite_path)

# Running tests with precomputed bert predictions:
suite.run_from_file(args.prediction_file, overwrite=True, file_format='pred_only')
# suite.visual_summary_table()
suite.summary(n_per_testcase=3)
