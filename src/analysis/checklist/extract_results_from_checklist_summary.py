"""
Code for extracting failure rates for CheckList tests from full summary reeport.

Adapted version of script written by Max Bartolo
"""

import argparse
import re
import os


"""
Example Usage:
python extract_results_from_checklist_summary.py '/private/home/maxbartolo/_code/checklist_max/results_bert_checklist.txt'
python extract_results_from_checklist_summary.py '/private/home/maxbartolo/_code/checklist_max/results_bert_round2.txt'
python extract_results_from_checklist_summary.py '/private/home/maxbartolo/_code/checklist_max/results_roberta_round2.txt'
"""

parser = argparse.ArgumentParser()
parser.add_argument("prediction_file", help="Path to the predictions", type=str)
args = parser.parse_args()

file_dir = os.path.dirname(args.prediction_file)
filename, ext = os.path.splitext(args.prediction_file)

with open(args.prediction_file, 'r') as f:
    summary = f.read()

results = re.findall(r"Fails \(rate\):    \d+ \(([\d|\.]+\%)\)", summary)
print(len(results))
print(results)

# Save output
with open(os.path.join(file_dir, filename+'_summary.txt'), 'w') as f:
    f.writelines(s + '\n' for s in results)
