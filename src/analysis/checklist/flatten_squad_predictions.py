"""
Raw predictions made by huggingface need to be flattened before passing to CheckList.

Adapted version of script written by Max Bartolo
"""

import argparse
import json
import os


"""
Example Usage:
python flatten_squad_predictions.py '/private/home/maxbartolo/_code/checklist_max/predictions_bert_round2.json'
"""

parser = argparse.ArgumentParser()
parser.add_argument("prediction_file", help="Path to the predictions", type=str)
args = parser.parse_args()

file_dir = os.path.dirname(args.prediction_file)
filename, ext = os.path.splitext(args.prediction_file)

with open(args.prediction_file, 'r') as f:
    preds_dict = json.load(f)

flat_preds = list(preds_dict.values())

# Save output
with open(os.path.join(file_dir, filename+'_flat.txt'), 'w') as f:
    f.writelines(s + '\n' for s in flat_preds)

