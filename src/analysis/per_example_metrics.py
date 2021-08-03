import argparse
import logging

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.analysis.generate_plot_data import generate_predictions_df
from src.analysis.utils import load_squadv1_dev_as_df
from src.analysis.utils import squad1_evaluation


def load_adversarial_dev_as_df(dmodel):
    logging.info(f"Loading {dmodel} dev data as DataFrame")
    adversarial_val = load_dataset('adversarial_qa', dmodel, split='validation')

    adversarial_val_df = pd.DataFrame(adversarial_val)

    logging.info(adversarial_val_df.shape)
    logging.info(adversarial_val_df.head())

    return adversarial_val_df


def main(model_filepath: str, seed: int, dataset: str):
    predictions_df = generate_predictions_df(model_filepath, seed)
    if dataset in ['dbert', 'dbidaf', 'droberta']:
        labels_df = load_adversarial_dev_as_df(dataset)
    elif dataset == 'squad':
        labels_df = load_squadv1_dev_as_df()

    combined = predictions_df.merge(labels_df, on='id', how='inner')
    assert predictions_df.shape[0] == combined.shape[0]

    # Compute metrics per example
    metric_list = []
    for _, row in tqdm(combined.iterrows(), total=combined.shape[0]):
        metrics = squad1_evaluation(
            row[['id']],
            row[['prediction_text']],
            row[['answers']]
        )
        metrics['id'] = row['id']
        metric_list.append(metrics)

    metrics_df = pd.DataFrame(metric_list)
    combined = combined.merge(metrics_df, on='id')
    combined = combined[['id', 'checkpoint', 'seed', 'exact_match', 'f1']]
    combined['dataset'] = dataset

    combined.to_csv(
        f'data/processed/per_example_metrics-squadv1-adversarialall-dataset={dataset}-seed={seed}',
        index=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_filepath", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    main(args.model_filepath, args.seed, args.dataset)
