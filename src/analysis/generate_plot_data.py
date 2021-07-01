import os
import itertools
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import \
    load_squadv2_dev_as_df, \
    squad2_evaluation

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

PREDICTION_PATH = 'predictions/albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384-seed={}/'
SEEDS = [27, 28, 29]
BATCH_SIZE = 32


def _create_filepath_dict():
    """
    Created nested dictionary containing all eval_predictions.json file paths
    """
    prediction_filepath_dict = {}

    for seed in SEEDS:

        checkpoint_str = list(
            os.walk(PREDICTION_PATH.format(seed))
        )[0][1]

        checkpoint_nbr = [int(x.split('-')[-1]) for x in checkpoint_str]

        checkpoint_fp = [PREDICTION_PATH.format(seed) + x + '/eval_predictions.json' for x in checkpoint_str]

        prediction_filepath_dict[seed] = dict(zip(checkpoint_nbr, checkpoint_fp))

    return prediction_filepath_dict


def generate_predictions_df():
    """
    Generate DataFrame of predictions by checkpoint and seed from raw JSON output files
    """
    logging.info('Loading predictions data')
    prediction_filepath_dict = _create_filepath_dict()

    predictions_df = pd.DataFrame()

    for seed in tqdm(SEEDS):
        for checkpoint, fp in prediction_filepath_dict[seed].items():
            eval_predictions_df = pd.read_json(fp, orient='index')
            eval_predictions_df.reset_index(inplace=True)
            eval_predictions_df.rename(columns={'index': 'id', 0: "prediction_text"}, inplace=True)
            eval_predictions_df['checkpoint'] = checkpoint
            eval_predictions_df['seed'] = seed

            predictions_df = predictions_df.append(eval_predictions_df)

    return predictions_df


def generate_metrics_by_category_df(
        full_df: pd.DataFrame,
        overall_metrics_df: pd.DataFrame,
        category_label: str,
        save: bool = False,
        savepath: str = None
) -> pd.DataFrame:
    """
    Compute metrics by categorical group.

    :param full_df: DataFrame containing 'num_examples', '{category_label}', 'seed', 'id', 'prediction_text', 'answers'
    columns
    :param overall_metrics_df: DataFrame containing overall metrics (over all examples) by checkpoint and seed
    :param category_label: Name of column containing example categories
    :param save: Save output DataFrame as CSV
    :param savepath: Filepath to save to incl. extension
    :return: DataFrame
    """
    full_df = full_df.copy()
    full_metrics = []

    for seed, num_examples, label in tqdm(
            list(
                itertools.product(
                    SEEDS,
                    full_df['num_examples'].unique(),
                    full_df[category_label].unique()
                )
            )
    ):

        full_df_subset = full_df.copy()[
            (full_df[category_label] == label) &
            (full_df['num_examples'] == num_examples) &
            (full_df['seed'] == seed)
            ]

        id_list = list(full_df_subset['id'])
        prediction_text_list = list(full_df_subset['prediction_text'])
        answers_list = list(full_df_subset['answers'])

        metrics = squad2_evaluation(
            id_list=id_list,
            prediction_text_list=prediction_text_list,
            answers_list=answers_list
        )

        metrics[category_label] = label
        metrics['num_examples'] = num_examples
        metrics['seed'] = seed

        full_metrics.append(metrics)

    full_metrics_df = pd.DataFrame(full_metrics)
    full_metrics_df['checkpoint'] = full_metrics_df['num_examples'] / BATCH_SIZE

    # Merge overall metrics
    full_metrics_df = full_metrics_df.merge(overall_metrics_df, on=['seed', 'checkpoint'])

    logging.info(full_metrics_df.shape)
    logging.info(full_metrics_df.head())

    if save:
        logging.info('Saving to CSV')
        full_metrics_df.to_csv(savepath, index=False)

    return full_metrics_df


def main():
    # ============================= #
    # Load model predictions
    # ============================= #
    predictions_df = generate_predictions_df()

    # ============================= #
    # Load labels
    # ============================= #
    squad_v2_val_df = load_squadv2_dev_as_df()

    # ============================= #
    # Add example categories
    # ============================= #

    # Load categories
    logging.info("Loading categories")
    squad2_categories = pd.read_csv('data/processed/squad2_dev_simple_categories.csv')
    logging.info(squad2_categories.shape)
    logging.info(squad2_categories.head())

    # Merge predictions and labels
    combined = predictions_df.merge(squad_v2_val_df, on='id', how='inner')
    assert combined.shape[0] == predictions_df.shape[0]

    # Merge category columns
    combined = combined.merge(squad2_categories, on='id', how='inner')
    assert combined.shape[0] == predictions_df.shape[0]

    combined['num_examples'] = combined['checkpoint'] * BATCH_SIZE

    logging.info(combined.head())

    # ============================= #
    # Get overall metrics
    # ============================= #
    logging.info("Computing overall metrics")
    overall_f1_perf = []

    for seed in SEEDS:
        for checkpoint in tqdm(combined['checkpoint'].unique()):

            subset = combined.copy()[(combined['checkpoint'] == checkpoint) & (combined['seed'] == seed)]

            eval_output = squad2_evaluation(
                id_list=list(subset['id']),
                prediction_text_list=list(subset['prediction_text']),
                answers_list=list(subset['answers'])
            )

            overall_f1_perf.append(
                {
                    'seed': seed,
                    'overall_f1': eval_output['f1'],
                    'checkpoint': checkpoint,
                    'overall_exact': eval_output['exact']
                }
            )

    overall_f1_perf_df = pd.DataFrame(overall_f1_perf)
    logging.info(overall_f1_perf_df.shape)
    logging.info(overall_f1_perf_df.head())

    # Merge onto combined DataFrame
    combined = combined.merge(overall_f1_perf_df, on=['checkpoint', 'seed'])

    # ============================= #
    # Generate plot data
    # ============================= #
    logging.info('Generating plot data')

    # Answerable vs unanswerable
    logging.info('Answerable vs unanswerable')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='unanswerable',
        save=True,
        savepath='data/processed/metrics_by_unanswerable-albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384.csv'
    )

    # WWWWWWH
    logging.info('WWWWWWH')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='w8h_label',
        save=True,
        savepath='data/processed/metrics_by_w6h-albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384.csv'
    )
    
    # Context length
    logging.info('Context length')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='context_length_bin',
        save=True,
        savepath='data/processed/metrics_by_context_length_bin-albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384.csv'
    )

    # Question length
    logging.info('Question length')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='question_length_bin',
        save=True,
        savepath='data/processed/metrics_by_question_length_bin-albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384.csv'
    )

    # Answer length
    logging.info('Answer length')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='answer_mode_length_bin',
        save=True,
        savepath='data/processed/metrics_by_answer_mode_length_bin-albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384.csv'
    )


if __name__ == '__main__':
    main()
