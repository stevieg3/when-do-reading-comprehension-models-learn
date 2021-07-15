import argparse
import os
import itertools
import logging

import pandas as pd
from tqdm import tqdm

from src.analysis.utils import \
    load_squadv2_dev_as_df, \
    squad2_evaluation, \
    load_squadv1_dev_as_df

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def create_filepath_dict(model_filepath: str) -> dict:
    """
    :param model_filepath: Filepath containing model checkpoints
    :return: Dict with checkpoint numbers as keys and corresponding path as values
    """

    checkpoint_str = list(
        os.walk(model_filepath)
    )[0][1]

    checkpoint_nbr = [int(x.split('-')[-1]) for x in checkpoint_str]

    checkpoint_fp = [f'{model_filepath}/{x}/eval_predictions.json' for x in checkpoint_str]

    prediction_filepath_dict = dict(zip(checkpoint_nbr, checkpoint_fp))

    return prediction_filepath_dict


def generate_predictions_df(model_filepath: str, seed: int):
    """
    Generate DataFrame of predictions by checkpoint and seed from raw JSON output files
    """
    logging.info('Loading predictions data')
    prediction_filepath_dict = create_filepath_dict(model_filepath)

    predictions_df = pd.DataFrame()

    for checkpoint, fp in prediction_filepath_dict.items():
        eval_predictions_df = pd.read_json(fp, orient='index').reset_index()
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

    for checkpoint, label in tqdm(
            list(
                itertools.product(
                    full_df['checkpoint'].unique(),
                    full_df[category_label].unique()
                )
            )
    ):

        full_df_subset = full_df.copy()[
            (full_df[category_label] == label) &
            (full_df['checkpoint'] == checkpoint)
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
        metrics['checkpoint'] = checkpoint

        full_metrics.append(metrics)

    full_metrics_df = pd.DataFrame(full_metrics)

    # Merge overall metrics
    full_metrics_df = full_metrics_df.merge(overall_metrics_df, on=['checkpoint'])

    logging.info(full_metrics_df.shape)
    logging.info(full_metrics_df.head())

    if save:
        logging.info('Saving to CSV')
        full_metrics_df.to_csv(savepath, index=False)

    return full_metrics_df


def main(squad_version: int, model_filepath: str, seed: int):
    # ============================= #
    # Load model predictions
    # ============================= #
    predictions_df = generate_predictions_df(model_filepath, seed)

    # ============================= #
    # Load labels
    # ============================= #
    if squad_version == 1:
        labels_df = load_squadv1_dev_as_df()
    elif squad_version == 2:
        labels_df = load_squadv2_dev_as_df()
    else:
        raise ValueError("squad_version must be 1 or 2")

    # ============================= #
    # Add example categories
    # ============================= #

    # Load categories
    logging.info("Loading categories")
    if squad_version == 1:
        categories = pd.read_csv(
            'data/processed/squadv1_dev_categories.csv',
            usecols=[
                'id', 'w6h_label', 'context_length_bin', 'question_length_bin', 'answer_mode_length_bin', 'answer_type'
            ]
        )
        logging.info(categories.shape)
        logging.info(categories.head())
    elif squad_version == 2:
        categories = pd.read_csv(
            'data/processed/squadv2_dev_categories.csv',
            usecols=[
                'id', 'w6h_label', 'context_length_bin', 'question_length_bin', 'answer_mode_length_bin', 'unanswerable', 'answer_type'
            ]
        )
        logging.info(categories.shape)
        logging.info(categories.head())

    # Merge predictions and labels
    combined = predictions_df.merge(labels_df, on='id', how='inner')
    assert combined.shape[0] == predictions_df.shape[0]

    # Merge category columns
    combined = combined.merge(categories, on='id', how='inner')
    assert combined.shape[0] == predictions_df.shape[0]

    logging.info(combined.head())

    # ============================= #
    # Get overall metrics
    # ============================= #
    logging.info("Computing overall metrics")
    overall_f1_perf = []

    for checkpoint in tqdm(combined['checkpoint'].unique()):

        subset = combined.copy()[(combined['checkpoint'] == checkpoint)]

        eval_output = squad2_evaluation(
            id_list=list(subset['id']),
            prediction_text_list=list(subset['prediction_text']),
            answers_list=list(subset['answers'])
        )

        overall_f1_perf.append(
            {
                'overall_f1': eval_output['f1'],
                'checkpoint': checkpoint,
                'overall_exact': eval_output['exact']
            }
        )

    overall_f1_perf_df = pd.DataFrame(overall_f1_perf)
    logging.info(overall_f1_perf_df.shape)
    logging.info(overall_f1_perf_df.head())

    # Merge onto combined DataFrame
    combined = combined.merge(overall_f1_perf_df, on=['checkpoint'])

    # ============================= #
    # Generate plot data
    # ============================= #
    logging.info('Generating plot data')

    model_name = model_filepath.split('/')[-1]

    if squad_version == 2:
        # Answerable vs unanswerable
        logging.info('Answerable vs unanswerable')
        generate_metrics_by_category_df(
            full_df=combined,
            overall_metrics_df=overall_f1_perf_df,
            category_label='unanswerable',
            save=True,
            savepath=f'data/processed/metrics_by_unanswerable-{model_name}.csv'
        )

    # Answer type
    logging.info('Answer type')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='answer_type',
        save=True,
        savepath=f'data/processed/metrics_by_answer_type-{model_name}.csv'
    )

    # WWWWWWH
    logging.info('WWWWWWH')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='w6h_label',
        save=True,
        savepath=f'data/processed/metrics_by_w6h-{model_name}.csv'
    )
    
    # Context length
    logging.info('Context length')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='context_length_bin',
        save=True,
        savepath=f'data/processed/metrics_by_context_length_bin-{model_name}.csv'
    )

    # Question length
    logging.info('Question length')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='question_length_bin',
        save=True,
        savepath=f'data/processed/metrics_by_question_length_bin-{model_name}.csv'
    )

    # Answer length
    logging.info('Answer length')
    generate_metrics_by_category_df(
        full_df=combined,
        overall_metrics_df=overall_f1_perf_df,
        category_label='answer_mode_length_bin',
        save=True,
        savepath=f'data/processed/metrics_by_answer_mode_length_bin-{model_name}.csv'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_version", type=int)
    parser.add_argument("--model_filepath", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    main(args.squad_version, args.model_filepath, args.seed)
