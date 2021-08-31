import os
import logging
import argparse

import pandas as pd
import numpy as np

from src.models.kmeans import \
    MODELS, \
    DEV_DATA_SIZES, \
    NUM_CHECKPOINTS, \
    get_kmeans_clusters


def create_filepath_dict_full(model_filepath: str) -> dict:
    """
    Filepath dict for full predictions incl. model confidences

    :param model_filepath: Filepath containing model checkpoints
    :return: Dict with checkpoint numbers as keys and corresponding path as values
    """
    checkpoint_str = list(
        os.walk(model_filepath)
    )[0][1]

    checkpoint_nbr = [int(x.split('-')[-1]) for x in checkpoint_str]

    checkpoint_fp = [f'{model_filepath}/{x}/eval_predictions_full.json' for x in checkpoint_str]

    prediction_filepath_dict = dict(zip(checkpoint_nbr, checkpoint_fp))

    return prediction_filepath_dict


def generate_predictions_df_full(model_filepath: str, seed: int):
    """
    Generate DataFrame of predictions by checkpoint and seed from raw JSON output files (incl. model confidences)
    """
    logging.info('Loading predictions data')
    prediction_filepath_dict = create_filepath_dict_full(model_filepath)

    predictions_df = pd.DataFrame()

    for checkpoint, fp in prediction_filepath_dict.items():
        eval_predictions_df = pd.read_json(fp, orient='index').reset_index()
        eval_predictions_df.rename(columns={'index': 'id'}, inplace=True)
        eval_predictions_df['checkpoint'] = checkpoint
        eval_predictions_df['seed'] = seed

        predictions_df = predictions_df.append(eval_predictions_df)

    return predictions_df


def load_per_example_confidences_df(seed: int) -> pd.DataFrame:
    all_predictions_df = pd.DataFrame()

    for dataset in MODELS:
        predictions_df = generate_predictions_df_full(
            model_filepath=f'predictions/full/albert-xlarge-v2-squadv1-adversarialall-wu=100-lr=3e5-bs=32-msl=384'
                           f'-seed={seed}-{dataset}',
            seed=seed
        )
        predictions_df['dataset'] = dataset

        all_predictions_df = all_predictions_df.append(predictions_df, ignore_index=True)

    # Replace -1 confidences with 0:
    all_predictions_df['model_conf'] = np.where(
        all_predictions_df['model_conf'] == -1,
        0,
        all_predictions_df['model_conf']
    )

    assert all_predictions_df.shape[0] == NUM_CHECKPOINTS * np.sum(list(DEV_DATA_SIZES.values()))
    logging.info(all_predictions_df.shape)

    return all_predictions_df


def main(seed, km_seed, n_clusters, savepath, max_iter):
    per_example_confidences_df = load_per_example_confidences_df(seed=seed)
    id_km_labels_df = get_kmeans_clusters(
        per_example_metrics_df=per_example_confidences_df,
        n_clusters=n_clusters,
        model_seed=seed,
        km_seed=km_seed,
        max_iter=max_iter,
        value='model_conf'
    )
    id_km_labels_df.to_csv(savepath, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--km_seed", type=int)
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--savepath", type=str)
    parser.add_argument("--max_iter", type=int)

    args = parser.parse_args()

    main(args.seed, args.km_seed, args.n_clusters, args.savepath, args.max_iter)
