import argparse
import logging

import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

SEEDS = [27, 28, 29]

MODELS = ['dbert', 'dbidaf', 'droberta', 'squad']

DEV_DATA_SIZES = {
    'dbert': 1000,
    'dbidaf': 1000,
    'droberta': 1000,
    'squad': 10570
}

NUM_CHECKPOINTS = 120


def load_per_example_metrics_df(seed: int) -> pd.DataFrame:
    logging.info('Loading per example metrics')
    example_metric_df = pd.DataFrame()

    for model in MODELS:
        df = pd.read_csv(f'data/processed/per_example_metrics-squadv1-adversarialall-dataset={model}-seed={seed}.csv')
        example_metric_df = example_metric_df.append(df)

    assert example_metric_df.shape[0] == NUM_CHECKPOINTS * np.sum(list(DEV_DATA_SIZES.values()))
    logging.info(example_metric_df.shape)

    return example_metric_df


def _prepare_data(per_example_metrics_df: pd.DataFrame, value: str = 'f1') -> (np.array, dict):
    """
    Prepare input array for k-means. Input is of dim (n_ts, sz, d) where n_ts=number of time series; sz=length of
    time series; d=dimensionality of time series
    :param per_example_metrics_df:
    :return:
    """
    logging.info('Preparing input for k-means')
    per_example_metrics_df = per_example_metrics_df.copy()
    per_example_metrics_df.sort_values(['id', 'checkpoint'], inplace=True)

    n_ts = per_example_metrics_df['id'].nunique()
    assert n_ts == np.sum(list(DEV_DATA_SIZES.values()))
    sz = NUM_CHECKPOINTS
    d = 1

    X = np.zeros((n_ts, sz, d))

    # Store mapping for index position to corresponding ID
    idx_to_id_dict = {}

    for idx, _id in tqdm(
            enumerate(per_example_metrics_df['id'].unique()),
            total=per_example_metrics_df['id'].nunique()
    ):

        idx_to_id_dict[idx] = _id

        X[idx, :, :] = per_example_metrics_df[per_example_metrics_df['id'] == _id][value].values.reshape(-1, 1)

    logging.info(X.shape)

    return X, idx_to_id_dict


def get_kmeans_clusters(
        per_example_metrics_df: pd.DataFrame,
        n_clusters: int,
        model_seed: int,
        km_seed: int,
        value: str = 'f1',
        max_iter: int = 300
) -> pd.DataFrame:

    X, idx_to_id_dict = _prepare_data(per_example_metrics_df, value=value)

    # Fit K-means
    logging.info('Fitting k-means')
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=max_iter,
        random_state=km_seed,
        verbose=0,
        n_jobs=-1
    )
    labels = km.fit_predict(X)
    logging.info('Finished k-means')
    logging.info('Processing labels')

    id_km_labels = []
    for idx, _id in idx_to_id_dict.items():
        id_km_labels.append((_id, labels[idx]))

    id_km_labels_df = pd.DataFrame(id_km_labels, columns=['id', 'KM_label'])
    assert id_km_labels_df.shape[0] == np.sum(list(DEV_DATA_SIZES.values()))

    id_km_labels_df['km_seed'] = km_seed
    id_km_labels_df['model_seed'] = model_seed

    logging.info(id_km_labels_df.shape)
    logging.info(id_km_labels_df.head())

    return id_km_labels_df


def main(seed, km_seed, n_clusters, savepath, max_iter):
    per_example_metrics_df = load_per_example_metrics_df(seed=seed)
    id_km_labels_df = get_kmeans_clusters(
        per_example_metrics_df=per_example_metrics_df,
        n_clusters=n_clusters,
        model_seed=seed,
        km_seed=km_seed,
        max_iter=max_iter
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
