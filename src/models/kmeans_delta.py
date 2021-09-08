import argparse

import pandas as pd

from src.models.kmeans import \
    load_per_example_metrics_df, \
    get_kmeans_clusters


def compute_f1_deltas(df: pd.DataFrame) -> None:
    df.sort_values(['id', 'checkpoint'], inplace=True)
    df['f1_delta'] = df['f1'] - df.groupby('id')['f1'].shift(1)
    df.dropna(axis=0, subset=['f1_delta'], inplace=True)  # Starting nulls due to shift


def main(seed, km_seed, n_clusters, savepath, max_iter):
    per_example_metrics_df = load_per_example_metrics_df(seed=seed)

    compute_f1_deltas(per_example_metrics_df)

    id_km_labels_df = get_kmeans_clusters(
        per_example_metrics_df=per_example_metrics_df,
        n_clusters=n_clusters,
        model_seed=seed,
        km_seed=km_seed,
        max_iter=max_iter,
        value='f1_delta'
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
