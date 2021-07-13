import logging
import argparse

import pandas as pd
import numpy as np
from scipy import stats

from src.analysis.utils import \
    load_squadv1_dev_as_df, \
    load_squadv2_dev_as_df

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

NUM_NEGATIVE_EXAMPLES_SQUAD_2_DEV = 5945  # https://arxiv.org/pdf/1806.03822.pdf

W6H_LABELS = ['what', 'how', 'who', 'when', 'which', 'where', 'why']

CONTEXT_LENGTH_BINS = {
    'bins': list(range(0, 501, 100)) + [999999],  # bins are right inclusive e.g. 500 in '400-500'
    'labels': ['0-100', '100-200', '200-300', '300-400', '400-500', '>500']
}

QUESTION_LENGTH_BINS = {
    'bins': list(range(0, 26, 5)) + [999999],
    'labels': ['0-5', '5-10', '10-15', '15-20', '20-25', '>25']
}


def add_w6h_category(df: pd.DataFrame, question_column: str) -> None:
    """
    Classify questions based on first word
    :param df: DataFrame containing questions in a single column
    :param question_column: Name of question column
    :return: None. Modifies in-place.
    """
    df['w6h_label'] = df[question_column].apply(lambda x: x.lower().split()[0])  # First word as category

    df['w6h_label'] = np.where(
        df['w6h_label'].isin(W6H_LABELS),
        df['w6h_label'],
        'other'
    )


def add_context_length_category(df: pd.DataFrame, context_column: str) -> None:
    """
    Create categories based on length of context in words.

    :param df: DataFrame containing context in a single column
    :param context_column: Name of context column
    :return: None. Modifies in-place.
    """
    df['context_length'] = df[context_column].apply(lambda x: len(x.split()))  # Split into words

    df['context_length_bin'] = pd.cut(
        df['context_length'],
        bins=CONTEXT_LENGTH_BINS['bins'],
        labels=CONTEXT_LENGTH_BINS['labels']
    ).astype(str)

    df.drop(columns='context_length', inplace=True)


def add_question_length_category(df: pd.DataFrame, question_column: str) -> None:
    """
    Create categories based on length of question in words.

    :param df: DataFrame containing questions in a single column
    :param question_column: Name of question column
    :return: None. Modifies in-place.
    """
    df['question_length'] = df[question_column].apply(lambda x: len(x.split()))

    df['question_length_bin'] = pd.cut(
        df['question_length'],
        bins=QUESTION_LENGTH_BINS['bins'],
        labels=QUESTION_LENGTH_BINS['labels']
    ).astype(str)

    df.drop(columns='question_length', inplace=True)


def add_answer_length_category(df: pd.DataFrame, answer_column: str) -> None:
    """
    Create categories based on length of answer in words. Since there can be multiple answers for a given question we
    take the modal answer length amongst all possibilities.

    :param df: DataFrame containing answers in a single column
    :param answer_column: Name of answer column
    :return: None. Modifies in-place.
    """
    df['text_lengths'] = df[answer_column].apply(lambda x: [len(y.split()) for y in x])
    df['answer_mode_length'] = (
        df['text_lengths']
        .apply(lambda x: stats.mode(x)[0])
        .apply(lambda x: x[0] if len(x) > 0 else 0)  # Select first modal value
    )

    df['answer_mode_length_bin'] = np.where(
        df['answer_mode_length'] > 9,
        '>9',
        df['answer_mode_length'].astype(str)
    )

    df.drop(columns=['text_lengths', 'answer_mode_length'], inplace=True)


def main(squad_version: int) -> None:
    if squad_version == 1:
        squad_df = load_squadv1_dev_as_df()
    elif squad_version == 2:
        squad_df = load_squadv2_dev_as_df()
    else:
        raise ValueError("squad_version must be 1 or 2")

    # Unpack answers column into answer_start and text columns
    squad_df = pd.concat(
        (squad_df, squad_df['answers'].apply(pd.Series)),
        axis=1
    )

    if squad_version == 2:
        # Create unanswerable flag
        squad_df['unanswerable'] = np.where(
            squad_df['answer_start'].apply(len) == 0,
            1,
            0
        )
        assert squad_df['unanswerable'].sum() == NUM_NEGATIVE_EXAMPLES_SQUAD_2_DEV

    # Add category columns
    add_w6h_category(squad_df, 'question')
    add_context_length_category(squad_df, 'context')
    add_answer_length_category(squad_df, 'text')
    add_question_length_category(squad_df, 'question')

    logging.info(squad_df.shape)
    logging.info(squad_df.head())
    squad_df.to_csv(f'data/processed/squadv{squad_version}_dev_categories.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_version", type=int)

    args = parser.parse_args()

    main(squad_version=args.squad_version)
