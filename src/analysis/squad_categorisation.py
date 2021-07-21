import logging
import argparse

import pandas as pd
import numpy as np
from scipy import stats
from nltk.tokenize import wordpunct_tokenize
from nltk.tag.stanford import \
    StanfordNERTagger, \
    StanfordPOSTagger

from src.analysis.utils import \
    load_squadv1_dev_as_df, \
    load_squadv2_dev_as_df, \
    load_squadv1_train_as_df, \
    load_squadv2_train_as_df, \
    classify_text

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

NUM_NEGATIVE_EXAMPLES_SQUAD_2_DEV = 5945  # https://arxiv.org/pdf/1806.03822.pdf

NUM_NEGATIVE_EXAMPLES_SQUAD_2_TRAIN = 43498  # https://arxiv.org/pdf/1806.03822.pdf

W6H_LABELS = ['what', 'how', 'who', 'when', 'which', 'where', 'why']

CONTEXT_LENGTH_BINS = {
    'bins': list(range(0, 201, 50)) + [999999],  # bins are right inclusive e.g. 200 in '150-200'
    'labels': ['0-50', '50-100', '100-150', '150-200', '>200']
}

QUESTION_LENGTH_BINS = {
    'bins': list(range(0, 16, 5)) + [999999],
    'labels': ['0-5', '5-10', '10-15', '>15']
}

NER_TAGGER = StanfordNERTagger(
    'data/external/stanford-ner-4.2.0/stanford-ner-2020-11-17/classifiers/english.all.3class.distsim.crf.ser.gz',
    'data/external/stanford-ner-4.2.0/stanford-ner-2020-11-17/stanford-ner.jar',
    encoding='utf-8'
)

POS_TAGGER = StanfordPOSTagger(
    'data/external/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/models/english-left3words-distsim.tagger',
    'data/external/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar',
    encoding='utf-8'
)


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


def add_answer_length_category(df: pd.DataFrame, answer_column: str) -> pd.DataFrame:
    """
    Create categories based on length of answer in words. Since there can be multiple answers for a given question we
    take the majority vote answer.

    :param df: DataFrame containing answers in a single column
    :param answer_column: Name of answer column
    :return: DataFrame with ID and answer_length_bin
    """
    df = df.copy()
    df['text_dict_format'] = df[answer_column].apply(lambda x: [{'text': text} for text in x])
    df['majority_vote_answer'] = df['text_dict_format'].apply(
        lambda x: get_majority(x)[0]['text']  # get_majority returns a list with a single dict item
    )
    df['answer_length'] = df['majority_vote_answer'].apply(lambda x: len(x.split()))

    df['answer_length_bin'] = np.where(
        df['answer_length'] > 9,
        '>9',
        df['answer_length'].astype(str)
    )

    return df[['id', 'answer_length_bin']]


def get_majority(answer_list: list) -> list:
    """
    Author: Max Bartolo
    Extract the majority vote answer or first answer in case of no majority for the dev data for consistency
    """
    dict_answer_counts = {}
    for i, ans in enumerate(answer_list):
        if ans['text'] in dict_answer_counts:
            dict_answer_counts[ans['text']]['count'] += 1
        else:
            dict_answer_counts[ans['text']] = {
                'id': i,  # return the first occurring answer or first answer seen
                'count': 1
            }

    # Extract counts and indices
    count_indices = [(ans['count'], ans['id']) for ans in dict_answer_counts.values()]
    # Sort, first by index ascending, then by count descending
    count_indices = sorted(sorted(count_indices, key=lambda x: x[1]), key=lambda x: x[0], reverse=True)

    # Check that we have as many counts as expects
    assert len(answer_list) == sum(x[0] for x in count_indices)

    # Return the most common answer by index in the list
    return [answer_list[count_indices[0][1]]]


def add_answer_type_category(df: pd.DataFrame, answer_column: str) -> pd.DataFrame:
    """
    :param df: DataFrame containing answerable questions only
    :param answer_column: Name of answer column
    :return: DataFrame with question IDs and answer type
    """
    logging.info("Getting answer types")
    df = df.copy()
    df['text_dict_format'] = df[answer_column].apply(lambda x: [{'text': text} for text in x])
    df['majority_vote_answer'] = df['text_dict_format'].apply(
        lambda x: get_majority(x)[0]['text']  # get_majority returns a list with a single dict item
    )

    answers_tok = df.majority_vote_answer.apply(wordpunct_tokenize).tolist()
    df['ANALYSIS_answer_POS_tags'] = POS_TAGGER.tag_sents(answers_tok)
    df['ANALYSIS_answer_NER_tags'] = NER_TAGGER.tag_sents(answers_tok)

    df['answer_type'] = ""
    for idx in df.index:
        try:
            df.loc[idx, 'answer_type'] = classify_text(
                df.loc[idx]['majority_vote_answer'],
                df.loc[idx]['ANALYSIS_answer_POS_tags'],
                df.loc[idx]['ANALYSIS_answer_NER_tags']
            )

        except:
            logging.info("Classified as 'Other Numeric':")
            logging.info(idx)
            logging.info(df.loc[idx]['majority_vote_answer'])
            df.loc[idx, 'answer_type'] = 'Other Numeric'

    logging.info(df['answer_type'].value_counts() / df.shape[0])

    return df[['id', 'answer_type']]


def main(squad_version: int, split: str) -> None:

    if split == 'train':
        if squad_version == 1:
            squad_df = load_squadv1_train_as_df()
        elif squad_version == 2:
            squad_df = load_squadv2_train_as_df()
        else:
            raise ValueError("squad_version must be 1 or 2")

    elif split == 'validation':
        if squad_version == 1:
            squad_df = load_squadv1_dev_as_df()
        elif squad_version == 2:
            squad_df = load_squadv2_dev_as_df()
        else:
            raise ValueError("squad_version must be 1 or 2")

    else:
        raise ValueError("Invalid split value. Must be one of 'train' or 'validation'")

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
        if split == 'train':
            assert squad_df['unanswerable'].sum() == NUM_NEGATIVE_EXAMPLES_SQUAD_2_TRAIN
        elif split == 'validation':
            assert squad_df['unanswerable'].sum() == NUM_NEGATIVE_EXAMPLES_SQUAD_2_DEV

    # Add category columns
    if squad_version == 2:
        ans_type_df = add_answer_type_category(
            df=squad_df.copy()[squad_df['unanswerable'] == 0],
            answer_column='text'
        )
    else:
        ans_type_df = add_answer_type_category(
            df=squad_df,
            answer_column='text'
        )
    shape_before = squad_df.shape
    squad_df = squad_df.merge(ans_type_df, on='id', how='left')
    shape_after = squad_df.shape
    assert shape_before[0] == shape_after[0]
    squad_df['answer_type'].fillna("UNANS", inplace=True)

    if squad_version == 2:
        ans_type_df = add_answer_length_category(
            df=squad_df.copy()[squad_df['unanswerable'] == 0],
            answer_column='text'
        )
    else:
        ans_type_df = add_answer_length_category(
            df=squad_df,
            answer_column='text'
        )
    shape_before = squad_df.shape
    squad_df = squad_df.merge(ans_type_df, on='id', how='left')
    shape_after = squad_df.shape
    assert shape_before[0] == shape_after[0]
    squad_df['answer_type'].fillna("0", inplace=True)

    add_w6h_category(squad_df, 'question')

    add_context_length_category(squad_df, 'context')

    add_question_length_category(squad_df, 'question')

    logging.info(squad_df.shape)
    logging.info(squad_df.head())

    if split == 'train':
        savepath = f'data/processed/squadv{squad_version}_train_categories.csv'
    elif split == 'validation':
        savepath = f'data/processed/squadv{squad_version}_dev_categories.csv'

    squad_df.to_csv(savepath, index=False)
    logging.info('Saved to CSV')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_version", type=int)
    parser.add_argument("--split", type=str)

    args = parser.parse_args()

    main(squad_version=args.squad_version, split=args.split)
