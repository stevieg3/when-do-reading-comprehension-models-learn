"""
General utilities

Answer-type utilities provided by Max Bartolo
"""
import logging
import itertools

import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
import datefinder
from nltk.tokenize import wordpunct_tokenize

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

SQUAD_V1_METRIC = load_metric("squad")

SQUAD_V2_METRIC = load_metric("squad_v2")


def load_squadv2_dev_as_df():
    logging.info("Loading SQuAD v2 data as DataFrame")
    squad_v2_val = load_dataset('squad_v2', split='validation')

    squad_v2_val_df = pd.DataFrame(squad_v2_val)

    logging.info(squad_v2_val_df.shape)
    logging.info(squad_v2_val_df.head())

    return squad_v2_val_df


def load_squadv1_dev_as_df():
    logging.info("Loading SQuAD v1 data as DataFrame")
    squad_v1_val = load_dataset('squad', split='validation')

    squad_v1_val_df = pd.DataFrame(squad_v1_val)

    logging.info(squad_v1_val_df.shape)
    logging.info(squad_v1_val_df.head())

    return squad_v1_val_df


def squad2_evaluation(id_list, prediction_text_list, answers_list):
    predictions = [
        {
            'prediction_text': prediction_text,
            'id': _id,
            'no_answer_probability': 0.0  # Same as https://github.com/huggingface/transformers/blob/master/examples/
            # pytorch/question-answering/run_qa.py#L532
        } for
        _id, prediction_text in zip(id_list, prediction_text_list)
    ]

    references = [
        {
            'id': _id,
            'answers': answers
        } for
        _id, answers in zip(id_list, answers_list)
    ]

    metrics = SQUAD_V2_METRIC.compute(predictions=predictions, references=references)

    return metrics


def squad1_evaluation(id_list, prediction_text_list, answers_list):
    predictions = [
        {
            'prediction_text': prediction_text,
            'id': _id
        } for
        _id, prediction_text in zip(id_list, prediction_text_list)
    ]

    references = [
        {
            'id': _id,
            'answers': answers
        } for
        _id, answers in zip(id_list, answers_list)
    ]

    metrics = SQUAD_V1_METRIC.compute(predictions=predictions, references=references)

    return metrics


def is_numeric(text):
    return any(char.isdigit() for char in str(text))

def is_date(text):
    # a generator will be returned by the datefinder module.
    matches = list(datefinder.find_dates(text))

    if len(matches) > 0:
        return True
    return False

def classify_NER_text(NER_text):
    NER_tag = NER_text[1]
    if  NER_tag == 'PERSON':
        return 'Person'
    elif NER_tag == 'LOCATION':
        return 'Location'
    elif NER_tag == 'ORGANIZATION':
        return 'Organisation'
    else:
        return 'Other Entity'

def most_common(lst):
    return max(set(lst), key=lst.count)

def verify_single_verb_group(POS_list):
    # Convert any verb-related tags to VB
    POS_list = ['VB' if 'VB' in x else x for x in POS_list]
    # Eliminate repeating elements (ie verb groups)
    POS_list = [k for k, g in itertools.groupby(POS_list)]
    # Check if the reduced list contains just one verb group
    if POS_list.count('VB') == 1:
        return True
    return False

def get_char_length(sent):
    return len(sent)

def get_word_length(sent):
    return len(sent.strip())

def classify_text(text, POS_text, NER_text):
    text_class = ''
    if is_numeric(text):
        if is_date(text):
            return 'Date'
        else:
            return 'Other Numeric'
    else:
        sent = wordpunct_tokenize(text)
        # POS_text = POS_tagger.tag(sent)
        POS_tags = np.array(POS_text)[:,1]

        # Detect Proper Noun Phrases
        if any('NNP' in POS for POS in POS_tags) and not any(tag in POS for tag in ['VB'] for POS in POS_tags):
            # NER classify proper nouns
            # NER_list = NER_tagger.tag([word for word, POS in POS_text if 'NNP' in POS])
            NER_list = [(word, NER) for (word, NER), POS in zip(NER_text, POS_tags) if 'NNP' in POS]
            NER_list = [classify_NER_text(NER_text) for NER_text in NER_list]
            return most_common(NER_list)

        # Detect Common Noun Phrases
        elif any('NN' in POS for POS in POS_tags) and not any(tag in POS for tag in ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] for POS in POS_tags):
            return 'Common Noun Phrase'

        # Detect Clauses (ie one verb group and one noun http://grammar.yourdictionary.com/grammar-rules-and-tips/Grammar-Clause.html)
        # https://www.merriam-webster.com/dictionary/clause
        elif any('NN' in POS for POS in POS_tags) and any(POS in ['VBD', 'VBG', 'VBP', 'VBZ'] for POS in POS_tags) and verify_single_verb_group(POS_tags):
            return 'Clause'

        # Detect Verb Phrase
        elif any('VB' in POS for POS in POS_tags):
            return 'Verb Phrase'

        # Detect Adjective Phrase
        elif any('JJ' in POS for POS in POS_tags):
            return 'Adjective Phrase'

        # Other
        else:
            return 'Other'
