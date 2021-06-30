import logging

import pandas as pd
from datasets import load_dataset, load_metric

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

SQUAD_V2_METRIC = load_metric("squad_v2")


def load_squadv2_dev_as_df():
    logging.info("Loading SQuAD v2 data as DataFrame")
    squad_v2_val = load_dataset('squad_v2', split='validation')

    squad_v2_val_df = pd.DataFrame(squad_v2_val)

    logging.info(squad_v2_val_df.shape)
    logging.info(squad_v2_val_df.head())

    return squad_v2_val_df


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
