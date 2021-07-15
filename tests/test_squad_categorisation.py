import unittest
import ast

import pandas as pd
import numpy as np
from parameterized import parameterized

from src.analysis.squad_categorisation import \
    add_w6h_category, \
    add_context_length_category, \
    add_question_length_category, \
    add_answer_length_category, \
    get_majority


class TestSquadCategorisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        test_cases = pd.read_csv('tests/data/squad_categories_test_cases.csv')
        cls.w6h_test_df = test_cases.copy()[test_cases['type'] == 'w6h']
        cls.context_test_df = test_cases.copy()[test_cases['type'] == 'context']
        cls.question_test_df = test_cases.copy()[test_cases['type'] == 'question']
        cls.answer_test_df = test_cases.copy()[test_cases['type'] == 'answer']
        cls.answer_test_df['text'] = cls.answer_test_df['text'].apply(lambda x: ast.literal_eval(x))

    def test_add_w6h_category(cls):
        add_w6h_category(df=cls.w6h_test_df, question_column='text')
        np.testing.assert_array_equal(
            cls.w6h_test_df['expected_category'],
            cls.w6h_test_df['w6h_label']
        )

    def test_add_context_length_category(cls):
        add_context_length_category(df=cls.context_test_df, context_column='text')
        np.testing.assert_array_equal(
            cls.context_test_df['expected_category'],
            cls.context_test_df['context_length_bin']
        )

    def test_add_question_length_category(cls):
        add_question_length_category(df=cls.question_test_df, question_column='text')
        np.testing.assert_array_equal(
            cls.question_test_df['expected_category'],
            cls.question_test_df['question_length_bin']
        )

    def test_add_answer_length_category(cls):
        add_answer_length_category(df=cls.answer_test_df, answer_column='text')
        np.testing.assert_array_equal(
            cls.answer_test_df['expected_category'],
            cls.answer_test_df['answer_mode_length_bin']
        )

    @parameterized.expand([
        (
            [{'text': 'France'}, {'text': 'Italy'}, {'text': 'Spain'}, {'text': 'Spain'}],
            'Spain'
        ),
        (
            [{'text': 'France'}, {'text': 'Italy'}, {'text': 'France'}, {'text': 'Spain'}, {'text': 'Spain'}],
            'France'
        ),
        (
            [{'text': 'Italy'}, {'text': 'France'}, {'text': 'France'}, {'text': 'Spain'}, {'text': 'Spain'}],
            'France'
        ),
        (
            [{'text': 'Italy'}, {'text': 'France'}, {'text': 'Spain'}],
            'Italy'
        ),
        (
            [{'text': 'Italy'}, {'text': 'Italy'}, {'text': 'France'}, {'text': 'Spain'}, {'text': 'Spain'}],
            'Italy'
        ),
    ])
    def test_get_majority(cls, input, exp_output):
        maj_ans = get_majority(input)[0]['text']
        np.testing.assert_equal(maj_ans, exp_output)
