import unittest
import json

from parameterized import parameterized

from src.analysis.generate_plot_data import \
    create_filepath_dict, \
    generate_predictions_df
from src.analysis.utils import \
    load_squadv2_dev_as_df, \
    load_squadv1_dev_as_df, \
    squad2_evaluation, \
    squad1_evaluation


class TestGeneratePlotData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @parameterized.expand([
        ('tests/data/albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384-seed=27', [3, 156, 1164, 3084, 5388, 5772])
    ])
    def test_create_filepath_dict(cls, model_filepath, checkpoints):
        """
        Check that predictions in live and retro runs contain the same number of players.
        """
        expected_dict = dict(
            zip(
                checkpoints,
                [model_filepath + f'/checkpoint-{chkpt}/eval_predictions.json' for chkpt in checkpoints]
            )
        )
        expected_dict = dict(sorted(expected_dict.items()))

        generated_dict = create_filepath_dict(model_filepath=model_filepath)
        generated_dict = dict(sorted(generated_dict.items()))

        cls.assertDictEqual(expected_dict, generated_dict)

    @parameterized.expand([
        (3, ),
        (156, ),
        (1164, ),
        (3084, ),
        (5388, ),
        (5772,)
    ])
    def test_squad2_evaluation(cls, checkpoint):
        predictions_df = generate_predictions_df(
            model_filepath='tests/data/albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384-seed=27',
            seed=27
        )
        labels = load_squadv2_dev_as_df()

        combined = predictions_df.merge(labels, on='id')
        combined = combined[combined['checkpoint'] == checkpoint]

        computed_metrics = squad2_evaluation(
            id_list=combined['id'],
            prediction_text_list=combined['prediction_text'],
            answers_list=combined['answers']
        )

        with open(
                f'tests/data/albert-xlarge-v2-squadv2-wu=100-lr=3e5-bs=32-msl=384-seed=27/'
                f'checkpoint-{checkpoint}/eval_results.json'
        ) as json_file:
            expected_metrics = json.load(json_file)

        expected_metrics = {k.split('eval_')[-1]: v for k, v in expected_metrics.items() if k != 'eval_samples'}

        cls.assertDictEqual(computed_metrics, expected_metrics)

    @parameterized.expand([
        (3, ),
        (684, ),
        (5388, )
    ])
    def test_squad1_evaluation(cls, checkpoint):
        predictions_df = generate_predictions_df(
            model_filepath='tests/data/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=29',
            seed=29
        )
        labels = load_squadv1_dev_as_df()

        combined = predictions_df.merge(labels, on='id')
        combined = combined[combined['checkpoint'] == checkpoint]

        computed_metrics = squad1_evaluation(
            id_list=combined['id'],
            prediction_text_list=combined['prediction_text'],
            answers_list=combined['answers']
        )

        with open(
                f'tests/data/albert-xlarge-v2-squadv1-wu=100-lr=3e5-bs=32-msl=384-seed=29/'
                f'checkpoint-{checkpoint}/eval_results.json'
        ) as json_file:
            expected_metrics = json.load(json_file)

        expected_metrics = {k.split('eval_')[-1]: v for k, v in expected_metrics.items() if k != 'eval_samples'}

        cls.assertDictEqual(computed_metrics, expected_metrics)
