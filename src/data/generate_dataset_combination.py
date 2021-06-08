import json
from urllib import request
import argparse

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/{}-v1.1.json"

SQUAD_NAME = "squadv1.1"
ADVERSARIAL_NAME = "adversarialqa_d{}"

ADVERSARIAL_MODEL_TO_FP = {
    'bidaf': 'data/external/aqa_v1.0/1_dbidaf/{}.json',
    'bert': 'data/external/aqa_v1.0/2_dbert/{}.json',
    'roberta': 'data/external/aqa_v1.0/3_droberta/{}.json'
}


def generate_squad_adversarial_combination(num_squad, num_adversarial, adversarial_model, split, version=''):
    """
    Create a weighted dataset combining SQuAD v1.1 with Adversarial QA dataset. Generated files can be passed as a
    training argument.

    :param num_squad: Number of copies of SQuAD v1.1 to include
    :param num_adversarial: Number of copies of adversarial QA dataset to include
    :param adversarial_model: One of ['bidaf', 'bert', 'roberta']
    :param split: One of ['train', 'dev']
    :param version: Version identifier
    :return: None. Writes new JSON file for constructed dataset
    """

    try:
        assert split in ['train', 'dev']
    except AssertionError:
        raise ValueError("Invalid `split`. Must be one of ['train', 'dev']")

    try:
        assert adversarial_model in ['bidaf', 'bert', 'roberta']
    except AssertionError:
        raise ValueError("Invalid `adversarial_model`. Must be one of ['bidaf', 'bert', 'roberta']")

    # Load SQuAD data
    with request.urlopen(SQUAD_URL.format(split)) as url:
        squad_data = json.loads(url.read().decode())

    # Add dataset name to example dictionary
    for data in squad_data['data']:
        for para in data['paragraphs']:
            para['dataset'] = SQUAD_NAME

    # Load Adversarial QA data
    adversarial_fp = ADVERSARIAL_MODEL_TO_FP[adversarial_model]
    adversarial_fp = adversarial_fp.format(split)

    with open(adversarial_fp) as f:
        aqa_data = json.load(f)

    # Add dataset name to example dictionary
    for data in aqa_data['data']:
        for para in data['paragraphs']:
            para['dataset'] = ADVERSARIAL_NAME.format(adversarial_model)

    # Create new data dictionary
    new_dataset_dict = {}
    new_data = []
    for _ in range(num_squad):
        new_data += squad_data['data']
    for _ in range(num_adversarial):
        new_data += aqa_data['data']

    new_dataset_dict['version'] = version
    new_dataset_dict['data'] = new_data

    # Write to JSON
    fp = f"data/processed/squad1_d{adversarial_model}_{int(num_squad)}_{int(num_adversarial)}_weighted_{split}.json"
    with open(fp, "w") as output:
        json.dump(new_dataset_dict, output)


def main(num_squad, num_adversarial, adversarial_model, version):
    generate_squad_adversarial_combination(
        num_squad, num_adversarial, adversarial_model, split='train', version=version
    )
    generate_squad_adversarial_combination(
        num_squad, num_adversarial, adversarial_model, split='dev', version=version
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_squad", type=int)
    parser.add_argument("--num_adversarial", type=int)
    parser.add_argument("--adversarial_model", type=str)
    parser.add_argument("--version", type=str, default='')

    args = parser.parse_args()

    main(
        num_squad=args.num_squad,
        num_adversarial=args.num_adversarial,
        adversarial_model=args.adversarial_model,
        version=args.version
    )
