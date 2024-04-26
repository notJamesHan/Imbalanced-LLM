import argparse

import logging
import os
import random

import datetime
from collections import Counter
from pathlib import Path
import math

from scipy.io.arff import loadarff

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    set_seed
)

from helper.note_generator import NoteGenerator
from helper.note_template import NoteTemplate
from helper.external_datasets_variables import *

logger = logging.getLogger(__name__)

cat_idx_dict = {
    "car": [0,1,2,3,4,5],
    "diabetes": [],
    "heart": [1,2,6,8,10],
    "income": [1,2,3,4,5,6,7,11],
    "creditg": [0,2,3,4,5,6,8,9,11,13,14,16,18,19],
    "blood": [],
    "bank": [1,2,3,4,6,7,8,10,15],
    "jungle": [],
    "calhousing": [],
}
bin_num = 10

def main():
    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Configuration
    data_dir = Path("/content/drive/MyDrive/Colab Notebooks/TabLLM/datasets")
    data_dir = data_dir / args.dataset
    temp_output = 'dataset-generation-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("/content/drive/MyDrive/Colab Notebooks/TabLLM/datasets_serialized") / temp_output
    if not args.debug:
        os.mkdir(output_dir)
    logger.info(f"Generate dataset {args.dataset}.")

    if not args.list and args.permuted:
        raise ValueError("Permuted note is not supported.")
    dataset_name = args.dataset + \
                   ('_list' if args.list else '') + \
                   ('_permuted' if args.permuted else '') + \
                   ('_values' if args.values else '') + \
                   ('_shuffled' if args.shuffled else '') + \
                   ('_importance' if args.feature_importance else '')
    dataset = load_train_validation_test(args.dataset, data_dir)

    # if args.debug:
    #     dataset['train'] = dataset['train'].sample(min(10, len(dataset['train'])))
    #     dataset['validation'] = dataset['validation'].sample(min(5, len(dataset['validation'])))
    #     dataset['test'] = dataset['test'].sample(min(5, len(dataset['test'])))
    # # logger.info(f"  Cohort size: {len(dataset['train'])}, {len(dataset['validation'])}, {len(dataset['test'])}")

    # if args.feature_importance:
    #     # Simply combine all examples and create a list of features as covariates of the linear model.
    #     dataset['train'] = pd.concat([dataset[k] for k in dataset.keys()])
    #     dataset['validation'] = dataset['validation'].sample(0)
    #     dataset['test'] = dataset['train'].sample(0)
    #     # For each of them generate all feature values
    #     output_linear_classifier_features(dataset['train'], output_dir, args.dataset)

    # template, template_config = None, None
    template = eval('template_' + dataset_name)
    print(template)
    template_config = eval('template_config_' + dataset_name)
    note_generator = NoteTemplate(template, **template_config)

    # External datasets are now split later
    dataset = pd.concat(list(dataset.values()), ignore_index=True)

    # # Shuffled: shuffle each feature column separately
    # if args.shuffled:
    #     # np.random.seed(42)
    #     def derange(n):
    #         orig = np.arange(n)
    #         derangement = orig
    #         while np.any(orig == derangement):
    #             derangement = np.random.permutation(orig)
    #         return derangement
        
    #     def shuffle_dataset(dataset):
    #         cat_idx = cat_idx_dict[args.dataset]
    #         derangement_dict = {}
    #         for column_idx, c in enumerate(dataset.columns):
    #             if column_idx in cat_idx and c != 'label':
    #                 derangement_dict[c] = {}
    #                 value_set = list(set(dataset[c].values))
    #                 derangement = derange(len(value_set))
    #                 derangement_dict[c] = {value: value_set[derangement[i]] for i, value in enumerate(value_set)}
    #                 dataset[c] = [derangement_dict[c][value] for value in dataset[c]]
    #             if column_idx not in cat_idx and c!= 'label':
    #                 value_list = dataset[c].values
    #                 ret_value_list = []
    #                 num_values = len(value_list)
    #                 sorted_value_list = sorted(list(value_list))
    #                 derangement = derange(bin_num)

    #                 bin_idx_intervals = []
    #                 bin_idx_endpoints = []
    #                 factor = num_values / bin_num
    #                 for bin_idx in range(bin_num):
    #                     lower_idx, upper_idx = math.floor(bin_idx * factor), math.floor((bin_idx + 1) * factor)
    #                     bin_idx_intervals.append([lower_idx, upper_idx])
    #                     bin_idx_endpoints.append([sorted_value_list[lower_idx], sorted_value_list[upper_idx-1]])

    #                 for value in value_list:
    #                     for bin_idx, (lower_value, upper_value) in enumerate(bin_idx_endpoints):
    #                         if value >= lower_value and value <= upper_value:
    #                             mapped_bin_lower_idx, mapped_bin_upper_idx = bin_idx_intervals[derangement[bin_idx]]
    #                             sampled_bin_values = sorted_value_list[mapped_bin_lower_idx : mapped_bin_upper_idx]
    #                             ret_value_list.append(random.choice(sampled_bin_values))
    #                             break
    #                 dataset[c] = ret_value_list
    #         return dataset
        
    #     dataset = shuffle_dataset(dataset)

    notes = [NoteGenerator.clean_note(note_generator.substitute(r)) for _, r in dataset.iterrows()]
    old_size_notes = 1
    start = 0  # 25000
    end = len(notes)
    notes = notes[start:end]
    dataset = dataset.iloc[start:end]
    print(f"Only consider dataset range between {start} and {end} (total: {old_size_notes})")

    for i in range(0, min(10, len(notes))):
        print('----')
        print(notes[i])
    dataset = Dataset.from_dict({'note': notes, 'label': dataset['label'].to_list()})

    if not args.debug:
        logger.info(f"Store generated datasets to {output_dir}/{dataset_name}")
        logger.info(f"\tn={len(dataset)}, feats={dataset.num_columns}, labels={dict(Counter(dataset['label']))}")
        dataset.save_to_disk(str(output_dir / dataset_name))


def load_train_validation_test(dataset_name, data_dir):
    # Load existing data, put into train, validation, test and create label
    def train_validation_test_split(data):
        # Don't want to shuffle bc done later with right seed to make it identical with external evaluation
        data_train, data_test = train_test_split(data, test_size=0.20, shuffle=False)
        data_valid, data_test = train_test_split(data_test, test_size=0.50, shuffle=False)
        return data_train, data_valid, data_test

    def byte_to_string_columns(data):
        for col, dtype in data.dtypes.items():
            if dtype == object:  # Only process byte object columns.
                data[col] = data[col].apply(lambda x: x.decode("utf-8"))
        return data

    if dataset_name == "creditg":
        dataset = pd.DataFrame(loadarff(data_dir / 'dataset_31_credit-g.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'good'
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)

    elif dataset_name == "blood":
        columns = {'V1': 'recency', 'V2': 'frequency', 'V3': 'monetary', 'V4': 'time', 'Class': 'label'}
        dataset = pd.DataFrame(loadarff(data_dir / 'php0iVrYT.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset['label'] = dataset['label'] == '2'
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)

    elif dataset_name == "bank":
        columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                   'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        columns = {'V' + str(i + 1): v for i, v in enumerate(columns)}
        dataset = pd.DataFrame(loadarff(data_dir / 'phpkIxskf.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns=columns, inplace=True)
        dataset.rename(columns={'Class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == '2'
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)

    elif dataset_name == "jungle":
        dataset = pd.DataFrame(loadarff(data_dir / 'jungle_chess_2pcs_raw_endgame_complete.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'class': 'label'}, inplace=True)
        dataset['label'] = dataset['label'] == 'w'  # Does white win?
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)

    elif dataset_name == "calhousing":
        dataset = pd.DataFrame(loadarff(data_dir / 'houses.arff')[0])
        dataset = byte_to_string_columns(dataset)
        dataset.rename(columns={'median_house_value': 'label'}, inplace=True)
        # Make binary task by labelling upper half as true
        median_price = dataset['label'].median()
        dataset['label'] = dataset['label'] > median_price
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)

    elif dataset_name == "income":
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                   'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
                   'native_country', 'label']

        def strip_string_columns(df):
            df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

        dataset_train = pd.read_csv(data_dir / 'adult.data', names=columns, na_values=['?', ' ?'])
        dataset_train = dataset_train.drop(columns=['fnlwgt', 'education_num'])
        original_size = len(dataset_train)
        strip_string_columns(dataset_train)
        # Multiply all dollar columns by two to adjust for inflation
        # dataset_train[['capital_gain', 'capital_loss']] = (1.79 * dataset_train[['capital_gain', 'capital_loss']]).astype(int)
        dataset_train['label'] = dataset_train['label'] == '>50K'

        dataset_test = pd.read_csv(data_dir / 'adult.test', names=columns, na_values=['?', ' ?'])
        dataset_test = dataset_test.drop(columns=['fnlwgt', 'education_num'])
        strip_string_columns(dataset_test)
        # Note label string in test set contains full stop
        # dataset_test[['capital_gain', 'capital_loss']] = (1.79 * dataset_test[['capital_gain', 'capital_loss']]).astype(int)
        dataset_test['label'] = dataset_test['label'] == '>50K.'

        dataset_train, dataset_valid = train_test_split(dataset_train, test_size=0.20, random_state=1)
        dataset = dataset_train
        assert len(dataset_train) + len(dataset_valid) == original_size

    elif dataset_name == "car":
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety_dict', 'label']
        dataset = pd.read_csv(data_dir / 'car.data', names=columns)
        original_size = len(dataset)
        label_dict = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
        dataset['label'] = dataset['label'].replace(label_dict)
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    elif dataset_name == "voting":
        columns = ['label', 'handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution',
                   'physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban',
                   'aid_to_nicaraguan_contras', 'mx_missile', 'immigration', 'synfuels_corporation_cutback',
                   'education_spending', 'superfund_right_to_sue', 'crime', 'duty_free_exports',
                   'export_administration_act_south_africa']
        dataset = pd.read_csv(data_dir / 'house-votes-84.data', names=columns, na_values=['?'])
        original_size = len(dataset)
        dataset['label'] = np.where(dataset['label'] == 'democrat', 1, 0)
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    elif dataset_name == "wine":
        columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                   'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
        dataset = pd.read_csv(data_dir / 'winequality-red.csv', names=columns, skiprows=[0])
        original_size = len(dataset)
        # Adopt grouping from: https://www.kaggle.com/code/vishalyo990/prediction-of-quality-of-wine
        bins = (2, 6.5, 8)
        dataset['quality'] = pd.cut(dataset['quality'], bins=bins, labels=[0, 1]).astype(int)  # bad, good
        dataset = dataset.rename(columns={'quality': 'label'})
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    elif dataset_name == "titanic":
        # Only use training set since no labels for test set
        dataset = pd.read_csv(data_dir / 'train.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'Survived': 'label'})
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    elif dataset_name == "heart":
        dataset = pd.read_csv(data_dir / 'heart.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'HeartDisease': 'label'})
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    elif dataset_name == "diabetes":
        dataset = pd.read_csv(data_dir / 'diabetes.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'Outcome': 'label'})
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    # Added by James
    elif dataset_name == "stroke":
        dataset = pd.read_csv(data_dir / 'stroke.csv')
        original_size = len(dataset)
        dataset = dataset.rename(columns={'stroke': 'label'})
        dataset_train, dataset_valid, dataset_test = train_validation_test_split(dataset)
        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == original_size

    else:
        raise ValueError("Dataset not found")

    # For final experiments, ensure correct numbers of features for each dataset
    dataset_specs = {
        'income': 13,
        'car': 7,
        'heart': 12,
        'diabetes': 9,
        'creditg': 21,
        'blood': 5,
        'bank': 17,
        'jungle': 7,
        'wine': 12,
        'calhousing': 9,
        'stroke': 12,
    }
    assert dataset_name in dataset_specs.keys() and len(dataset.columns) == dataset_specs[dataset_name]

    dataset = {"train": dataset_train, "validation": dataset_valid, "test": dataset_test}
    return dataset



# def output_linear_classifier_features(examples, output_dir, dataset):
#     def remove_constants(data):
#         return data[[c for c in data if data[c].nunique() > 1]]
#     # Normalize numerical variables analogously to LR, copied from fitted scaler in evaluate_external_dataset (seed 42).
#     scalings = {
#         'income': {'age': [38.66194047, 13.70079038], 'capital_gain': [1092.03493461, 7514.89341966],
#                    'capital_loss': [87.05228675, 401.7001878], 'hours_per_week': [40.45123231, 12.43397048]},
#         'car': {},
#         'heart': {'Age': [53.63760218, 9.38893213], 'RestingBP': [132.09264305, 18.09209337],
#                   'Cholesterol': [201.70844687, 107.50566557], 'FastingBS': [0.23160763, 0.42185962],
#                   'MaxHR': [136.59945504, 25.12828773], 'Oldpeak': [0.92711172, 1.06128969]},
#         'diabetes': {'Pregnancies': [3.68403909, 3.28025968], 'Glucose': [120.41042345, 32.63939221],
#                      'BloodPressure': [68.75081433, 19.83518715], 'SkinThickness': [20.22638436, 15.68020872],
#                      'Insulin': [79.43485342, 114.8289827], 'BMI': [31.77654723, 8.02507088],
#                      'DiabetesPedigreeFunction': [0.47113192, 0.33090205], 'Age': [32.90879479, 11.66936554]}
#     }
#     scaling = scalings[dataset]

#     def normalize_examples(data):
#         for c in scaling.keys():
#             data[c] = (data[c] - scaling[c][0]) / scaling[c][1]
#         return data

#     examples_dummies = remove_constants(pd.get_dummies(examples, dummy_na=True))

#     if dataset == 'income':
#         assert len(examples_dummies.columns) == 107

#     # Also write out weighted version for linear explanation model
#     examples_dummies = normalize_examples(examples_dummies)
#     examples_dummies.to_pickle(output_dir / (dataset + '_lr_examples.p'))

#     # Might be necessary for income
#     # examples_dummies = remove_constants(pd.get_dummies(examples, dummy_na=True))

#     # Sample examples for debugging
#     # index_samples = np.random.choice(examples.index, min(200, len(examples)))
#     # examples = examples.loc[index_samples]
#     # examples_dummies = examples_dummies.loc[index_samples]

def parse_args():
    parser = argparse.ArgumentParser(description="Create note dataset from cohort.")
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dataset",
        type=str
    )
    parser.add_argument(
        "--list",
        action="store_true",
    )
    parser.add_argument(
        "--tabletotext",
        action="store_true",
    )
    parser.add_argument(
        "--t0serialization",
        action="store_true",
    )
    parser.add_argument(
        "--permuted",
        action="store_true",
    )
    parser.add_argument(
        "--values",
        action="store_true",
    )
    parser.add_argument(
        "--shuffled",
        action="store_true",
    )
    parser.add_argument(
        "--feature_importance",
        action="store_true",
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()