import datetime
import os
import json
import pickle

import numpy as np
import yaml
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    DatasetDict,
    Dataset,
)
from promptsource.templates import DatasetTemplates
# import pkg_resources
# from promptsource import templates
# import csv
# from typing import Dict, List, Optional, Tuple
# import re
# import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

templates_for_custom_tasks = {
    "income": "50000_dollars",
    "car": "rate_decision",
    "heart": "heart_disease",
    "diabetes": "diabetes",
    "creditg": "creditg",
    "bank": "bank",
    "blood": "blood",
    "jungle": "jungle",
    "calhousing": "calhousing",
}


def is_custom_task(cfg):
    task = cfg.dataset.split("_")[0].lower()
    if task in templates_for_custom_tasks.keys():
        return True


def get_dataset_reader(config):
    dataset_class = CustomCategoricalReader
    return dataset_class(config)


DATASETS_OFFLINE = "./datasets_serialized"
MAX_EXAMPLES_PER_DATASET = 500_000
TASK_BLACKLIST = [
    # Tasks which often tokenize to > 1024 tokens currently
    "hotpot_qa_distractor_Generate_Explanations",
    "hotpot_qa_fullwiki_Generate_Explanations",
    "hotpot_qa_distractor_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer_and_Explanations",
    "hotpot_qa_fullwiki_Generate_Answer",
    "hotpot_qa_distractor_Generate_Answer",
    "hotpot_qa_distractor_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_2",
    "hotpot_qa_fullwiki_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Title_1",
    "hotpot_qa_distractor_Generate_Question",
    "hotpot_qa_fullwiki_Generate_Question",
    "tab_fact_tab_fact_tab_fact_3",
    "tab_fact_tab_fact_tab_fact_2",
    "tab_fact_tab_fact_tab_fact_1",
    "tab_fact_tab_fact_tab_fact_7",
    "tab_fact_tab_fact_tab_fact_4",
    "tab_fact_tab_fact_tab_fact_5",
    "tab_fact_tab_fact_tab_fact_6",
    "wiki_hop_masked_Choose_Best_Object_Candidate",
    "wiki_hop_masked_Indirect_Question_about_Birthplace_Citizenship_Place_of_Death",
    "narrativeqa_Template_05",
    "ecthr_cases_alleged_violation_prediction_silver_rationales",
    "gigaword_summarize_",
]


class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, config, dataset_stash):
        """
        :param config:
        """
        self.config = config
        self.dataset_stash = dataset_stash

        self.templates = DatasetTemplates(*self.dataset_stash)
        self.train_template = self.get_template(self.config.train_template_idx)
        self.eval_template = self.get_template(self.config.eval_template_idx)

    def get_template(self, template_idx):
        template_names = self.templates.all_template_names
        if template_idx >= 0:
            return self.templates[template_names[template_idx]]
        elif template_idx == -1:

            list_idx = []
            list_templates = []
            for idx, template_name in enumerate(template_names):
                if self.templates[template_name].metadata.original_task:
                    list_idx.append(idx)
                    list_templates.append(self.templates[template_name])

            return list_templates
        elif template_idx == -2:
            return [self.templates[template_name] for template_name in template_names]

    def get_train_template(self):
        return self.train_template

    def get_eval_template(self):
        return self.eval_template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if os.path.exists(DATASETS_OFFLINE):
            try:
                orig_data = load_from_disk(
                    os.path.join(DATASETS_OFFLINE, *self.dataset_stash)
                )[split]
            except FileNotFoundError:
                orig_data = load_from_disk(
                    os.path.join(DATASETS_OFFLINE, self.dataset_stash[0])
                )[split]
        else:
            orig_data = load_dataset(
                *self.dataset_stash, split=split, cache_dir=os.environ["HF_HOME"]
            )
        return orig_data

    def read_few_shot_dataset(self):
        file_dir = os.path.join(
            "data", "few_shot", self.config.dataset, f"{self.config.num_shot}_shot"
        )
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(
            file_dir, f"{self.config.few_shot_random_seed}_seed.jsonl"
        )

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))

            return data
        else:
            orig_data = self.read_orig_dataset("train")
            selected_data = self._sample_few_shot_data(orig_data)

            with open(file_path, "w+") as fout:
                for example in selected_data:
                    fout.write(json.dumps(example) + "\n")
            return selected_data

    def _sample_few_shot_data(self, orig_data):
        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        orig_data = [x for x in orig_data]
        np.random.shuffle(orig_data)
        selected_data = orig_data[: self.config.num_shot]
        np.random.set_state(saved_random_state)
        return selected_data

    def compute_metric(self, accumulated):
        matching = [
            a == b for a, b in zip(accumulated["prediction"], accumulated["label"])
        ]
        accuracy = sum(matching) / len(matching)
        return {"accuracy": accuracy}


class CustomCategoricalReader(BaseDatasetReader):
    def __init__(self, config):
        task = config.dataset.split("_")[0].lower()
        # Select correct subtask (especially for right template)
        subtask = templates_for_custom_tasks[task]
        assert subtask is not None
        super().__init__(config, dataset_stash=(config.dataset, subtask))

    # There are no pre-defined templates for this custom task, so load them manually by hijacking this function.
    def get_template(self, template_idx):
        # Add custom template
        task = self.config.dataset.split("_")[0].lower()
        yaml_dict = yaml.load(
            open("./templates/templates_" + task + ".yaml", "r"), Loader=yaml.FullLoader
        )
        prompts = yaml_dict["templates"]

        # Set DatasetTemplates object in self.templates to None bs cannot build it here
        self.templates = None
        # Return a list of prompts (usually only a single one with dataset_stash[1] name)
        return [t for k, t in prompts.items() if t.get_name() == self.dataset_stash[1]]

    def read_orig_dataset(self, split):
        # External datasets are not yet shuffled, so do it now
        orig_data = load_from_disk(
            os.path.join(DATASETS_OFFLINE, self.dataset_stash[0])
        )
        # Debug output for importance
        split_data = True  # Default True
        if split_data:
            data = orig_data.train_test_split(test_size=0.20, seed=self.config.seed)
            data2 = data["test"].train_test_split(test_size=0.50, seed=self.config.seed)
            # No validation/test split used for external datasets
            dataset_dict = DatasetDict(
                {
                    "train": data["train"],
                    "validation": concatenate_datasets([data2["train"], data2["test"]]),
                    "test": Dataset.from_dict({"note": [], "label": []}),
                }
            )
            orig_data = dataset_dict[split]

        # In case dataset has no idx per example, add that here bc manually created ones might not have an idx.
        if "idx" not in orig_data.column_names:
            orig_data = orig_data.add_column(
                name="idx", column=range(0, orig_data.num_rows)
            )

        return orig_data

    def _sample_few_shot_data(self, orig_data):
        if self.config.num_shot == "all":
            return [x for x in orig_data]

        if self.config.num_shot == 0 or self.config.num_shot == "0":
            return []

        # if not self.config.balanced_ibc:
        #     return super()._sample_few_shot_data(orig_data)

        saved_random_state = np.random.get_state()
        np.random.seed(self.config.few_shot_random_seed)
        # Create a balanced dataset for categorical data
        labels = {
            label: len([ex["idx"] for ex in orig_data if ex["label"] == label])
            for label in list(set(ex["label"] for ex in orig_data))
        }
        num_labels = len(labels.keys())
        ex_label = int(self.config.num_shot / num_labels)
        ex_last_label = self.config.num_shot - ((num_labels - 1) * ex_label)
        ex_per_label = (num_labels - 1) * [ex_label] + [ex_last_label]
        assert sum(ex_per_label) == self.config.num_shot

        # Select num instances per label
        old_num_labels = []
        datasets_per_label = []
        for i, label in enumerate(labels.keys()):
            indices = [ex["idx"] for ex in orig_data if ex["label"] == label]
            old_num_labels.append(len(indices))
            # Sample with replacement from label indices
            samples_indices = list(np.random.choice(indices, ex_per_label[i]))
            datasets_per_label.append(orig_data.select(samples_indices))
        orig_data = concatenate_datasets(datasets_per_label)

        # Check new labels
        old_labels = labels
        labels = {
            label: len([ex["idx"] for ex in orig_data if ex["label"] == label])
            for label in list(set(ex["label"] for ex in orig_data))
        }
        print(
            f"Via sampling with replacement old label distribution {old_labels} to new {labels}"
        )
        assert sum(labels.values()) == self.config.num_shot
        assert len(orig_data) == self.config.num_shot

        np.random.set_state(saved_random_state)
        # Now randomize and (selection of num_shots redundant now bc already done).
        return super()._sample_few_shot_data(orig_data)

    def compute_metric(self, accumulated):
        metrics = super().compute_metric(accumulated)
        # print(accumulated['probabilities'])

        binary = all([True if l in [0, 1] else False for l in accumulated["label"]])
        if binary:
            pos_probs = [p[1] for p in accumulated["probabilities"]]
            roc_auc = roc_auc_score(accumulated["label"], pos_probs)
            pr_auc = pr_auc_score(accumulated["label"], pos_probs)
        else:
            probs = [p for p in accumulated["probabilities"]]
            roc_auc = roc_auc_score(
                accumulated["label"], probs, multi_class="ovr", average="macro"
            )
            # Abuse pr for AUC ovo here
            pr_auc = roc_auc_score(
                accumulated["label"], probs, multi_class="ovo", average="macro"
            )

        micro_f1 = f1_score(
            accumulated["label"], accumulated["prediction"], average="micro"
        )
        macro_f1 = f1_score(
            accumulated["label"], accumulated["prediction"], average="macro"
        )
        metrics = {
            "AUC": roc_auc,
            "PR": pr_auc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            **metrics,
        }
        # Also record number of instances evaluated
        metrics = {**metrics, "num": len(accumulated["prediction"])}

        # Debug: Only for importance
        store_probabilities = False  # Default False
        if store_probabilities:
            prop_output = (
                "t0-probabilities-"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                + ".p"
            )
            with open(prop_output, "wb") as f:
                pickle.dump(accumulated["probabilities"], f)

        return metrics


def pr_auc_score(labels, probabilities):
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    return auc(recall, precision)
