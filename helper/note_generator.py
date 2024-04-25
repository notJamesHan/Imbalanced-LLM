import itertools
import json
import re
import datetime

from datetime import timedelta

import pandas as pd


gender_categories = ['female', 'male']
race_categories = ['race not recorded', 'hispanic or latino', 'asian', 'black or african american',
                   'american indian or alaska native', 'white', 'native hawaiian or other pacific islander']

class NoteGenerator:
    def __init__(self, task, data, training_end_date="2020-10-01", feature_weights_list=None, template_suffix='',
                 zero_shot_weights=None, concept_names=(None, None)):
        """ Parse data of a OMOP cohort into class structures. """
        self.person_ids = [person['person_id'] for person in data]  # correct ordering of persons
        self.visits = None
        self.non_temporal = None
        self.temporal = None
        self.visits, self.non_temporal, self.temporal = self.parse_data(data)
        assert set(self.person_ids) == set(self.visits['person_id'].tolist())

        # Load id - value mapping
        with open('/root/omop-pkg/misc/permute_concepts/id_value_map_eol_loh_surgery.txt', 'r') as file:
            self.values = {int(k): v for k, v in (json.load(file)).items()}
        if feature_weights_list is None or zero_shot_weights is not None:
            feature_weights_list = pd.DataFrame([], columns=['concept_id', 'interval', 'weight'])
        # Necessary to assign feature weights with intervals
        self.training_end_date = datetime.datetime.strptime(training_end_date, '%Y-%m-%d')
        self.task = task
        self.template_suffix = template_suffix

        self.shortened_conditions = read_shortened_concepts(concept_names[0])
        self.shortened_procedures = read_shortened_concepts(concept_names[1])

        # Read dictionaries for permuting concepts
        def read_permuted_concepts(file_name):
            with open('/root/omop-pkg/misc/permute_concepts/' + file_name) as f:
                return eval(f.readline())
        self.permuted_conditions = read_permuted_concepts('condition_ids_permuted.txt')
        self.permuted_procedures = read_permuted_concepts('procedure_ids_permuted.txt')

        # Pandas debug settings
        # pd.set_option('display.max_rows', 500)
        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 500)

        # Create unique temporal concepts for each visit
        self.temporal = self.temporal.drop_duplicates(ignore_index=True)
        # If numbered conditions exist, prefer them over non-numbered
        self.temporal.loc[self.temporal['concept'] == 'condition', 'concept'] = 'condition none'
        self.temporal = pd.concat([self.temporal.loc[~(self.temporal['concept'].str.startswith('condition'))],
                                   self.temporal.loc[self.temporal['concept'].str.startswith('condition')].
                                  sort_values('concept').drop_duplicates(
                                       [c for c in self.temporal.columns if c != 'concept'], keep='first')])
        self.temporal.sort_values(['person_id', 'visit_id', 'concept'], inplace=True)

        # Read feature weights and assign them to concepts
        # feat_name_regex = re.compile(r"(\d+) - (condition|procedure|drug) - (.+) - (\d+) days")
        self.feature_weights = feature_weights_list
        # Distinguish feature weight file with person specific weights
        if 'person_id' in feature_weights_list.columns:
            self.feature_weights['person_id'] = self.feature_weights['person_id'].astype(int)
            assert set(self.person_ids) == set(self.feature_weights['person_id'].tolist())
        self.feature_weights['concept_id'] = self.feature_weights['concept_id'].astype(int)
        self.feature_weights['interval'] = pd.to_timedelta(pd.to_numeric(self.feature_weights['interval']), unit='days')
        self.feature_weights['interval_start'] = self.training_end_date - self.feature_weights['interval']
        self.feature_weights['abs_weight'] = abs(self.feature_weights['weight'])

        if not self.feature_weights.empty:
            # Assign feature weights to visits when they occur in given interval
            original_columns = self.visits.columns.tolist()
            original_size = self.visits.shape[0]
            # Assign values to concept_ids temporarily bc visit entries have no concept_id
            self.feature_weights.loc[self.feature_weights['concept_id'] == 581458, 'value'] = 'Pharmacy visit'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 581477, 'value'] = 'Office Visit'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 8546, 'value'] = 'Hospice'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 9201, 'value'] = 'Inpatient Visit'
            self.feature_weights.loc[self.feature_weights['concept_id'] == 9202, 'value'] = 'Outpatient Visit'
            # Distinguish feature weight file with person specific weights
            if 'person_id' not in feature_weights_list.columns:
                self.visits = self.visits.merge(self.feature_weights.loc[self.feature_weights['concept_id'].isin([581458, 581477, 8546, 9201, 9202])],
                                                how='left', left_on='type', right_on='value')
            else:
                self.visits = self.visits.merge(self.feature_weights.loc[self.feature_weights['concept_id'].isin([581458, 581477, 8546, 9201, 9202])],
                                                how='left', left_on=['type', 'person_id'], right_on=['value', 'person_id'])
                assert set(self.visits['person_id'].tolist()) == set(self.feature_weights['person_id'].tolist())
            self.feature_weights = self.feature_weights.drop(columns=['value'])
            self.visits[['weight', 'abs_weight']] = self.visits[['weight', 'abs_weight']].fillna(0.)
            # Set all weights for visits before interval to zero
            self.visits.loc[self.visits['start'] < self.visits['interval_start'], ['weight', 'abs_weight']] = 0
            # Only keep weight of earliest visit (also ignore interval now)
            self.visits.sort_values('start', ascending=False, inplace=True)
            self.visits['num_visits_per_category'] = self.visits.groupby(['person_id', 'value']).cumcount()
            self.visits.loc[self.visits['num_visits_per_category'] != 0, ['weight', 'abs_weight']] = 0
            self.visits = self.visits.drop(columns=['num_visits_per_category'])

            # Sum all interval weights together.
            self.visits = self.visits[original_columns + ['weight', 'abs_weight']] \
                .groupby(original_columns, dropna=False)[['weight', 'abs_weight']].sum().reset_index()
            # Lowercase specialties
            self.visits['specialty'] = self.visits['specialty'].str.lower()
            assert original_size == self.visits.shape[0]
        else:
            self.visits[['weight', 'abs_weight']] = 0

        original_columns = self.temporal.columns.tolist()
        original_size = self.temporal.shape[0]
        if not self.feature_weights.empty:
            # Assign feature weights to concepts when they occur in the given interval
            self.temporal = self.temporal.merge(self.visits[['id', 'start']], how='left', left_on='visit_id', right_on='id')
            # Distinguish feature weight file with person specific weights
            if 'person_id' not in feature_weights_list.columns:
                self.temporal = self.temporal.merge(
                    self.feature_weights.loc[:, [c for c in self.feature_weights.columns if c not in ['concept', 'value']]],
                    how='left', on='concept_id')
            else:
                self.temporal = self.temporal.merge(
                    self.feature_weights.loc[:, [c for c in self.feature_weights.columns if c not in ['concept', 'value']]],
                    how='left', on=['concept_id', 'person_id'])
            self.temporal[['weight', 'abs_weight']] = self.temporal[['weight', 'abs_weight']].fillna(0.)
            # Set all weights for visits before interval to zero and sum all interval weights together.
            self.temporal.loc[self.temporal['start'] < self.temporal['interval_start'], ['weight', 'abs_weight']] = 0
            self.temporal = self.temporal[original_columns + ['weight', 'abs_weight']] \
                .groupby(original_columns)[['weight', 'abs_weight']].sum().reset_index()
            assert original_size == self.temporal.shape[0]
        else:
            self.temporal[['weight', 'abs_weight']] = 0
            self.temporal = self.temporal.merge(self.visits[['id', 'start']], how='left', left_on='visit_id', right_on='id')

            # Optionally if specific zero shot weighting given use this schema
            if zero_shot_weights.startswith('most_frequent'):
                self.temporal['weight'] = self.temporal.groupby(['person_id', 'concept_id'])['concept_id'].transform('count')
            elif zero_shot_weights.startswith('least_frequent'):
                self.temporal['weight'] = 10000 - self.temporal.groupby(['person_id', 'concept_id'])['concept_id'].transform('count')
            elif zero_shot_weights.startswith('oldest'):
                self.temporal['weight'] = (self.training_end_date - self.temporal['start']).dt.days
            elif zero_shot_weights.startswith('recent'):
                self.temporal['weight'] = 10000 - (self.training_end_date - self.temporal['start']).dt.days
            # Only keep selected concepts
            if zero_shot_weights.endswith('_conditions'):
                self.temporal.loc[self.temporal['concept'].str.startswith('procedure'), 'weight'] = 0.
            if zero_shot_weights.endswith('_procedures'):
                self.temporal.loc[self.temporal['concept'].str.startswith('condition'), 'weight'] = 0.

            self.temporal['abs_weight'] = abs(self.temporal['weight'])
            self.temporal = self.temporal[original_columns + ['weight', 'abs_weight']]

        # Create unique concept numbering. First criterion existing numbering, second weight, third concept_id.
        self.temporal = self.temporal.sort_values(by=['person_id', 'visit_id', 'concept', 'weight', 'concept_id'],
                                                  ascending=[True, True, True, False, True])
        # Columns concept_id and abs_weight not necessary here
        self.temporal.loc[
            self.temporal['concept'].str.startswith('condition'),
            'concept'] = 'condition' + (self.temporal.loc[self.temporal['concept'].str.startswith('condition')]
                                        .groupby(by=['person_id', 'visit_id'], sort=False).cumcount() + 1).apply(str)
        self.temporal.sort_values(['person_id', 'visit_id', 'concept'], inplace=True)
        assert original_size == self.temporal.shape[0]

        # Lowercase first letter of concept values
        self.values = {k: lower_first_char_if_second_low(v) for k, v in self.values.items()}

        # Add additional non temporal variables
        self.non_temporal.rename({'Age at end_date': 'age'}, axis=1, inplace=True)
        self.non_temporal['gender'] = [gender_categories[i] for i in self.non_temporal['Gender M(1)/F(0)'].to_list()]
        self.non_temporal['race_text'] = [race_categories[i] if race_categories[i] != 'race not recorded' else None
                                          for i in self.non_temporal['Race'].to_list()]

        # Debug: Output concept renaming
        # def output_concept_name_statistics(x):
        #     collect = []
        #     for concept_id in self.temporal['concept_id'].tolist():
        #         if concept_id in self.values.keys():
        #             if self.values[concept_id] in x.keys():
        #                 collect.append((self.values[concept_id], x[self.values[concept_id]]))
        #     c = list(Counter(collect).items())
        #     c.sort(key=lambda x: x[1], reverse=True)
        #     for i in range(0, min(100, len(c))):
        #         print(c[i][1], c[i][0][0], '--->', c[i][0][1])

        # output_concept_name_statistics(read_shortened_concepts('conditions_eol_loh_surgery_short.txt'))
        # output_concept_name_statistics(read_shortened_concepts('procedures_eol_loh_surgery_short.txt'))

   
    @staticmethod
    def parse_data(dataset):
        """ Parse data into separate dataframes for visits, non-temporal and temporal dataframes. """
        # Parse non-temporal attributes
        non_temporal = (dataset.remove_columns(['visits', 'dates', 'tok_visits'])).to_pandas()
        # Columns to expect as attributes for each visit
        visit_data_columns = ['id', 'type', 'start', 'duration', 'specialty']
        temporal_data_columns = ['person_id', 'visit_id', 'concept_id', 'concept']

        # Define and compiles regexes
        visit_regex = re.compile(r"\d+ - visit (\w+) - (.+)")
        concept_regex = re.compile(r"(\d+) - (condition|condition \d+|procedure|drug) - (.+)")

        # Prepare dataset
        dataset = dataset.add_column('person_visits', [[]] * len(dataset))
        dataset = dataset.add_column('person_temporal', [[]] * len(dataset))

        def parse_visits_and_temporal(person):
            assert len(person['visits']) == len(person['dates'])
            # Go through all visit entries in reverse order because they can actually correspond to several visits.
            # Use the visit_id to merge same visits. Going through them in reverse order allows that the most
            # relevant visit (the first) can overwrite visit related data from later ones.
            person['dates'] = [pd.to_datetime(d) for d in person['dates']]
            person_visits = {}
            person_temporal = []
            for i, visit in reversed(list(enumerate(person['visits']))):
                visit_id = int((person['dates'][i] - person['dates'][i]
                                .replace(microsecond=0, second=0, minute=0, hour=0)) / timedelta(microseconds=1))
                if visit_id not in person_visits.keys():
                    # Add empty default entries in case they are not provided
                    person_visits[visit_id] = {k: None for k in visit_data_columns}
                visit_data = []
                for concept in visit:
                    if visit_regex.match(concept):
                        # Parse all visits to get visit data as tuples (attr, value)
                        visit_data.append(visit_regex.search(concept).groups())
                    elif concept_regex.match(concept):
                        # All other concepts
                        # Need to keep everything as string to make it hf dataset compatible
                        id_concept_value = list(concept_regex.search(concept).groups())
                        person_temporal.append([str(person['person_id']), str(visit_id)] + id_concept_value[0:2])
                    else:
                        raise ValueError(f"Unknown concept string: {concept}")

                # Earlier visits with the same id update this data dictionary of the visit
                person_visits[visit_id].update(dict(visit_data))
                person_visits[visit_id]["person_id"] = str(person['person_id'])

            # Parse visit data into list format with fixed ordering
            person['person_visits'] = [[visit_dict[k] for k in (['person_id'] + visit_data_columns)]
                                       for visit_dict in person_visits.values()]
            person['person_temporal'] = person_temporal
            return person

        # dataset = [parse_visits_and_temporal(ex) for ex in dataset]
        dataset = dataset.map(parse_visits_and_temporal)

        # Create dataframe from list of lists of visit data / temporal data for each person
        temporal = pd.DataFrame(itertools.chain.from_iterable(dataset['person_temporal']), columns=temporal_data_columns)
        temporal[['person_id', 'visit_id', 'concept_id']] = temporal[['person_id', 'visit_id', 'concept_id']].apply(pd.to_numeric)
        visits = pd.DataFrame(itertools.chain.from_iterable(dataset['person_visits']), columns=['person_id'] + visit_data_columns)
        dataset = dataset.remove_columns(['person_visits', 'person_temporal'])

        # There is an edge case where concepts before inclusion date but visit after, so visit id not included.
        old_num_visits = visits.shape[0]
        visits = visits[visits['id'].notna()]
        if old_num_visits - visits.shape[0] > 0:
            print(f"Deleted {old_num_visits - visits.shape[0]} artifact visits without a visit id.")
        assert len(visits['id'].unique().tolist()) == len(visits['id'].tolist())

        visits[['id', 'duration', 'person_id']] = visits[['id', 'duration', 'person_id']].apply(pd.to_numeric)
        visits['duration'] = pd.to_timedelta(visits['duration'], unit='days')
        visits['start'] = pd.to_datetime(visits['start'], format='%Y-%m-%d')

        return visits, non_temporal, temporal

    @staticmethod
    def clean_note(note):
        # Template remove all repeated whitespaces and more than double newlines
        note = re.sub(r"[ \t]+", " ", note)
        note = re.sub("\n\n\n+", "\n\n", note)
        # Remove all leading and trailing whitespaces
        note = re.sub(r"^[ \t]+", "", note)
        note = re.sub(r"\n[ \t]+", "\n", note)
        note = re.sub(r"[ \t]$", "", note)
        note = re.sub(r"[ \t]\n", "\n", note)
        # Remove whitespaces before colon at the end of the line
        note = re.sub(r"\s*\.$", ".", note)
        note = re.sub(r"\s*\.\n", ".\n", note)
        # Remove repeated dots and the end of the line
        note = re.sub(r"\.+$", ".", note)
        note = re.sub(r"\.+\n", ".\n", note)
        # Remove whitespaces before colon at the end of the line
        note = re.sub(r"\s*\.$", ".", note)
        note = re.sub(r"\s*\.\n", ".\n", note)
        # Template remove all repeated whitespaces and more than double newlines
        note = re.sub(r"[ \t]+", " ", note)
        note = re.sub("\n\n\n+", "\n\n", note)
        # Remove repetitive whitespace colon sequences
        # Ignore for ... in creditg dataset
        if '... ' not in note:
            note = re.sub(r"(\s*\.)+ +", ". ", note)

        return note


def lower_first_char_if_second_low(s):
    return (s if s[1].isupper() else s[0].lower() + s[1:]) if (not pd.isna(s) and len(s) > 1) else s


def read_shortened_concepts(file_name):
    # Read dictionaries for shortening concepts
    if file_name is None:
        return {}
    with open('/root/omop-pkg/misc/shorten_concepts/' + file_name) as f:
        shortened_concepts = eval(f.readline())
        for k, v in shortened_concepts.items():
            v = v.strip()
            v = v if v[-1].isalnum() else v[:-1]
            # Lower first character if second not upper case
            shortened_concepts[k] = lower_first_char_if_second_low(v)
    # Lower all values longer than four character that are all upper case
    return {k.lower(): v.lower() if (len(v) > 4 and v.isupper()) else v for k, v in shortened_concepts.items()}

