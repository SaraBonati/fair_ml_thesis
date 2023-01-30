# This file defines a Sklearn_Model class that preprocesses the data, defines which classifier to use
# and which  metrics are used to evaluate the classifier predictions.
#
# Author @ Sara Bonati
# Supervisor: Prof. Dr. Claudia Müller-Birn - Prof. Dr. Eirini Ntoutsi
# Project: Data Science MSc Master Thesis
#############################################

# general utility import
import argparse
import glob
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from aif360.sklearn.metrics import statistical_parity_difference, disparate_impact_ratio
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.sklearn.inprocessing import AdversarialDebiasing, GridSearchReduction, ExponentiatedGradientReduction
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


#########################
# START CLASS DEFINITION
#########################
class Model:

    def __init__(self, train_data_source, test_data_source, task, cat_columns, num_columns, target_col: str,
                 spatial_year=None, temporal_train_year=None, temporal_test_years=None, train_state=None,
                 test_states=None):
        """_summary_

        Args:
            train_data_source (_type_): _description_
            test_data_source (_type_): _description_
            task (_type_): _description_
            cat_columns (_type_): _description_
            num_columns (_type_): _description_
            target_col (_type_): description
            spatial_year (_type_, optional): _description_. Defaults to None.
            temporal_train_year (_type_, optional): _description_. Defaults to None.
            temporal_test_years (_type_, optional): _description_. Defaults to None.
            train_state (_type_, optional): _description_. Defaults to None.
            test_states (_type_, optional): _description_. Defaults to None.

        Returns:
            None, initializes Model object
        """

        # assert tests
        assert task in ["ACSEmployment", "ACSIncome", "ACSHealthInsurance"], \
            f'Task should be a value in {["ACSEmployment", "ACSIncome", "ACSHealthInsurance"]}, got {task} instead'

        # Task name
        self.task = task
        # collect the categorical and numerical columns of this task
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.target_col = target_col
        print("TARGET COL: ",self.target_col)
        # year chosen for the spatial context (if spatial context was selected)
        self.spatial_year = spatial_year
        # training year (if temporal context was selected)
        self.train_year = temporal_train_year
        # test years (if temporal context was selected)
        self.test_years = temporal_test_years

        # Prepare train data
        self.train_df = train_data_source
        self.train_state = train_state
        self.cols = self.train_df.columns
        self.y_train = self.train_df[self.target_col]

        # Prepare test data
        self.all_states = False
        if type(test_data_source) is list:
            self.all_states = True
            self.test_states = test_states
            self.test_dfs = test_data_source
        else:
            self.test_df = test_data_source
            self.y_test = self.test_df[self.target_col]

        print(self.test_states)
        # seed (for reproducible output across multiple function calls)
        self.seed = 42

        # define classifiers available
        estimator = LogisticRegression(solver='liblinear', random_state=self.seed)
        self.clf_set = {"AdversarialDebiasing": AdversarialDebiasing(prot_attr=['SEX', 'RAC1P'],
                                                                     random_state=self.seed)#,
                        #"ExponentiatedGradientReduction": ExponentiatedGradientReduction(prot_attr=['SEX','RAC1P'],
                        #                                                                estimator=estimator,
                        #                                                                constraints="EqualizedOdds",
                        #                                                                drop_prot_attr=False,
                        #                                                                random_state=self.seed),
                        }

        # define cross-validation strategy
        self.cv_set = {'KFold': KFold(n_splits=5, shuffle=True, random_state=self.seed)}

    def preprocess(self, df):
        """
        This method creates a pre-processing pipeline to be used on the training and testing data.
        The pipeline includes:

        - for numerical columns (e.g. age) an imputer first (to deal with NaN values) followed by a StandardScaler
          to perform feature standardization
        - for categorical columns a One Hot Encoder for each level of the categorical factor/feature

        Args:
            df (pd.Dataframe): the dataframe on which to apply preprocessing
        """

        df['RAC1P'] = pd.to_numeric(df['RAC1P'])
        # take care of special case of RAC1P column for ACSHealthInsurance task
        if self.task == "ACSHealthInsurance":
            df['RAC1P'] = df[['RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACPI', 'RACSOR', 'RACWHT']].idxmax(axis=1)
            race_codes = {'RACAIAN': 5, 'RACASN': 6, 'RACBLK': 2, 'RACNH': 7, 'RACPI': 7, 'RACSOR': 8, 'RACWHT': 1}
            df['RAC1P'] = df['RAC1P'].map(race_codes)
            df.drop(['RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACPI', 'RACSOR', 'RACWHT'], axis=1, inplace=True)
        # recode RAC1P values for all tasks
        df.loc[df['RAC1P'] > 2, 'RAC1P'] = 3

        # set protected attributes to be an index (this is the format expected by aif360)
        prot_attr_copy = df[['SEX', 'RAC1P']]
        df.index = pd.MultiIndex.from_frame(prot_attr_copy)
        #df.set_index(['SEX', 'RAC1P'],inplace=True)

        # turn most columns into categories (usually first col is age, that stays numeric)
        for i in self.cat_columns:
            # first remove the trailing zero by turning categorical numbers into ints
            df.loc[:, i] = df[i].astype(np.int64)
            # then treat columns as categories
            df.loc[:, i] = df[i].astype("category")

        # apply standard scaler to numeric variables
        X_num = df[self.num_columns]
        numeric_transformer = StandardScaler()
        X_num.loc[:, self.num_columns] = numeric_transformer.fit_transform(X_num[self.num_columns])

        # apply one hot encoding to categorical columns
        # X_cat = df[self.cat_columns[:-2]]
        # X_cat = pd.get_dummies(X_cat, columns=self.cat_columns[:-2], drop_first=False)
        X_cat = df[self.cat_columns]
        X_cat = pd.get_dummies(X_cat, columns=self.cat_columns, drop_first=False)
        # print("Cat columns: ", self.cat_columns)
        # print("X_cat columns: ", X_cat.columns)

        return pd.concat([X_num, X_cat], axis=1)

    def test_model_spatial(self):
        """
        This method tests sklearn classifiers for the specific task
        """

        metricss = {}

        for clf_name, clfier in self.clf_set.items():

            # initialize fairness metrics dict + specify clfier results
            metricss[clf_name] = {}

            # fit on training data (one state)
            # for train, test in cv_object.split(bold_masked,**cv_object_args):
            self.X_train = self.preprocess(self.train_df)
            clfier.fit(self.X_train, self.y_train)

            for t in range(len(self.test_dfs)):

                # create testing data
                self.X_test_to_preprocess, self.y_test = self.test_dfs[t][self.cols[:-1]], self.test_dfs[t][
                    self.target_col]
                # create y_test in the form of dataframe indexed
                # by protected attributes (for metric calculation purposes)
                self.y_test_df = self.X_test_to_preprocess[['SEX', 'RAC1P']]
                self.y_test_df[self.target_col] = self.y_test
                self.y_test_df.set_index(['SEX', 'RAC1P'], inplace=True)
                self.X_test = self.preprocess(self.X_test_to_preprocess)

                # because some categories might not be present in test data but are still expected by
                # the classifier after fit check if this is the case, and if yes add these columns to X_test
                cols_to_fill = set(self.X_train.columns) - set(self.X_test.columns)
                print("COLS TO FILL:", cols_to_fill)
                if cols_to_fill:
                    for c in cols_to_fill:
                        self.X_test[c] = 0
                self.X_test = self.X_test[self.X_train.columns]

                # get predictions from classifier
                y_pred = clfier.predict(self.X_test)

                # calculate metrics
                metricss[clf_name][self.test_states[t]] = {}
                metricss[clf_name][self.test_states[t]]['accuracy'] = accuracy_score(self.y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['bal_accuracy'] = balanced_accuracy_score(self.y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['precision'] = precision_score(self.y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['recall'] = recall_score(self.y_test, y_pred)

                metricss[clf_name][self.test_states[t]]['sex_spd'] = statistical_parity_difference(self.y_test_df,
                                                                                                   y_pred,
                                                                                                   prot_attr='SEX',
                                                                                                   priv_group=1,
                                                                                                   pos_label=1,
                                                                                                   sample_weight=None)
                metricss[clf_name][self.test_states[t]]['sex_dir'] = disparate_impact_ratio(self.y_test_df,
                                                                                            y_pred,
                                                                                            prot_attr='SEX',
                                                                                            priv_group=1,
                                                                                            pos_label=1,
                                                                                            sample_weight=None)
                metricss[clf_name][self.test_states[t]]['rac_spd'] = statistical_parity_difference(self.y_test_df,
                                                                                                   y_pred,
                                                                                                   prot_attr='RAC1P',
                                                                                                   priv_group=1,
                                                                                                   pos_label=1,
                                                                                                   sample_weight=None)
                metricss[clf_name][self.test_states[t]]['rac_dir'] = disparate_impact_ratio(self.y_test_df,
                                                                                            y_pred,
                                                                                            prot_attr='RAC1P',
                                                                                            priv_group=1,
                                                                                            pos_label=1,
                                                                                            sample_weight=None)
            print(metricss[clf_name])
            # save all test states' results
            dfObj = pd.DataFrame.from_dict(metricss[clf_name], orient='index')
            # create results folder for the training state if not present already
            if not os.path.isdir(os.path.join(rdir, self.task, str(self.spatial_year), 'aif360', self.train_state)):
                os.makedirs(os.path.join(rdir, self.task, str(self.spatial_year), 'aif360', self.train_state))
            # save results
            dfObj.to_csv(os.path.join(rdir, self.task, str(self.spatial_year), 'aif360', self.train_state,
                                      f'spatial_{self.train_state}_test_all_{clf_name}.csv'),
                         encoding='utf-8',
                         index=True)

        return metricss

    def test_model_temporal(self):
        """
        This method tests sklearn classifiers for the specific task in the temporal context
        e.g. given a training state CA in year 2014 we train a classifier on CA 2014 and
        test on CA 2015, CA 2016 ...
        """

        metricss = {}

        for clf_name, clfier in self.clf_set.items():

            # initialize fairness metrics dict + specifiy clfier results
            metricss[clf_name] = {}

            # fit on training data (one state)
            # for train, test in cv_object.split(bold_masked,**cv_object_args):
            self.X_train = self.preprocess(self.train_df)
            clfier.fit(self.X_train, self.y_train)

            # loop over all test states, get metrics and save results in .tsv file
            for t in range(len(self.test_dfs)):
                print(f"YEAR TEST: {self.test_years[t]}")
                # metrics init
                metricss[clf_name][self.test_years[t]] = {}

                # create testing data
                self.X_test_to_preprocess, self.y_test = self.test_dfs[t][self.cols[:-1]], self.test_dfs[t][
                    self.target_col]
                print(self.y_test)
                # create y_test in the form of dataframe indexed vby protected attributes
                # (for metric calculation purposes)
                self.y_test_df = self.X_test_to_preprocess[['SEX', 'RAC1P']]
                self.y_test_df['ESR'] = self.y_test
                self.y_test_df.set_index(['SEX', 'RAC1P'], inplace=True)
                self.X_test = self.preprocess(self.X_test_to_preprocess)

                # because some categories might not be present in test data but are still expected by
                # the classifier after fit check if this is the case, and if yes add these columns to X_test
                cols_to_fill = set(self.X_train.columns) - set(self.X_test.columns)
                print(cols_to_fill)
                if cols_to_fill:
                    for c in cols_to_fill:
                        self.X_test[c] = 0
                self.X_test = self.X_test[self.X_train.columns]

                # get predictions from classifier
                y_pred = clfier.predict(self.X_test)

                print(self.y_test.value_counts())
                print(y_pred)
                # calculate metrics
                metricss[clf_name][self.test_years[t]] = {}
                metricss[clf_name][self.test_years[t]]['accuracy'] = accuracy_score(self.y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['bal_accuracy'] = balanced_accuracy_score(self.y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['precision'] = precision_score(self.y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['recall'] = recall_score(self.y_test, y_pred)
                # calculate fairness metrics
                metricss[clf_name][self.test_years[t]]['sex_spd'] = statistical_parity_difference(self.y_test_df,
                                                                                                  y_pred,
                                                                                                  prot_attr='SEX',
                                                                                                  priv_group=1,
                                                                                                  pos_label=1,
                                                                                                  sample_weight=None)
                metricss[clf_name][self.test_years[t]]['sex_dir'] = disparate_impact_ratio(self.y_test_df,
                                                                                           y_pred,
                                                                                           prot_attr='SEX',
                                                                                           priv_group=1,
                                                                                           pos_label=1,
                                                                                           sample_weight=None)
                metricss[clf_name][self.test_years[t]]['rac_spd'] = statistical_parity_difference(self.y_test_df,
                                                                                                  y_pred,
                                                                                                  prot_attr='RAC1P',
                                                                                                  priv_group=1,
                                                                                                  pos_label=1,
                                                                                                  sample_weight=None)
                metricss[clf_name][self.test_years[t]]['rac_dir'] = disparate_impact_ratio(self.y_test_df,
                                                                                           y_pred,
                                                                                           prot_attr='RAC1P',
                                                                                           priv_group=1,
                                                                                           pos_label=1,
                                                                                           sample_weight=None)

            # save all test states' results
            dfObj = pd.concat({k: pd.DataFrame(v).T for k, v in metricss.items()}, axis=0)
            # create results folder for the training state if not present already
            if not os.path.isdir(os.path.join(rdir, self.task, str(self.train_year), 'aif360', self.train_state)):
                os.makedirs(os.path.join(rdir, self.task, str(self.train_year), 'aif360', self.train_state))
            # save results
            dfObj.to_csv(os.path.join(rdir, self.task, str(self.train_year), 'aif360', self.train_state,
                                      f'temporal_{self.train_state}_test_{str(self.train_year)}.csv'),
                         encoding='utf-8',
                         index=True)

        return metricss


#########################
# SCRIPT
#########################
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Run AIF360 models over the data')
    parser.add_argument('--mode', type=str, help='Which context do you want to explore (spatial or temporal)',
                        required=True)
    parser.add_argument('--task', type=str,
                        help='Which classification task do you want to run',
                        required=True)
    args = parser.parse_args()

    # directory management
    wdir = os.getcwd()
    udir = os.path.join(os.path.split(wdir)[0], "utils")
    ddir = os.path.join(os.path.split(os.path.split(wdir)[0])[0], "fair_ml_thesis_data", "rawdata")
    rdir = os.path.join(os.path.split(os.path.split(wdir)[0])[0], "fair_ml_thesis_data", "results")

    # json task specs
    json_file_path = os.path.join(udir, 'tasks_metadata.json')
    with open(json_file_path, 'r') as j:
        task_infos = json.loads(j.read())

    # define regex pattern to extract names of all test states
    test_state_pattern = re.compile(r"_([^_]+)_")

    # if mode is spatial:
    if args.mode == "spatial":

        for select_year in task_infos["years"]:
            data_file_paths = glob.glob(
                os.path.join(ddir, str(select_year), '1-Year') + f'/{str(select_year)}_*_{args.task}.csv')
            data_file_paths.sort()

            for select_state_train in task_infos["states"][1:]:
                print("YEAR ",select_year," TRAIN STATE: ",select_state_train)
                # load train data
                train_df = pd.read_csv(os.path.join(ddir, str(select_year), '1-Year',
                                                    f'{str(select_year)}_{select_state_train}_{args.task}.csv'),
                                       sep=',', index_col=0)

                # load test data
                test_dfs = []
                test_states = []
                test_data_file_paths = [f for f in data_file_paths if select_state_train not in f]
                for p in range(len(test_data_file_paths)):
                    test_path = Path(test_data_file_paths[p])
                    test_states.append(test_state_pattern.findall(test_path.name)[-1])
                    test_df = pd.read_csv(test_data_file_paths[p], sep=',', index_col=0)
                    test_dfs.append(test_df)

                M = Model(train_df, test_dfs, args.task, task_infos['tasks'][task_infos['task_col_map'][args.task]][
                                                                                 'cat_columns'],
                          #task_infos['tasks'][1]['cat_columns'],
                          task_infos['tasks'][1]['num_columns'],
                          task_infos["tasks"][task_infos["task_col_map"][args.task]]["target"],
                          spatial_year=select_year,
                          train_state=select_state_train,
                          test_states=test_states)
                metrics = M.test_model_spatial()

    # if mode is temporal:
    if args.mode == "temporal":

        for select_year in task_infos["years"]:
            # train year paths
            data_file_paths = glob.glob(os.path.join(ddir, str(select_year), '1-Year') +
                                        f'/{str(select_year)}*_{args.task}.csv')
            data_file_paths.sort()

            # loop over each state (the training state)
            for select_state_train in task_infos["states"]:

                print(f"Context: {args.mode}, current year: {select_year}, current_state: {select_state_train}")
                print(f"\n")
                train_df = pd.read_csv(os.path.join(ddir, str(select_year), '1-Year',
                                                    f'{str(select_year)}_{select_state_train}_{args.task}.csv'),
                                       sep=',', index_col=0)

                test_dfs = []
                test_states = []
                years_to_test = [item for item in task_infos["years"] if item not in [select_year]]
                test_data_file_paths = [
                    os.path.join(ddir, str(y), '1-Year', f'{str(y)}_{select_state_train}_{args.task}.csv')
                    for y in years_to_test]

                for p in range(len(test_data_file_paths)):
                    test_path = Path(test_data_file_paths[p])
                    test_states.append(test_state_pattern.findall(test_path.name)[-1])
                    test_df = pd.read_csv(test_data_file_paths[p], sep=',', index_col=0)
                    test_dfs.append(test_df)

                M = Model(train_df, test_dfs, args.task, task_infos['tasks'][task_infos['task_col_map'][args.task]][
                                                                                 'cat_columns'],
                          task_infos['tasks'][1]['num_columns'],
                          task_infos["tasks"][task_infos["task_col_map"][args.task]]["target"],
                          temporal_train_year=select_year,
                          temporal_test_years=years_to_test,
                          train_state=select_state_train,
                          test_states=test_states)
                metrics = M.test_model_temporal()
