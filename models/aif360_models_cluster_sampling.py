# This file defines a Sklearn_Model class that preprocesses the data, defines which classifier to use
# and which  metrics are used to evaluate the classifier predictions.
#
# Author @ Sara Bonati
# Supervisor: Prof. Dr. Claudia Müller-Birn - Prof. Dr. Eirini Ntoutsi
# Project: Data Science MSc Master Thesis
#############################################

# general utility import
import argparse
import numpy as np
import pandas as pd
import os
import glob
import json
import re
import pickle
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_curve, \
    roc_auc_score

from aif360.sklearn.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, \
true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate


#########################
# START CLASS DEFINITION
#########################
class Model:

    def __init__(self, train_data_source, test_data_paths, task, columns, cat_columns, num_columns,
                 target_col: str, spatial_year=None,
                 temporal_train_year=None, temporal_test_years=None, train_state=None, test_states=None):
        """_summary_

        Args:
            train_data_source (_type_): _description_
            test_data_paths (_type_): _description_
            task (_type_): _description_
            columns (_type_): _description_
            cat_columns (_type_): _description_
            num_columns (_type_): _description_
            spatial_year (_type_, optional): _description_. Defaults to None.
            temporal_train_year (_type_, optional): _description_. Defaults to None.
            temporal_test_years (_type_, optional): _description_. Defaults to None.
            train_state (_type_, optional): _description_. Defaults to None.
            test_states (_type_, optional): _description_. Defaults to None.

        Returns:
            None, initializes Model object
        """

        # assert tests
        assert task in ["ACSEmployment"], f'Task should be a value in {["ACSEmployment"]}, got {task} instead'

        # Task name
        self.task = task
        # collect all columns, the categorical and numerical columns of this task
        self.columns = columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.target_col = target_col
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
        if type(test_data_paths) is list:
            self.all_states = True
            self.test_states = test_states
            self.test_data_paths = test_data_paths

        else:
            self.test_data_paths = test_data_paths
            self.y_test = self.test_df[self.target_col]

        # seed (for reproducible output across multiple function calls)
        self.seed = 42

        estimator = LogisticRegression(solver='lbfgs', max_iter=10000)
        prot_attr_cols=['SEX_1','SEX_2','RAC1P_1','RAC1P_2','RAC1P_3']
        # define available classifiers + cross-validation strategy
        self.clf_set = {'AdversarialDebiasing': AdversarialDebiasing(prot_attr=['SEX', 'RAC1P'],
                             debias=True,
                             verbose=True,
                             random_state=42),
                        'ExponentiatedGradientReduction': ExponentiatedGradientReduction(
                            prot_attr=prot_attr_cols,
                            estimator=estimator,
                            constraints="EqualizedOdds",
                            drop_prot_attr=False)
                        }
        # define cross-validation strategy
        self.cv_object = KFold(n_splits=10, shuffle=True, random_state=self.seed)

    def preprocess(self, df):

        # turn everything into int
        #for c in task_infos['tasks'][1]['columns']:
        #    df.loc[:, c] = df.loc[:, c].astype(int)

        # filter for age
        #df = df[df['AGEP'].between(16, 90)]
        df.reset_index(drop=True,inplace=True)
        target = df['ESR']
        target.reset_index(drop=True, inplace=True)

        # record race column
        #df.loc[df['RAC1P'] > 2, 'RAC1P'] = 3
        sex_col = df['SEX']
        race_col = df['RAC1P']

        prot_attr_copy = df[['SEX', 'RAC1P']]
        df.index = pd.MultiIndex.from_frame(prot_attr_copy)

        # standardscaler for age column
        # X_num = df[task_infos['tasks'][1]['num_columns']]
        numeric_transformer = StandardScaler()
        # X_num.loc[:, task_infos['tasks'][1]['num_columns']] = numeric_transformer.fit_transform
        # (X_num[task_infos['tasks'][1]['num_columns']])
        df.loc[:, 'AGEP'] = numeric_transformer.fit_transform(df['AGEP'].values.reshape(-1, 1))

        # turn category columns into category + one hot encoding
        for i in task_infos['tasks'][1]['cat_columns']:
            df.loc[:, i] = df.loc[:, i].astype("category")

        # X_cat = df[task_infos['tasks'][1]['cat_columns']]
        # X_cat = pd.get_dummies(X_cat, columns=task_infos['tasks'][1]['cat_columns'], drop_first=False)

        ohe = make_column_transformer(
            (OneHotEncoder(sparse=False), df.dtypes == 'category'),
            remainder='passthrough', verbose_feature_names_out=False)

        df2 = pd.DataFrame(ohe.fit_transform(df), columns=ohe.get_feature_names_out(), index=df.index)
        cols = list(df2.columns)
        df2_X = df2[cols[:-1]]
        df2_y = df2[['ESR']]
        # df_without_target = pd.concat([X_num, X_cat], axis=1)

        return df2_X, df2_y, sex_col, race_col

    def test_model_spatial(self):
        """
        This method tests sklearn classifiers for the specific task
        """

        metricss = {}

        # preprocessing
        X_train, y_train, _, _ = self.preprocess(self.train_df)

        for clf_name, clfier in self.clf_set.items():

            # initialize fairness metrics dict + specifiy clfier results
            metricss[clf_name] = {}
            # initialize cross validation Kfold results dict
            kfold_metrics = {}
            kfold_metrics["accuracy"] = []

            # cross validation for training state
            for train_indices, val_indices in self.cv_object.split(X_train):
                train_X = X_train.iloc[train_indices]
                val_X = X_train.iloc[val_indices]
                train_y = y_train.iloc[train_indices]
                val_y = y_train.iloc[val_indices]

                clfier.fit(train_X, train_y)
                fold_pred = clfier.predict(val_X)
                print(f"Classifier {clf_name} has in this fold accuracy of {accuracy_score(val_y, fold_pred)}")
                kfold_metrics["accuracy"].append(accuracy_score(val_y, fold_pred))
            print("CV training done!")

            # now apply fitted model to test data
            for t in range(len(self.test_data_paths)):

                test_df = pd.read_csv(self.test_data_paths[t], sep=',', index_col=0)

                # preprocessing
                X_test, y_test, sex_test, race_test = self.preprocess(test_df)

                # because some categories might not be present in test data but are still expected by
                # the classifier after fit check if this is the case, and if yes add these columns to X_test
                cols_to_fill = set(X_train.columns) - set(X_test.columns)
                if cols_to_fill:
                    print("There are columns ot add to the test data: ", cols_to_fill)
                    for c in cols_to_fill:
                        X_test[c] = 0
                X_test = X_test[X_train.columns]

                # get predictions from classifier over test data
                y_pred = clfier.predict(X_test)
                if clf_name in ["AdversarialDebiasing", "ExponentiatedGradientReduction"]:
                    y_pred_p = clfier.predict_proba(X_test)[:, 1]
                else:
                    y_pred_p = clfier.decision_function(X_test)

                # calculate metrics
                metricss[clf_name][self.test_states[t]] = {}
                metricss[clf_name][self.test_states[t]]['train_kfold_accuracy'] = np.mean(kfold_metrics["accuracy"])
                metricss[clf_name][self.test_states[t]]['accuracy'] = accuracy_score(y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['bal_accuracy'] = balanced_accuracy_score(y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['precision'] = precision_score(y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['recall'] = recall_score(y_test, y_pred)
                metricss[clf_name][self.test_states[t]]['tpr_fairlearn'] = true_positive_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_states[t]]['fpr_fairlearn'] = false_positive_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_states[t]]['tnr_fairlearn'] = true_negative_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_states[t]]['fnr_fairlearn'] = false_negative_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_states[t]]['auc'] = roc_auc_score(y_test, y_pred_p)

                print(y_test)
                print(y_pred)

                # fairlearn metrics
                metricss[clf_name][self.test_states[t]]['sex_dpd'] = demographic_parity_difference(
                    y_test,
                    y_pred,
                    sensitive_features=sex_test,
                    method="between_groups")
                metricss[clf_name][self.test_states[t]]['sex_dpr'] = demographic_parity_ratio(
                    y_test,
                    y_pred,
                    sensitive_features=sex_test,
                    method="between_groups")
                metricss[clf_name][self.test_states[t]]['sex_eod'] = equalized_odds_difference(
                    y_test,
                    y_pred,
                    sensitive_features=sex_test,
                    method="between_groups")
                metricss[clf_name][self.test_states[t]]['rac_dpd'] = demographic_parity_difference(
                    y_test,
                    y_pred,
                    sensitive_features=race_test,
                    method="between_groups")
                metricss[clf_name][self.test_states[t]]['rac_dpr'] = demographic_parity_ratio(
                    y_test,
                    y_pred,
                    sensitive_features=race_test,
                    method="between_groups")
                metricss[clf_name][self.test_states[t]]['rac_eod'] = equalized_odds_difference(
                    y_test,
                    y_pred,
                    sensitive_features=race_test,
                    method="between_groups")


                if clf_name in ["AdversarialDebiasing", "ExponentiatedGradientReduction"]:
                    # save ROC curve information for the specific protected attribute in each test state
                    ROC_dict = {}
                    ROC_dict['SEX_fpr'], ROC_dict['SEX_tpr'], ROC_dict['SEX_auc'] = {}, {}, {}
                    ROC_dict['RAC1P_fpr'], ROC_dict['RAC1P_tpr'], ROC_dict['RAC1P_auc'] = {}, {}, {}

                    # ROC w.r.t. SEX
                    indices_male = np.where(sex_test == 1)[0]
                    indices_female = np.where(sex_test == 2)[0]

                    ROC_dict['SEX_fpr']['male'], ROC_dict['SEX_tpr']['male'], _ = \
                        roc_curve(y_test.iloc[indices_male], y_pred_p[indices_male], pos_label=1)
                    ROC_dict['SEX_fpr']['female'], ROC_dict['SEX_tpr']['female'], _ = \
                        roc_curve(y_test.iloc[indices_female], y_pred_p[indices_female], pos_label=1)
                    ROC_dict['SEX_auc']['male'] = roc_auc_score(y_test.iloc[indices_male], y_pred_p[indices_male])
                    ROC_dict['SEX_auc']['female'] = roc_auc_score(y_test.iloc[indices_female], y_pred_p[indices_female])

                    # ROC w.r.t. RAC1P
                    indices_white = np.where(race_test == 1)[0]
                    indices_black = np.where(race_test == 2)[0]
                    indices_other = np.where(race_test == 3)[0]

                    ROC_dict['RAC1P_fpr']['white'], ROC_dict['RAC1P_tpr']['white'], _ = \
                        roc_curve(y_test.iloc[indices_white], y_pred_p[indices_white], pos_label=1)
                    ROC_dict['RAC1P_fpr']['black'], ROC_dict['RAC1P_tpr']['black'], _ = \
                        roc_curve(y_test.iloc[indices_black], y_pred_p[indices_black], pos_label=1)
                    ROC_dict['RAC1P_fpr']['other'], ROC_dict['RAC1P_tpr']['other'], _ = \
                        roc_curve(y_test.iloc[indices_other], y_pred_p[indices_other], pos_label=1)
                    ROC_dict['RAC1P_auc']['white'] = roc_auc_score(y_test.iloc[indices_white], y_pred_p[indices_white])
                    ROC_dict['RAC1P_auc']['black'] = roc_auc_score(y_test.iloc[indices_black], y_pred_p[indices_black])
                    ROC_dict['RAC1P_auc']['other'] = roc_auc_score(y_test.iloc[indices_other], y_pred_p[indices_other])

                    # create results folder for the training state if not present already
                    if not os.path.isdir(os.path.join(rdir, self.task, str(self.spatial_year),
                                                      'aif360', self.train_state)):
                        os.makedirs(os.path.join(rdir, self.task, str(self.spatial_year), 'aif360', self.train_state))
                    # save to pickle file
                    with open(os.path.join(rdir, self.task, str(self.spatial_year),
                                           'aif360', self.train_state,
                                           f'spatial_{self.train_state}_test_{self.test_states[t]}_{clf_name}.pickle'),
                              'wb') as handle:
                        pickle.dump(ROC_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                         
            # prova
            if self.clf_name =='AdversarialDebiasing':
                clfier.sess_.close()

        return metricss

    def test_model_temporal(self):
        """
        This method tests sklearn classifiers for the specific task in the temporal context
        e.g. given a training state CA in year 2014 we train a classifier on CA 2014 and
        test on CA 2015, CA 2016 ...
        """
        # initalize metrics dict
        metricss = {}
        # preprocessing
        X_train, y_train, _, _ = self.preprocess(self.train_df)


        # fit on training data (one state)
        kfold_metrics = {}
        kfold_metrics["accuracy"] = []

        for clf_name, clfier in self.clf_set.items():

            # initialize fairness metrics dict + specifiy clfier results
            metricss[clf_name] = {}

            for train_indices, val_indices in self.cv_object.split(X_train):
                train_X = X_train.iloc[train_indices]
                val_X = X_train.iloc[val_indices]
                train_y = y_train.iloc[train_indices]
                val_y = y_train.iloc[val_indices]

                clfier.fit(train_X, train_y)
                fold_pred = clfier.predict(val_X)
                print(f"Classifier {clf_name} has in this fold accuracy of {accuracy_score(val_y, fold_pred)}")
                kfold_metrics["accuracy"].append(accuracy_score(val_y, fold_pred))
            print("CV training done!")

            # loop over all test states, get metrics and save results in .tsv file
            for t in range(len(self.test_data_paths)):
                print(f"YEAR TEST: {self.test_years[t]}")
                # metrics init
                metricss[clf_name][self.test_years[t]] = {}

                test_df = pd.read_csv(self.test_data_paths[t], sep=',', index_col=0)

                # create testing data
                X_test, y_test, sex_test, race_test = self.preprocess(test_df)
                # because some categories might not be present in test data but are still expected by
                # the classifier after fit check if this is the case, and if yes add these columns to X_test
                cols_to_fill = set(X_train.columns) - set(X_test.columns)
                print(cols_to_fill)
                if cols_to_fill:
                    for c in cols_to_fill:
                        X_test[c] = 0
                X_test = X_test[X_train.columns]

                # get predictions from classifier
                y_pred = clfier.predict(X_test)
                if clf_name in ["AdversarialDebiasing", "ExponentiatedGradientReduction"]:
                    y_pred_p = clfier.predict_proba(X_test)[:, 1]
                else:
                    y_pred_p = clfier.decision_function(X_test)

                # calculate metrics
                metricss[clf_name][self.test_years[t]] = {}
                metricss[clf_name][self.test_years[t]]['train_kfold_accuracy'] = np.mean(kfold_metrics["accuracy"])
                metricss[clf_name][self.test_years[t]]['accuracy'] = accuracy_score(y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['bal_accuracy'] = balanced_accuracy_score(y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['precision'] = precision_score(y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['recall'] = recall_score(y_test, y_pred)
                metricss[clf_name][self.test_years[t]]['tpr_fairlearn'] = true_positive_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_years[t]]['fpr_fairlearn'] = false_positive_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_years[t]]['tnr_fairlearn'] = true_negative_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_years[t]]['fnr_fairlearn'] = false_negative_rate(y_test.iloc[:,0], y_pred, pos_label=1)
                metricss[clf_name][self.test_years[t]]['auc'] = roc_auc_score(y_test, y_pred_p)

                # fairlearn
                metricss[clf_name][self.test_years[t]]['sex_dpd'] = demographic_parity_difference(y_test,
                                                                                                  y_pred,
                                                                                                  sensitive_features=sex_test,
                                                                                                  method="between_groups")
                metricss[clf_name][self.test_years[t]]['sex_dpr'] = demographic_parity_ratio(y_test,
                                                                                             y_pred,
                                                                                             sensitive_features=sex_test,
                                                                                             method="between_groups")
                metricss[clf_name][self.test_years[t]]['sex_eod'] = equalized_odds_difference(y_test,
                                                                                              y_pred,
                                                                                              sensitive_features=sex_test,
                                                                                              method="between_groups")
                metricss[clf_name][self.test_years[t]]['rac_dpd'] = demographic_parity_difference(y_test,
                                                                                                  y_pred,
                                                                                                  sensitive_features=race_test,
                                                                                                  method="between_groups")
                metricss[clf_name][self.test_years[t]]['rac_dpr'] = demographic_parity_ratio(y_test,
                                                                                             y_pred,
                                                                                             sensitive_features=race_test,
                                                                                             method="between_groups")
                metricss[clf_name][self.test_years[t]]['rac_eod'] = equalized_odds_difference(y_test,
                                                                                              y_pred,
                                                                                              sensitive_features=race_test,
                                                                                              method="between_groups")

                if clf_name in ["AdversarialDebiasing", "ExponentiatedGradientReduction"]:
                    # save ROC curve information for the specific protected attribute in each test state
                    ROC_dict = {}
                    ROC_dict['SEX_fpr'], ROC_dict['SEX_tpr'], ROC_dict['SEX_auc'] = {}, {}, {}
                    ROC_dict['RAC1P_fpr'], ROC_dict['RAC1P_tpr'], ROC_dict['RAC1P_auc'] = {}, {}, {}

                    # ROC w.r.t. SEX
                    indices_male = np.where(sex_test == 1)[0]
                    indices_female = np.where(sex_test == 2)[0]

                    ROC_dict['SEX_fpr']['male'], ROC_dict['SEX_tpr']['male'], _ = \
                        roc_curve(y_test.iloc[indices_male], y_pred_p[indices_male], pos_label=1)
                    ROC_dict['SEX_fpr']['female'], ROC_dict['SEX_tpr']['female'], _ = \
                        roc_curve(y_test.iloc[indices_female], y_pred_p[indices_female], pos_label=1)
                    ROC_dict['SEX_auc']['male'] = roc_auc_score(y_test.iloc[indices_male], y_pred_p[indices_male])
                    ROC_dict['SEX_auc']['female'] = roc_auc_score(y_test.iloc[indices_female], y_pred_p[indices_female])

                    # ROC w.r.t. RAC1P
                    indices_white = np.where(race_test == 1)[0]
                    indices_black = np.where(race_test == 2)[0]
                    indices_other = np.where(race_test == 3)[0]

                    ROC_dict['RAC1P_fpr']['white'], ROC_dict['RAC1P_tpr']['white'], _ = \
                        roc_curve(y_test.iloc[indices_white], y_pred_p[indices_white], pos_label=1)
                    ROC_dict['RAC1P_fpr']['black'], ROC_dict['RAC1P_tpr']['black'], _ = \
                        roc_curve(y_test.iloc[indices_black], y_pred_p[indices_black], pos_label=1)
                    ROC_dict['RAC1P_fpr']['other'], ROC_dict['RAC1P_tpr']['other'], _ = \
                        roc_curve(y_test.iloc[indices_other], y_pred_p[indices_other], pos_label=1)
                    ROC_dict['RAC1P_auc']['white'] = roc_auc_score(y_test.iloc[indices_white], y_pred_p[indices_white])
                    ROC_dict['RAC1P_auc']['black'] = roc_auc_score(y_test.iloc[indices_black], y_pred_p[indices_black])
                    ROC_dict['RAC1P_auc']['other'] = roc_auc_score(y_test.iloc[indices_other], y_pred_p[indices_other])

                    # create results folder for the training state if not present already
                    if not os.path.isdir(
                            os.path.join(rdir, self.task, str(self.train_year), 'aif360', self.train_state)):
                        os.makedirs(os.path.join(rdir, self.task, str(self.train_year), 'aif360', self.train_state))
                    # save to pickle file
                    with open(os.path.join(rdir, self.task, str(self.train_year), 'aif360', self.train_state,
                                           f'temporal_{self.train_state}_test_{self.test_states[t]}'
                                           f'_{clf_name}.pickle'),
                              'wb') as handle:
                        pickle.dump(ROC_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    parser = argparse.ArgumentParser(description='Run SKLEARN models over the data')
    parser.add_argument('--mode', type=str, help='Which context do you want to explore (spatial or temporal)',
                        required=True)
    parser.add_argument('--train', type=str,
                        help='Which train state do you want to run',
                        required=True)
    parser.add_argument('--year', type=int,
                        help='Which year do you want to focus on',
                        required=True)
    args = parser.parse_args()

    # directory management
    wdir = os.getcwd()
    udir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis", "utils")
    ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data", "new_clustering") #clustering
    rdir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data", "new_results_sampling1") #results_sampling or new_results_sampling
    print(f"working directory: {wdir}")

    # json task specs
    json_file_path = os.path.join(udir, 'tasks_metadata.json')
    with open(json_file_path, 'r') as j:
        task_infos = json.loads(j.read())

    task = "ACSEmployment"

    # define regex pattern to extract names of all test states
    test_state_pattern = re.compile(r"_([^_]+)_")


    # if mode is spatial:
    if args.mode == "spatial":

        for select_year in [args.year]:
            data_file_paths = glob.glob(
                os.path.join(ddir, str(select_year), '1-Year') + f'/*/{str(select_year)}_*_{task}.csv', recursive=True)
            data_file_paths.sort()


            print(f"CONTEXT: {args.mode} - YEAR: {select_year} - TRAIN DATA STATE: {args.train}")

            # load train data
            train_df = pd.read_csv(os.path.join(ddir, str(select_year), '1-Year', args.train,
                                                f'{str(select_year)}_{args.train}_{task}.csv'),
                                   sep=',', index_col=0)
            # load test data
            test_states = []
            test_data_file_paths = [f for f in data_file_paths if args.train not in f]

            for p in range(len(test_data_file_paths)):
                test_path = Path(test_data_file_paths[p])
                test_states.append(test_state_pattern.findall(test_path.name)[-1])

            M = Model(train_df,
                      test_data_file_paths,
                      task,
                      task_infos['tasks'][1]['columns'],
                      task_infos['tasks'][1]['cat_columns'],
                      task_infos['tasks'][1]['num_columns'],
                      task_infos["tasks"][task_infos["task_col_map"][task]]["target"],
                      spatial_year=select_year,
                      train_state=args.train,
                      test_states=test_states)
            metrics = M.test_model_spatial()

    # if mode is temporal:
    if args.mode == "temporal":

        for select_year in [args.year]:
            # train year paths
            data_file_paths = glob.glob(os.path.join(ddir, str(select_year), '1-Year') +
                                        f'/*/{str(select_year)}*_{task}.csv', recursive=True)
            data_file_paths.sort()


            print(f"CONTEXT: {args.mode} - YEAR: {select_year} - TRAIN DATA STATE: {args.train}")
            train_df = pd.read_csv(os.path.join(ddir, str(select_year), '1-Year', args.train,
                                                f'{str(select_year)}_{args.train}_{task}.csv'),
                                   sep=',', index_col=0)

            test_dfs = []
            test_states = []
            years_to_test = [item for item in task_infos["years"] if item not in [select_year]]
            test_data_file_paths = [
                os.path.join(ddir, str(y), '1-Year', args.train, f'{str(y)}_{args.train}_{task}.csv')
                for y in years_to_test]

            print(test_states)
            print(test_data_file_paths)
            
            for p in range(len(test_data_file_paths)):
                test_path = Path(test_data_file_paths[p])
                test_states.append(test_state_pattern.findall(test_path.name)[-1])

            M = Model(train_df,
                      test_data_file_paths,
                      task,
                      task_infos['tasks'][1]['columns'],
                      task_infos['tasks'][1]['cat_columns'],
                      task_infos['tasks'][1]['num_columns'],
                      task_infos["tasks"][task_infos["task_col_map"][task]]["target"],
                      temporal_train_year=select_year,
                      temporal_test_years=years_to_test,
                      train_state=args.train,
                      test_states=test_states)
            metrics = M.test_model_temporal()