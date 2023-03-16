import argparse
import json
import os
import glob
import pickle
import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSHealthInsurance

wdir = os.getcwd()
ddir = os.path.join(os.path.split(os.path.split(wdir)[0])[0], "fair_ml_thesis_data", "rawdata")
# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())

#############################################
# Functions related to download of data
#############################################
def download_data(survey_year: int):
    """
    Download data to folder in local system
    :param survey_year: year to be downloaded
    :return:
    """

    data_source = ACSDataSource(survey_year=int(survey_year), horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=task_infos["states_list"], download=True)


def rawdata_to_task(task: str, state: str, year: int):
    """
    Transform the rawdata for a specific state and year into a task dataframe
    :param task: name of classification task
    :param state:
    :param year:
    :return:
    """

    assert os.path.exists(os.path.join(ddir, str(year), "1-Year", f"{str(year)}_{state}.csv")), \
        f"Rawdata file for {str(year)} {state} does not exist!"

    raw_df = pd.read_csv(os.path.join(ddir, str(year), "1-Year", f"{str(year)}_{state}.csv"), sep=',')
    print(raw_df['PINCP'].dtype)
    print(raw_df['WKHP'].dtype)
    print(raw_df['PWGTP'].dtype)
    if not (raw_df['PINCP'].dtype == np.float64 or raw_df['PINCP'].dtype == np.int64):
        raw_df['PINCP'] = raw_df['PINCP'].str.lstrip('0')
        raw_df['PINCP'] = pd.to_numeric(raw_df['PINCP'], errors='coerce')
    if not (raw_df['WKHP'].dtype == np.float64 or raw_df['WKHP'].dtype == np.int64):
        raw_df['WKHP'] = pd.to_numeric(raw_df['WKHP'], errors='coerce')
    if not (raw_df['PWGTP'].dtype == np.float64 or raw_df['PWGTP'].dtype == np.int64):
        raw_df['PWGTP'] = pd.to_numeric(raw_df['PWGTP'], errors='coerce')

    if task == "ACSIncome":
        features, label, _ = ACSIncome.df_to_numpy(raw_df)
        c = np.append(features, np.expand_dims(label, axis=1), axis=1)
        data = pd.DataFrame(data=c, columns=task_infos['tasks'][0]['columns'])
    elif task == "ACSEmployment":
        features, label, _ = ACSEmployment.df_to_numpy(raw_df)
        c = np.append(features, np.expand_dims(label, axis=1), axis=1)
        data = pd.DataFrame(data=c, columns=task_infos['tasks'][1]['columns'])
    elif task == "ACSHealthInsurance":
        features, label, _ = ACSHealthInsurance.df_to_numpy(raw_df)
        c = np.append(features, np.expand_dims(label, axis=1), axis=1)
        data = pd.DataFrame(data=c, columns=task_infos['tasks'][2]['columns'])

    if not os.path.exists(os.path.join(ddir, str(year), "1-Year")):
        os.makedirs(os.path.join(ddir, str(year), "1-Year"))
    data.to_csv(os.path.join(ddir, str(year), "1-Year", f"{str(year)}_{state}_{task}.csv"), sep=',', index=False)


def check_missing_values_presence(task: str):
    """
    Given a task, this function checks the presence of missing values for each year and state.
    This can serve as an indication of whether to impute missing values or drop rows
    :param task:
    :return:
    """
    for year in task_infos["years"]:
        data_files = glob.glob(os.path.join(ddir, str(year), '1-Year') + f'/*{task}.csv')
        for d in data_files:
            df = pd.read_csv(d, sep=',')
            for c in list(df.columns):
                print(df[c].isnull().sum())


def load_classifier_results_data(data_path: str, model_type: str):
    """
    Loads the classifier data for results visualization, changing the index column to be the corresponding state
    :param data_path: (str)
    :param data_path: (str)
    :return:
    """
    print("TODO")


if __name__ == "__main__":

    # --------Specify arguments--------------------------
    parser = argparse.ArgumentParser(
        description="Data Utils (Project: Fair ML Master Thesis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--mode", type=str, help="indicate what method to run (download or rawdata_to_task)"
    )
    args = parser.parse_args()

    if args.mode == 'download':
        download_data()
    elif args.mode == 'rawdata_to_task':
        for t in ["ACSHealthInsurance"]:
            for y in task_infos["years"]:
                for s in task_infos["states"]:
                    print(t, y, s)
                    rawdata_to_task(t, s, y)
    elif args.mode == 'check_missings':
        for task in task_infos["task_names"]:
            check_missing_values_presence(task)
