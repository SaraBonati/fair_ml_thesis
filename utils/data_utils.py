import argparse
import json
import os

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSHealthInsurance

wdir = os.getcwd()
ddir = os.path.join(os.path.split(os.path.split(wdir)[0])[0], "fair_ml_thesis_data", "rawdata")
# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())


def download_data(survey_year: int, survey='person', horizon='1-Year'):
    """
    Download data to folder in local system
    :param survey_year: year to be downloaded
    :param survey: string indicating if person or household data (defaults to 'person')
    :param horizon: string indicating if 1 or 5 year horizon data (defaults to '1-Year')
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

    if task == "ACSIncome":
        features, label, group = ACSIncome.df_to_numpy(raw_df)
        c = np.append(features, np.expand_dims(label, axis=1), axis=1)
        data = pd.DataFrame(data=c, columns=task_infos['tasks'][0]['columns'])
    elif task == "ACSEmployment":
        features, label, group = ACSEmployment.df_to_numpy(raw_df)
        c = np.append(features, np.expand_dims(label, axis=1), axis=1)
        data = pd.DataFrame(data=c, columns=task_infos['tasks'][1]['columns'])
    elif task == "ACSHealthInsurance":
        features, label, group = ACSHealthInsurance.df_to_numpy(raw_df)
        c = np.append(features, np.expand_dims(label, axis=1), axis=1)
        data = pd.DataFrame(data=c, columns=task_infos['tasks'][2]['columns'])

    data.to_csv(os.path.join(ddir, str(year), "1-Year", f"{str(year)}_{state}_{task}.csv"), sep=',')


if __name__ == "__main__":

    # --------Specify arguments--------------------------
    parser = argparse.ArgumentParser(
        description="Data Utils (Project: Fair ML Master Thesis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--mode", type = str, help="indicate what method to run (download or rawdata_to_task)"
    )
    args = parser.parse_args()

    if args.mode == 'download':
        download_data()
    elif args.mode == 'rawdata_to_task':
        for t in task_infos["task_names"]:
            for y in task_infos["years"]:
                for s in task_infos["states"]:
                    print(t, y, s)
                    rawdata_to_task(t, s, y)
