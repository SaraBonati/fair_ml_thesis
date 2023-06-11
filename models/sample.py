import numpy as np
import pandas as pd
from collections import Counter
import os
import glob
import json
import operator
import argparse

class Sample:
    def __init__(self):
        """

        :param adir:
        :param udir:
        :param rdir:
        """
        self.task = 'ACSEmployment'
        self.random_state = 42

        # json task specs
        json_file_path = os.path.join(udir, 'clustering_infos.json')
        with open(json_file_path, 'r') as j:
            clustering_infos = json.loads(j.read())
        self.bea_infos = clustering_infos['BEA_clustering']

        # load imbalance numbers if they have been calculated already
        imbalance_file_path = os.path.join(udir, 'imbalance_infos.json')
        with open(imbalance_file_path, 'r') as j:
            self.imb_infos = json.loads(j.read())


    def calculate_imbalance_table(self):
        """

        :return:
        """
        imbalance_dict = {}
        for year in task_infos['years']:

            imbalance_dict[str(year)]={}

            for state in task_infos['states']:
                state_data = pd.read_csv(os.path.join(ddir, str(args.year), '1-Year', f'{args.year}_{state}_{task}.csv'),
                                         sep=',')

                # turn everything into int
                for c in task_infos['tasks'][1]['columns']:
                    state_data.loc[:, c] = state_data.loc[:, c].astype(int)

                # filter for age
                state_data = state_data[state_data['AGEP'].between(16, 90)]
                state_data.reset_index(drop=True, inplace=True)

                # record race column
                state_data.loc[state_data['RAC1P'] > 2, 'RAC1P'] = 3

                count_dict = state_data.groupby(by=['ESR', 'SEX', 'RAC1P']).size().to_frame('num').reset_index()
                count_dict.set_index(['ESR', 'SEX', 'RAC1P'], inplace=True)
                imbalance_dict[str(year)][state] = {str(k):v['num'] for k,v in count_dict.to_dict('index').items()}

        # save imbalance dict
        with open(os.path.join(udir,f"imbalance_infos.json"), "w") as fp:
            json.dump(imbalance_dict, fp)


    def get_cluster_data(self, train_state: str, year: int):
        """

        :param train_state:
        :param year:
        :return:
        """

        # get specific year - trian state combo imbalance numbers
        imbalance_infos = self.imb_infos[str(year)][train_state]
        print(imbalance_infos)
        # get states that belong in same cluster as train state for provided year
        cluster_states = [i for k, v in self.bea_infos[str(year)].items()
                          for i in v
                          if (train_state in v) and (i != train_state)
                          ]
        print(cluster_states)
        max_key = max(imbalance_infos.items(), key=lambda k: k[1])[0]
        max_value = max(imbalance_infos.items(), key=lambda k: k[1])[1]
        sample_dict = {}

        # create pool of data from similar states present in cluster
        data_dfs = []
        for c in cluster_states:
            df = pd.read_csv(os.path.join(ddir, str(args.year), '1-Year', f'{args.year}_{c}_{task}.csv'),
                                         sep=',')

            # turn everything into int
            for c in task_infos['tasks'][1]['columns']:
                df.loc[:, c] = df.loc[:, c].astype(int)

            # filter for age
            df = df[df['AGEP'].between(16, 90)]
            df.reset_index(drop=True, inplace=True)

            # record race column
            df.loc[df['RAC1P'] > 2, 'RAC1P'] = 3

            data_dfs.append(df)
        data_df = pd.concat(data_dfs, ignore_index=True)

        dfs = []
        #for state in cluster_states:
            # year_data = self.data[(self.data['YEAR'] == year) & (self.data['STATE'].isin(cluster_states))]
        for k, v in imbalance_infos.items():
            if k != max_key:
                df = data_df.query(f'(ESR == {eval(k)[0]}) &'
                                     f' (SEX == {eval(k)[1]}) &'
                                     f' (RAC1P == {eval(k)[2]})')

                print(f"key {k} has originally {v}, needs {(max_value - v)}, selected df has {len(df)}")
                if len(df)<(max_value-v):
                    a = df.sample(n=len(df), random_state=self.random_state)
                else:
                    a = df.sample(n=(max_value - v), random_state=self.random_state)

                print(k, len(a))

                dfs.append(a)

        new_data_samples = pd.concat(dfs, ignore_index=True)

        state_df = pd.read_csv(os.path.join(ddir, str(args.year), '1-Year', f'{args.year}_{args.state}_{task}.csv'),
                                         sep=',')

        # turn everything into int
        for c in task_infos['tasks'][1]['columns']:
            state_df.loc[:, c] = state_df.loc[:, c].astype(int)

        # filter for age
        state_df = state_df[state_df['AGEP'].between(16, 90)]
        state_df.reset_index(drop=True, inplace=True)

        # record race column
        state_df.loc[state_df['RAC1P'] > 2, 'RAC1P'] = 3

        final = pd.concat([state_df,new_data_samples], ignore_index=True)
        print(final.groupby(by=['ESR', 'SEX', 'RAC1P']).size().to_frame('num').reset_index())

        if not os.path.isdir(os.path.join(cdir, str(args.year), '1-Year', args.state)):
            os.makedirs(os.path.join(cdir, str(args.year), '1-Year', args.state))
        final.to_csv(os.path.join(cdir, str(args.year), '1-Year', args.state,
                                  f'{args.year}_{args.state}_{task}.csv'),
                                sep=',', index=False)

#########################
# SCRIPT
#########################
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Run clustering over the data')
    parser.add_argument('--state', type=str, help='Which train state to use',
                        required=True)
    parser.add_argument('--year', type=int,
                        help='Which year do you want to focus on',
                        required=True)
    args = parser.parse_args()

    # directory management
    wdir = os.getcwd()
    udir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis", "utils")
    ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data", "rawdata")
    cdir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data", "clustering")

    json_file_path = os.path.join(udir, 'tasks_metadata.json')
    with open(json_file_path, 'r') as j:
        task_infos = json.loads(j.read())

    task='ACSEmployment'
    S = Sample()

    if not os.path.exists(os.path.join(udir,'imbalance_infos.json')):
        S.calculate_imbalance_table()

    S.get_cluster_data(args.state, args.year)



