import argparse
import json
import os.path
import glob

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes


class Cluster:

    def __init__(self, data_paths, task, target_col, features):
        """

        :param data_paths:
        :param task: name of task
        :param target_col: categorical columns for the task
        :param features: all columns other than labels for the task
        """

        self.cluster_set = {}
        self.data_paths = data_paths
        self.target_col = target_col

    @staticmethod
    def preprocess(data):
        """
        Applies PCA to data before clustering and scales numericla attributes
        :param data:
        :return:
        """
        # first standard scaler
        standard_scaler = StandardScaler()
        data.loc[:, "AGEP"] = standard_scaler.fit_transform(data.loc[:, "AGEP"])
        # now pca
        transformed_data = PCA(n_components=2).fit_transform(data)
        return transformed_data

    def apply_clustering(self):
        """
        Applies clustering
        :return:
        """
        all_dfs = []

        for file_path in self.data_paths:
            df = pd.read_csv(file_path, sep=",")
            df = self.preprocess(df)
            df['STATE'] = os.path.split(file_path)[1][5:7]
            df['YEAR'] = os.path.split(file_path)[1][0:4]
            all_dfs.append(df)

        data = pd.concat(all_dfs, ignore_index=True)
        X = data[:]
        true_labels = data[["STATE", "YEAR", self.target_col]]

        # kmeans
        kmeans = MiniBatchKMeans(
            init="k-means++",
            n_clusters=3,
            batch_size=32,
            n_init=10,
            max_no_improvement=10,
            verbose=0,
        ).fit(X)
        labels = kmeans.labels_

        # kmodes
        km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)
        clusters = km.fit_predict(X)

        results = pd.DataFrame.from_dict({
            "homogeneity": homogeneity_score(true_labels, labels),
            "completeness": completeness_score(true_labels, labels),
            "v-measure": v_measure_score(true_labels, labels),
            "adjusted_rand_index": adjusted_rand_score(true_labels, labels),
            "adjusted_mutual_information": adjusted_mutual_info_score(true_labels, labels),
            "silhouette_score": silhouette_score(X, labels)
        },
            orient='index', columns=['value'])
        results.to_csv("k_means_", sep=",")

    @staticmethod
    def plot_clusters():
        """
        Plots the clusters
        :return:
        """
        print("TODO")


#########################
# SCRIPT
#########################
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Run clustering over the data')
    parser.add_argument('--method', type=str, help='Which clustering algorithm to use',
                        required=True)
    parser.add_argument('--task', type=str,
                        help='Which classification task do you want to focus on',
                        required=True)
    parser.add_argument('--year', type=int,
                        help='Which year do you want to focus on',
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

    # get data paths
    data_file_paths = glob.glob(
        os.path.join(ddir, str(args.year), '1-Year') + f'/{str(args.year)}_*_{args.task}.csv')

    C = Cluster(data_file_paths,
                args.task,
                task_infos["tasks"][task_infos["task_col_map"][args.task]]["target"],
                task_infos["tasks"][task_infos["task_col_map"][args.task]]["columns"].remove(
                    task_infos["tasks"][task_infos["task_col_map"][args.task]]["target"]
                ))
    C.apply_clustering()
