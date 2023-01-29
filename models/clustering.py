import argparse
import json
import os.path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import utils
import prince
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes


class Cluster:

    def __init__(self, data_paths, task: str, target_col: str, features):
        """

        :param data_paths:
        :param task: name of task
        :param target_col: categorical columns for the task
        :param features: all columns other than labels for the task
        """

        self.task = task
        self.cluster_set = {}
        self.data_paths = data_paths
        self.target_col = target_col
        self.features = features

    @staticmethod
    def preprocess_kmeans(data):
        """
        Applies FAMD (factor analysis of mixed data) to data before clustering and scales numerical attributes
        :param data:
        :return:
        """
        # first standard scaler
        standard_scaler = StandardScaler()
        data[["AGEP"]] = standard_scaler.fit_transform(data[["AGEP"]])

        data['RAC1P'] = pd.to_numeric(data['RAC1P'])
        # recode RAC1P values for all tasks
        data.loc[data['RAC1P'] > 2, 'RAC1P'] = 3

        # then categorical variables get turned into pandas category
        for i in ["SCHL", "MAR", "CIT", "SEX", "RAC1P", "STATE"]:
            data[i] = data[i].astype("object")

        final_features = ["AGEP", "SCHL", "MAR", "CIT", "SEX", "RAC1P", "STATE"]
        X = data[final_features]
        X.replace([np.inf, -np.inf,np.nan], 999, inplace=True)


        # now dimensionality reduction
        transformed_data = prince.FAMD(n_components=3,
                                       n_iter=3,
                                       copy=True,
                                       check_input=False,
                                       engine='auto',
                                       random_state=42).fit_transform(X)

        return transformed_data

    def apply_clustering_kmodes(self, task: str, year: int):
        """
        This function applies clustering kmodes, meaning we turn the columns of interest
        for each state and year into pandas categories and proceed to use a clustering method
        for categorical variables
        :return:
        """
        all_dfs = []
        for file_path in self.data_paths:
            df = pd.read_csv(file_path, sep=",")
            df['STATE'] = os.path.split(file_path)[1][5:7]
            all_dfs.append(df)

        data = pd.concat(all_dfs, ignore_index=True)
        X = data[["AGEP", "SCHL", "MAR", "CIT", "SEX", "RAC1P", "STATE"]]
        # first standard scaler
        standard_scaler = StandardScaler()
        X[["AGEP"]] = standard_scaler.fit_transform(X[["AGEP"]])
        true_labels = X[["STATE"]]

        cost = []
        K = range(3, 7)
        for num_clusters in list(K):
            kmode = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
            clusters = kmode.fit_predict(X, categorical=[1, 2, 3, 4, 5, 6])
            cost.append(kmode.cost_)
            # add final prediction to dataset
            X['PREDICTION'] = clusters
            X.to_csv(os.path.join(tdir, str(year), task, f"{task}_{str(year)}_kmodes.csv"))

        plt.plot(K, cost, 'bx-')
        plt.xlabel('No. of clusters')
        plt.ylabel('Cost')
        plt.title('Elbow Method For Optimal k')
        if not os.path.exists(os.path.join(tdir, str(year), task)):
            os.makedirs(os.path.join(tdir, str(year), task))
            plt.savefig(os.path.join(tdir, str(year), task, f"{task}_{str(year)}_elbow_kmodes.png"),
                        format='png', dpi=300)

    def apply_clustering_kmeans(self, task: str, year: int):
        """
        This function applies clustering using kmeans (specifically minibatch k means). Since
        k means needs numerical data we first use FAMD to reduce data dimensionality and obtain
        numerical coordinates in a 3d space for each data point
        :return:
        """
        all_dfs = []

        for file_path in self.data_paths[:5]:
            df = pd.read_csv(file_path, sep=",")
            df['STATE'] = os.path.split(file_path)[1][5:7]
            df = self.preprocess_kmeans(df)
            all_dfs.append(df)

        X = pd.concat(all_dfs, ignore_index=True)

        # kmeans
        inertias = []
        K = range(3, 7)
        for num_clusters in list(K):
            kmeans = MiniBatchKMeans(
                init="k-means++",
                n_clusters=num_clusters,
                batch_size=32,
                n_init=10,
                max_no_improvement=10,
                verbose=1,
            ).fit(X)
            labels = kmeans.labels_
            inertias.append(kmeans.inertia_)

        plt.plot(K, inertias, 'bx-')
        plt.xlabel('No. of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        if not os.path.exists(os.path.join(tdir, str(year), task)):
            os.makedirs(os.path.join(tdir, str(year), task))
        plt.savefig(os.path.join(tdir, str(year), task, f"{task}_{str(year)}_elbow_kmeans.png"),
                    format='png',
                    dpi=300)

        # results = pd.DataFrame.from_dict({
        #     "homogeneity": homogeneity_score(true_labels, labels),
        #     "completeness": completeness_score(true_labels, labels),
        #     "v-measure": v_measure_score(true_labels, labels),
        #     "adjusted_rand_index": adjusted_rand_score(true_labels, labels),
        #     "adjusted_mutual_information": adjusted_mutual_info_score(true_labels, labels),
        #     "silhouette_score": silhouette_score(X, labels)
        # },
        #     orient='index', columns=['value'])
        # results.to_csv("k_means_", sep=",")

    @staticmethod
    def plot_clusters(n_components: int, data, labels, method: str):
        """
        Plots the clusters
        :return:
        """
        if n_components == 2:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1)
            ax.scatter(data[0], data[1], c=labels.astype(float), edgecolor="k")
            # customize plot
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"{method} clusters")

        elif n_components == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection="3d", elev=48, azim=134)
            ax.scatter(data[:, 3], data[:, 0], data[:, 2], c=labels.astype(float), edgecolor="k")
            # customize plot
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_zlabel("Principal Component 3")
            ax.set_title(f"{method} clusters")


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
    tdir = os.path.join(os.path.split(os.path.split(wdir)[0])[0], "fair_ml_thesis_data", "clustering")
    rdir = os.path.join(os.path.split(os.path.split(wdir)[0])[0], "fair_ml_thesis_data", "results")

    # json task specs
    json_file_path = os.path.join(udir, 'tasks_metadata.json')
    with open(json_file_path, 'r') as j:
        task_infos = json.loads(j.read())

    # get data paths
    data_file_paths = glob.glob(
        os.path.join(ddir, str(args.year), '1-Year') + f'/{str(args.year)}_*_{args.task}.csv')

    # get feature names (exclude target variable)
    features = task_infos["tasks"][task_infos["task_col_map"][args.task]]["columns"].remove(
        task_infos["tasks"][task_infos["task_col_map"][args.task]]["target"])

    # initialize cluster object
    C = Cluster(data_file_paths,
                args.task,
                task_infos["tasks"][task_infos["task_col_map"][args.task]]["target"],
                features
                )

    if args.method == "kmeans":
        C.apply_clustering_kmeans(args.task, args.year)
    elif args.method == "kmodes":
        C.apply_clustering_kmodes(args.task, args.year)
