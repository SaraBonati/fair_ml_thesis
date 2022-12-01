import pandas as pd
import os
import json
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

wdir = os.getcwd()
ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data")
# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, "utils", 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())


def make_mapplot(df, metric: str, title: str, context: str, colors: str = "Viridis_r"):
    """
    Creates a choropleth map plot that is animated showing yearly progression
    of specified metric
    :param df: dataframe to use in plot
    :param metric: name of variable to display in animated map plot
    :param title: title of plot
    :param context: spatial or temporal
    :param colors: name of colormap to use in plot
    :return:
    """
    if context == "Temporal":
        fig = px.choropleth(df,
                            locations='state_code',
                            animation_frame="year",
                            color=metric,
                            color_continuous_scale=colors,
                            locationmode='USA-states',
                            scope="usa",
                            height=600)
    elif context == "Spatial":
        fig = px.choropleth(df,
                            locations='state',
                            color=metric,
                            color_continuous_scale=colors,
                            locationmode='USA-states',
                            scope="usa",
                            height=600)

    return fig


def make_demographic_plots(df, target_name: str):
    """
    Returns histograms and pie charts of demographic variables
    for a state - year specific dataframe
    :param df: dataframe
    :param target_name: name of target variable column
    :return:
    """

    fig1 = px.histogram(df, x="RAC1P", y=target_name, facet_row=target_name,
                        color='SEX',
                        color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                        barmode='group',
                        histfunc='count',
                        width=800,
                        height=800)

    fig2 = px.box(df, x="RAC1P", y="AGEP", facet_row=target_name,
                  color='SEX',
                  color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                  boxmode='group',
                  width=800,
                  height=800)

    # accuracy
    fig3 = make_subplots(rows=len(df["SCHL"].unique()), cols=1,
                        shared_yaxes=True,
                        vertical_spacing=0.02)

    for p in range(len(result_paths)):
        df = pd.read_csv(result_paths[p], header=0, sep=',')
        df.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        fig.add_trace(go.Bar(name=task_infos['classifier_order'][p],
                             x=df['state'],
                             y=df['accuracy']),
                      row=p + 1,
                      col=1)
        # Update xaxis properties
        fig.update_xaxes(title_text="State", row=p + 1, col=1)
        # Update yaxis properties
        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=p + 1, col=1)

        # Update title and height
        fig.update_layout(title_text=result_paths[p][:-20], width=1400, height=1200)

    return fig1, fig2


def plot_ml_results_spatial(result_paths):
    """

    :param result_paths: list of paths to results (different files for different classifiers)
    :return:
    """

    # accuracy
    fig = make_subplots(rows=len(result_paths), cols=1,
                        shared_yaxes=True,
                        vertical_spacing=0.02)

    for p in range(len(result_paths)):
        df = pd.read_csv(result_paths[p], header=0, sep=',')
        df.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        fig.add_trace(go.Bar(name=task_infos['classifier_order'][p],
                             x=df['state'],
                             y=df['accuracy']),
                      row=p + 1,
                      col=1)
        # Update xaxis properties
        fig.update_xaxes(title_text="State", row=p + 1, col=1)
        # Update yaxis properties
        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=p + 1, col=1)

        # Update title and height
        fig.update_layout(title_text=result_paths[p][:-20], width=1400, height=1200)
    return fig


def plot_ml_results_fairness_metrics_spatial(df, protected_attr):
    """

    :param df:
    :param protected_attr:
    :return:
    """
    # spd
    fig = px.bar(df, x=df['state'], y=df[f'{protected_attr}_spd'], title=f"{protected_attr} SPD")
    # dir
    fig2 = px.bar(df, x=df['state'], y=df[f'{protected_attr}_dir'], title=f"{protected_attr} DIR")

    return fig, fig2


def fairness_metrics(df, target: str, protected: str):
    """
    Given a dataframe and an indication of what are the protected and target attributes,
    this function returns the disparate impact ratio (ratio of positive outcomes in the
    unprivileged group divided by the ratio of positive outcomes in the privileged group)
    as well as statistical parity difference.

    :param df: state - year specific dataframe
    :param target: name of target variable column
    :param protected: name of protected variable column
    :return:
    """

    if protected == 'SEX':
        female_df = df[df['SEX'] == 2]
        len_female = female_df.shape[0]
        male_df = df[df['SEX'] == 1]
        len_male = male_df.shape[0]
        unprivileged_outcomes = female_df[female_df[target] == 1].shape[0]
        unprivileged_ratio = unprivileged_outcomes / len_female
        privileged_outcomes = male_df[male_df[target] == 1].shape[0]
        privileged_ratio = privileged_outcomes / len_male
        disp_ir = unprivileged_ratio / privileged_ratio
        spd = unprivileged_ratio - privileged_ratio

    elif protected == 'RAC1P':
        nonwhite_df = df[df['RAC1P'] != 1]
        len_nonwhite = nonwhite_df.shape[0]
        white_df = df[df['RAC1P'] == 1]
        len_white = white_df.shape[0]
        unprivileged_outcomes = nonwhite_df[nonwhite_df[target] == 1].shape[0]
        unprivileged_ratio = unprivileged_outcomes / len_nonwhite
        privileged_outcomes = white_df[white_df[target] == 1].shape[0]
        privileged_ratio = privileged_outcomes / len_white
        disp_ir = unprivileged_ratio / privileged_ratio
        spd = unprivileged_ratio - privileged_ratio
    return [disp_ir, spd]


def eda_metrics_usa(task: str):
    """
    This function computes for each US state and each time point present in
    the data folder demographic and fairness metrics of the original dataset,
    such as statistical partiy difference or disparate impact ratio.
    Results are saved in a dataframe to be later used for visualization purposes.
    :param task: name of the classification task
    :return:
    """

    if os.path.isfile(os.path.join(ddir, 'results', 'metrics', 'metrics_all_usa.tsv')):
        metrics = pd.read_csv(os.path.join(ddir, 'results', 'metrics', 'metrics_all_usa.tsv'), sep='\t')
        return metrics

    else:

        all_dfs = []

        for y in [str(x) for x in task_infos['years']]:
            male_percentage_priv = []
            white_percentage_priv = []
            spd_sex = []
            spd_race = []
            disp_sex = []
            disp_race = []

            for s in task_infos['states']:
                data = pd.read_csv(os.path.join(ddir, 'rawdata', y, '1-Year', f'{y}_{s}_{task}.csv'), sep=',')
                male_percentage_priv.append(round((len(data[(data['SEX'] == 1) & (data['ESR'] == 1)]) / len(data)), 2))
                white_percentage_priv.append(
                    round((len(data[(data['RAC1P'] == 1) & (data['ESR'] == 1)]) / len(data)), 2))
                disp_sex.append(fairness_metrics(data, 'ESR', 'SEX')[0])
                spd_sex.append(fairness_metrics(data, 'ESR', 'SEX')[1])
                disp_race.append(fairness_metrics(data, 'ESR', 'RAC1P')[0])
                spd_race.append(fairness_metrics(data, 'ESR', 'RAC1P')[1])

            distributions = pd.DataFrame(
                list(zip(task_infos['states'], [y] * len(task_infos['states']), male_percentage_priv,
                         white_percentage_priv, spd_sex, spd_race, disp_sex, disp_race)),
                columns=['state_code',
                         'year',
                         'male_percent',
                         'white_percent',
                         'spd_sex',
                         'spd_race',
                         'disp_sex',
                         'disp_race'])
            all_dfs.append(distributions)

        metrics = pd.concat(all_dfs).reset_index(drop=True)
        metrics.to_csv(os.path.join(ddir, 'results', 'metrics', f'metrics_all_usa_{task}.csv'), sep='\t')
        return metrics


if __name__ == "__main__":
    eda_metrics_usa("ACSEmployment")
    # eda_metrics_usa("ACSHealthInsurance")
