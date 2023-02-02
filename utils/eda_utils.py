import numpy as np
import pandas as pd
import os
import re
import json
import argparse
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

wdir = os.getcwd()
ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data")
# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, 'utils', 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())


def preprocess_healthinsurance(df):
    """

    :param df:
    :return:
    """
    df['RAC1P'] = df[['RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACPI', 'RACSOR', 'RACWHT']].idxmax(axis=1)
    race_codes = {'RACAIAN': 5, 'RACASN': 6, 'RACBLK': 2, 'RACNH': 7, 'RACPI': 7, 'RACSOR': 8, 'RACWHT': 1}
    df['RAC1P'] = df['RAC1P'].map(race_codes)
    df.drop(['RACAIAN', 'RACASN', 'RACBLK', 'RACNH', 'RACPI', 'RACSOR', 'RACWHT'], axis=1, inplace=True)
    return df


def categorize(df, cat_columns):
    """
    This function turns a column in pandas dataframe to int first, and then to category
    :param df:
    :param cat_columns:
    :return:
    """

    for c in cat_columns:
        df[c] = df[c].astype(np.int64)
        df[c] = df[c].astype('category')

def map_int_to_cat(df, cols_infos, cols):
    """
    This function, given a dataframe and a dict containing number to category mappings,
    turns the pandas category column with numerical values to columns displaying the category corresponding to each
    number. The resulting dataframe is returned
    :param df:
    :param cols_infos:
    :return:
    """
    for c in cols:
        if c in list(cols_infos.keys()) and c not in ["ESR","HINS2"]:
            df[c] = df[c].map(cols_infos[c])
    return df

def merge_dataframes_eda(paths: list, task: str):
    """
    given a list of paths to dataframes, this function creates a state specific dataframe with all
    years merged
    :param paths:
    :param task:
    :return:
    """

    dfs = []
    for p in range(len(paths)):
        df = pd.read_csv(paths[p], sep=",")

        if task == "ACSHealthInsurance":
            df = preprocess_healthinsurance(df)

        df["YEAR"] = task_infos["years"][p]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def make_mapplot(df, metric: str, title: str, context: str, state_col: str, colors: str = "Viridis_r"):
    """
    Creates a choropleth map plot that is animated showing yearly progression
    of specified metric
    :param df: dataframe to use in plot
    :param metric: name of variable to display in animated map plot
    :param title: title of plot
    :param context: spatial or temporal
    :param state_col: specifies the name of the df column with state codes
    :param colors: name of colormap to use in plot

    :return:
    """
    if context == "Temporal":
        fig = px.choropleth(df,
                            locations=state_col,
                            animation_frame="year",
                            color=metric,
                            color_continuous_scale=colors,
                            locationmode='USA-states',
                            scope="usa",
                            height=1300)
    elif context == "Spatial":
        fig = px.choropleth(df,
                            locations=state_col,
                            color=metric,
                            color_continuous_scale=colors,
                            locationmode='USA-states',
                            scope="usa",
                            height=1300)

    return fig


def make_protected_plots(df, df_all_years, target_name: str):
    """
    Returns basic plot of distribution of protected attributes
    :param df:
    :param df_all_years:
    :param target_name:
    :return:
    """
    # sex
    df_percent = df['SEX'].value_counts(normalize=True).reset_index().rename(columns={'index': 'sex_class'})
    fig1 = px.pie(df_percent,
                  values='SEX',
                  names='sex_class',
                  color='sex_class',
                  color_discrete_map={2: 'coral', 1: 'teal'}) # were "female" and "male" before
    fig1.update_traces(textposition='inside', textinfo='percent+label')

    # race
    df_percent = df['RAC1P'].value_counts(normalize=True).reset_index().rename(columns={'index': 'race'})
    fig2 = px.pie(df_percent,
                  values='RAC1P',
                  names='race',
                  color='race',
                  color_discrete_map={
                      1: "Crimson",
                      2: "SteelBlue",
                      3: "DarkViolet"
                  })
                  # color_discrete_map={
                  #     "White": "Crimson",
                  #     "Black/African American": "SteelBlue",
                  #     "American Indian": "Silver",
                  #     "Alaska Native": "PowderBlue",
                  #     "American Indian and Alaska Native tribes": "Chocolate",
                  #     "Asian": "DarkViolet",
                  #     "Native Hawaiian and Other Pacific Islander": "LimeGreen",
                  #     "Some Other Race": "DarkSlateGrey",
                  #     "Two or More Races": "DarkSeaGreen"
                  # })
    fig2.update_traces(textposition='inside', textinfo='percent+label')

    # race detailed
    df_percent = df['RAC1P_r'].value_counts(normalize=True).reset_index().rename(columns={'index': 'race'})
    fig3 = px.pie(df_percent,
                  values='RAC1P_r',
                  names='race',
                  color='race',
                  color_discrete_map={
                      1: "Crimson",
                      2: "SteelBlue",
                      3: "DarkViolet"
                  })
                  # color_discrete_map={
                  #     "White": "Crimson",
                  #     "Black/African American": "SteelBlue",
                  #     "American Indian": "Silver",
                  #     "Alaska Native": "PowderBlue",
                  #     "American Indian and Alaska Native tribes": "Chocolate",
                  #     "Asian": "DarkViolet",
                  #     "Native Hawaiian and Other Pacific Islander": "LimeGreen",
                  #     "Some Other Race": "DarkSlateGrey",
                  #     "Two or More Races": "DarkSeaGreen"
                  # })
    fig3.update_traces(textposition='inside', textinfo='percent+label')

    return fig1, fig2, fig3


def make_demographic_plots(df, df_all_years, target_name: str):
    """
    Returns histograms and pie charts of demographic variables
    for a state - year specific dataframe
    :param df: dataframe (state - year specific)
    :param df_all_years: dataframe (state specific, all years)
    :param target_name: name of target variable column
    :return:
    """
    # one year only
    fig1 = px.histogram(df, x="RAC1P", y=target_name, facet_col=target_name,
                        color='SEX',
                        color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                        barmode='group',
                        histfunc='count',
                        width=800,
                        height=800)

    # all years
    fig2 = px.histogram(df_all_years, x="RAC1P", y=target_name,
                        facet_row=target_name,
                        facet_col="YEAR",
                        color='SEX',
                        color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                        barmode='group',
                        histfunc='count'
                        )
    # AGEP
    # one year only
    fig3 = px.box(df, x="RAC1P", y="AGEP", facet_col=target_name,
                  color='SEX',
                  color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                  boxmode='group',
                  width=1200,
                  height=800)

    # all years
    fig4 = px.box(df_all_years, x="RAC1P", y="AGEP", facet_row=target_name,
                  facet_col="YEAR",
                  color='SEX',
                  color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                  boxmode='group',
                  width=1200,
                  height=800
                  )
    # SCHL
    # one year
    fig5 = px.histogram(df, x="SCHL", y=target_name,
                        facet_col="RAC1P",
                        color='SEX',
                        category_orders={"SCHL": [
                            "No schooling completed",
                            "Nursery school, preschool",
                            "Kindergarten",
                            "Grade 1",
                            "Grade 2",
                            "Grade 3",
                            "Grade 4",
                            "Grade 5",
                            "Grade 6",
                            "Grade 7",
                            "Grade 8",
                            "Grade 9",
                            "Grade 10",
                            "Grade 11",
                            "Grade 12 - no diploma",
                            "Regular high school diploma",
                            "GED or alternative credential",
                            "Some college, but less than 1 year",
                            "1 or more years of college credit, no degree",
                            "Associate degree",
                            "Bachelor degree",
                            "Master degree",
                            "Professional degree beyond a bachelor degree",
                            "Doctorate degree"
                        ]
                        },
                        color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                        barmode='group',
                        histfunc='count',
                        width=1000,
                        height=800)
    # all years
    fig6 = px.histogram(df_all_years, x="SCHL", y=target_name,
                        facet_row="RAC1P",
                        facet_col="YEAR",
                        color='SEX',
                        category_orders={"SCHL": [
                            "No schooling completed",
                            "Nursery school, preschool",
                            "Kindergarten",
                            "Grade 1",
                            "Grade 2",
                            "Grade 3",
                            "Grade 4",
                            "Grade 5",
                            "Grade 6",
                            "Grade 7",
                            "Grade 8",
                            "Grade 9",
                            "Grade 10",
                            "Grade 11",
                            "Grade 12 - no diploma",
                            "Regular high school diploma",
                            "GED or alternative credential",
                            "Some college, but less than 1 year",
                            "1 or more years of college credit, no degree",
                            "Associate degree",
                            "Bachelor degree",
                            "Master degree",
                            "Professional degree beyond a bachelor degree",
                            "Doctorate degree"
                        ]
                        },
                        color_discrete_map={'Female': 'coral', 'Male': 'teal'},
                        barmode='group',
                        histfunc='count',
                        width=2600,
                        height=2000)

    return fig1, fig2, fig3, fig4, fig5, fig6


def plot_ml_results_spatial(result_paths: list):
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
        fig.update_layout(title_text=result_paths[p][:-20], width=1400, height=2500)
    return fig


def map_plot_ml_results_spatial(result_paths_sklearn: list, result_paths_aif: list):
    """

    :param result_paths_sklearn:
    :param result_paths_aif:
    :return:
    """

    dfs = []
    for p in range(len(result_paths_sklearn)):
        df = pd.read_csv(result_paths_sklearn[p], header=0, sep=',')
        state = os.path.split(result_paths_sklearn[p])[1][:2]
        start, end = f'{state}_test_all_', '.csv'
        df['CLASSIFIER'] = re.search('%s(.*)%s' % (start, end), result_paths_sklearn[p]).group(1)
        df['CLASSIFIER_TYPE'] = "normal"
        dfs.append(df)
    for p in range(len(result_paths_aif)):
        df = pd.read_csv(result_paths_aif[p], header=0, sep=',')
        state = os.path.split(result_paths_aif[p])[1][8:10]
        start, end = f'spatial_{state}_test_all_', '.csv'
        df['CLASSIFIER'] = re.search('%s(.*)%s' % (start, end), result_paths_aif[p]).group(1)
        df['CLASSIFIER_TYPE'] = "fairness_aware"
        dfs.append(df)

    final_df = pd.concat(dfs)
    fig = px.box(final_df, x="CLASSIFIER", y="accuracy", color='CLASSIFIER_TYPE', points="all",
                 width=1900)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig


def plot_ml_results_temporal(result_path):
    """

    :param result_paths: list of paths to results (different files for different classifiers)
    :return:
    """

    df = pd.read_csv(result_path, sep=',')
    df.rename(columns={'Unnamed: 0': 'clf', 'Unnamed: 1': 'year'}, inplace=True)

    #fig = px.bar(df, x="year", y="accuracy", color="clf", barmode="group")
    fig = px.line(df, x="year", y="accuracy", color="clf",markers=True)
    # Update xaxis properties
    fig.update_xaxes(title_text="Year")
    # Update yaxis properties
    fig.update_yaxes(title_text="Accuracy", range=[0, 1])

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


def eda_metrics_usa(task: str, overwrite: bool = False):
    """
    This function computes for each US state and each time point present in
    the data folder demographic and fairness metrics of the original dataset,
    such as statistical partiy difference or disparate impact ratio.
    Results are saved in a dataframe to be later used for visualization purposes.
    :param overwrite:
    :param task: name of the classification task
    :return:
    """

    if os.path.isfile(os.path.join(ddir, 'results', 'metrics', f'metrics_all_usa_{task}.tsv')) and not overwrite:
        metrics = pd.read_csv(os.path.join(ddir, 'results', 'metrics', f'metrics_all_usa_{task}.tsv'), sep='\t')
        return metrics

    else:

        all_dfs = []
        target = task_infos["tasks"][task_infos["task_col_map"][task]]["target"]

        for y in [str(x) for x in task_infos['years']]:
            male_percentage_priv = []
            white_percentage_priv = []
            spd_sex = []
            spd_race = []
            disp_sex = []
            disp_race = []

            for s in task_infos['states']:
                data = pd.read_csv(os.path.join(ddir, 'rawdata', y, '1-Year', f'{y}_{s}_{task}.csv'), sep=',')

                if task == "ACSHealthInsurance":
                    data = preprocess_healthinsurance(data)
                    # recode RAC1P values for all tasks

                data.loc[data['RAC1P'] > 2, 'RAC1P'] = 3

                male_percentage_priv.append(round((len(data[(data['SEX'] == 1) & (data[target] == 1)]) / len(data)), 2))
                white_percentage_priv.append(
                    round((len(data[(data['RAC1P'] == 1) & (data[target] == 1)]) / len(data)), 2))
                disp_sex.append(fairness_metrics(data, target, 'SEX')[0])
                spd_sex.append(fairness_metrics(data, target, 'SEX')[1])
                disp_race.append(fairness_metrics(data, target, 'RAC1P')[0])
                spd_race.append(fairness_metrics(data, target, 'RAC1P')[1])

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
        metrics.to_csv(os.path.join(ddir, 'results', 'metrics', f'metrics_all_usa_{task}.csv'), sep=',')
        return metrics


if __name__ == "__main__":

    # --------Specify arguments--------------------------
    parser = argparse.ArgumentParser(
        description="EDA Utils (Project: Fair ML Master Thesis)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--mode", type=str, help="indicate what method to run (edametrics)", required=False
    )
    parser.add_argument(
        "-o", "--overwrite", action='store_true', help="for (edametrics) overwrite existing file", required=False
    )
    args = parser.parse_args()

    if args.mode == 'edametrics':
        wdir = os.path.split(os.getcwd())[0]
        ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data")
        # tasks metadata (e,g, which columns are categorical, which column is the target etc..)
        json_file_path = os.path.join(wdir, 'utils', 'tasks_metadata.json')
        with open(json_file_path, 'r') as j:
            task_infos = json.loads(j.read())

        print(ddir)
        print(json_file_path)

        # eda_metrics_usa("ACSEmployment", args.overwrite)
        eda_metrics_usa("ACSHealthInsurance", args.overwrite)
