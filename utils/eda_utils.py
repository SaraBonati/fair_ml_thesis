import numpy as np
import pandas as pd
import os
import re
import glob
import json
import pickle
import argparse
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

wdir = os.getcwd()
ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data")
# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, 'utils', 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())


def merge_pickle_roc_files(files_path: list):
    """
    This function merges into one dataframe all fpr and tpr values from multiple states
    :param files_path:
    :return:
    """
    final_df_list = []
    pickle_files = glob.glob(files_path + r'\*.pickle')

    for f in pickle_files:
        # open a file, where you stored the pickled data
        file = open(pickle_files[0], 'rb')
        data = pickle.load(file)
        sex_data = 1
        race_data = 1


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
        if c in list(cols_infos.keys()) and c not in ["ESR", "HINS2"]:
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
    if context == "temporal":
        fig = px.choropleth(df,
                            locations=state_col,
                            animation_frame="year",
                            color=metric,
                            color_continuous_scale=colors,
                            locationmode='USA-states',
                            scope="usa",
                            height=1300)
    elif context == "spatial":
        fig = px.choropleth(df,
                            locations=state_col,
                            color=metric,
                            color_continuous_scale=colors,
                            locationmode='USA-states',
                            scope="usa",
                            height=1300)
        fig.update_traces(colorbar=dict(orientation='v',thickness=0.1))
        #fig.update_layout(coloraxis=dict(colorbar=dict(orientation='h')))

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
                  color_discrete_map={2: 'coral', 1: 'teal'})  # were "female" and "male" before
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

def roc_curve_sex(state: str, year: str):
    """

    :param state:
    :param year:
    :return:
    """
    import matplotlib.gridspec as gridspec
    protected_variables = {'SEX': ['male', 'female'], 'RAC1P': ['white', 'black', 'other']}
    classifier_names = ["LogReg", "LinearSVC", "XGBoost", "AdversarialDebiasing"]
    states=[state]

    fig = plt.figure(figsize=(15, 4))
    gs = gridspec.GridSpec(1, 4)

    for s in range(len(states)):
        for c in range(len(classifier_names)):
            print(f"{states[s]} - {classifier_names[c]}")
            if len(classifier_names[c]) > 10:
                state_spatial_path = os.path.join(r"C:\Users\sarab\Desktop\results2_download\ACSEmployment", str(year),
                                                  'aif360', states[s])
            else:
                state_spatial_path = os.path.join(r"C:\Users\sarab\Desktop\results2_download\ACSEmployment", str(year),
                                                  'sklearn', states[s])

            pickle_files = glob.glob(state_spatial_path + f'\*_{classifier_names[c]}.pickle')
            fpr_mean = np.linspace(0, 1, 100)
            interp_tprs = {i: np.zeros((len(pickle_files), 100)) for i in protected_variables['SEX']}
            for i in range(len(pickle_files)):
                file = open(pickle_files[i], 'rb')
                data = pickle.load(file)

                fpr_male = data['SEX_fpr']['male']
                tpr_male = data['SEX_tpr']['male']
                fpr_female = data['SEX_fpr']['female']
                tpr_female = data['SEX_tpr']['female']

                interp_tpr_male = np.interp(fpr_mean, fpr_male, tpr_male)
                interp_tpr_male[0] = 0.0
                interp_tprs['male'][i, :] = interp_tpr_male
                interp_tpr_female = np.interp(fpr_mean, fpr_female, tpr_female)
                interp_tpr_female[0] = 0.0
                interp_tprs['female'][i, :] = interp_tpr_female

            tpr_mean_male = np.mean(interp_tprs['male'], axis=0)
            tpr_mean_male[-1] = 1.0
            tpr_std_male = np.std(interp_tprs['male'], axis=0)
            tpr_upper_male = np.clip(tpr_mean_male + tpr_std_male, 0, 1)
            tpr_lower_male = tpr_mean_male - tpr_std_male

            tpr_mean_female = np.mean(interp_tprs['female'], axis=0)
            tpr_mean_female[-1] = 1.0
            tpr_std_female = np.std(interp_tprs['female'], axis=0)
            tpr_upper_female = np.clip(tpr_mean_female + tpr_std_female, 0, 1)
            tpr_lower_female = tpr_mean_female - tpr_std_female

            ax = fig.add_subplot(gs[s, c])
            ax.set_title(f'{classifier_names[c]}' + "\n" + f'(train state={states[s]})', fontsize=15)
            ax.set_xlabel('false positive rate (fpr)', fontsize=15)
            if c == 0:
                ax.set_ylabel('true positive rate (tpr)', fontsize=15)

            ax.plot(fpr_mean, tpr_mean_male, color='teal', label='SEX = male')
            ax.fill_between(fpr_mean, tpr_lower_male, tpr_upper_male, color='teal', alpha=0.5)
            ax.plot(fpr_mean, tpr_mean_female, color='coral', label='SEX = female')
            ax.fill_between(fpr_mean, tpr_lower_female, tpr_upper_female, color='coral', alpha=0.5)

            if c == 0 and s == 0:
                ax.legend(loc=4)

    return fig


def roc_curve_race(state: str, year:str):
    """

    :param state:
    :param year:
    :return:
    """
    import matplotlib.gridspec as gridspec
    protected_variables = {'SEX': ['male', 'female'], 'RAC1P': ['white', 'black', 'other']}
    classifier_names = ["LogReg", "LinearSVC", "XGBoost", "AdversarialDebiasing"]
    states=[state]

    fig2 = plt.figure(figsize=(15,4))
    gs2 = gridspec.GridSpec(1, 4)

    for s in range(len(states)):
        for c in range(len(classifier_names)):
            print(f"{states[s]} - {classifier_names[c]}")
            if len(classifier_names[c]) > 10:
                state_spatial_path = os.path.join(r"C:\Users\sarab\Desktop\results2_download\ACSEmployment", str(year),
                                                  'aif360', states[s])
            else:
                state_spatial_path = os.path.join(r"C:\Users\sarab\Desktop\results2_download\ACSEmployment", str(year),
                                                  'sklearn', states[s])

            pickle_files = glob.glob(state_spatial_path + f'\*_{classifier_names[c]}.pickle')
            fpr_mean = np.linspace(0, 1, 100)
            interp_tprs = {i: np.zeros((len(pickle_files), 100)) for i in protected_variables['RAC1P']}
            for i in range(len(pickle_files)):
                file = open(pickle_files[i], 'rb')
                data = pickle.load(file)

                fpr_white = data['RAC1P_fpr']['white']
                tpr_white = data['RAC1P_tpr']['white']
                fpr_black = data['RAC1P_fpr']['black']
                tpr_black = data['RAC1P_tpr']['black']
                fpr_other = data['RAC1P_fpr']['other']
                tpr_other = data['RAC1P_tpr']['other']

                interp_tpr_white = np.interp(fpr_mean, fpr_white, tpr_white)
                interp_tpr_white[0] = 0.0
                interp_tprs['white'][i, :] = interp_tpr_white
                interp_tpr_black = np.interp(fpr_mean, fpr_black, tpr_black)
                interp_tpr_black[0] = 0.0
                interp_tprs['black'][i, :] = interp_tpr_black
                interp_tpr_other = np.interp(fpr_mean, fpr_other, tpr_other)
                interp_tpr_other[0] = 0.0
                interp_tprs['other'][i, :] = interp_tpr_other

            tpr_mean_white = np.mean(interp_tprs['white'], axis=0)
            tpr_mean_white[-1] = 1.0
            tpr_std_white = np.std(interp_tprs['white'], axis=0)
            tpr_upper_white = np.clip(tpr_mean_white + tpr_std_white, 0, 1)
            tpr_lower_white = tpr_mean_white - tpr_std_white

            tpr_mean_black = np.mean(interp_tprs['black'], axis=0)
            tpr_mean_black[-1] = 1.0
            tpr_std_black = np.std(interp_tprs['black'], axis=0)
            tpr_upper_black = np.clip(tpr_mean_black + tpr_std_black, 0, 1)
            tpr_lower_black = tpr_mean_black - tpr_std_black

            tpr_mean_other = np.mean(interp_tprs['other'], axis=0)
            tpr_mean_other[-1] = 1.0
            tpr_std_other = np.std(interp_tprs['other'], axis=0)
            tpr_upper_other = np.clip(tpr_mean_other + tpr_std_other, 0, 1)
            tpr_lower_other = tpr_mean_other - tpr_std_other

            ax = fig2.add_subplot(gs2[s, c])
            ax.set_title(f'{classifier_names[c]}' + "\n" + f'(train state={states[s]})', fontsize=15)
            ax.set_xlabel('false positive rate (fpr)', fontsize=15)
            if c == 0:
                ax.set_ylabel('true positive rate (tpr)', fontsize=15)
            ax.plot(fpr_mean, tpr_mean_white, color='blue', label='RAC1P = white')
            ax.fill_between(fpr_mean, tpr_lower_white, tpr_upper_white, color='blue', alpha=0.5)
            ax.plot(fpr_mean, tpr_mean_black, color='red', label='RAC1P = black')
            ax.fill_between(fpr_mean, tpr_lower_black, tpr_upper_black, color='red', alpha=0.5)
            ax.plot(fpr_mean, tpr_mean_other, color='green', label='RAC1P = other')
            ax.fill_between(fpr_mean, tpr_lower_other, tpr_upper_other, color='green', alpha=0.5)

            if c == 0 and s == 0:
                ax.legend(loc=4)

    return fig2

def map_plot_ml_results_spatial(result_paths_sklearn: list, result_paths_aif: list, dependent: str):
    """

    :param result_paths_sklearn: list of paths to csv result files (sklearn)
    :param result_paths_aif: list of paths to csv result files (aif)
    :param dependent: the variable to show on y-axis in the plot
    :return:
    """

    dependent_color = {"accuracy":"#569de8",
                       "sex_dpd":"#d1c84d",
                       "sex_eod": "#c44629",
                       "rac_dpd": "#948809",
                       "rac_eod": "#e8917d",
                       }

    dfs = []
    for p in range(len(result_paths_sklearn)):
        df = pd.read_csv(result_paths_sklearn[p], header=0, sep=',')
        df.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        state = os.path.split(result_paths_sklearn[p])[1][8:10]
        start, end = f'spatial_{state}_test_all_', '.csv'
        df['CLASSIFIER'] = re.search('%s(.*)%s' % (start, end), result_paths_sklearn[p]).group(1)
        #df['CLASSIFIER_TYPE'] = "normal"
        dfs.append(df)
    #for p in range(len(result_paths_aif)):
    df = pd.read_csv(result_paths_aif[0], header=0, sep=',')
    df.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
    state = os.path.split(result_paths_aif[0])[1][8:10]
    start, end = f'spatial_{state}_test_all_', '.csv'
    df['CLASSIFIER'] = re.search('%s(.*)%s' % (start, end), result_paths_aif[0]).group(1)
    #df['CLASSIFIER_TYPE'] = "fairness_aware"
    dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.reset_index(inplace=True,drop=True)
    # remove puerto rico from result files (not included in final states in the end)
    final_df = final_df[~final_df['state'].isin(['PR'])]

    fig = px.box(final_df,
                 x="CLASSIFIER",
                 y=dependent,
                 hover_data=["state"],
                 points="all",
                 color_discrete_sequence=[dependent_color[dependent]],
                 width=1900)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1),
    )
    return fig


def plot_ml_results_temporal(result_path):
    """

    :param result_paths: list of paths to results (different files for different classifiers)
    :return:
    """

    df = pd.read_csv(result_path, sep=',')
    df.rename(columns={'Unnamed: 0': 'clf', 'Unnamed: 1': 'year'}, inplace=True)

    # fig = px.bar(df, x="year", y="accuracy", color="clf", barmode="group")
    fig = px.line(df, x="year", y="accuracy", color="clf", markers=True)
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
