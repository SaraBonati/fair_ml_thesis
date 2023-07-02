# This script launches a Streamlit app designed ot help to visualize
# the machine learning analysis results of the ACS PUMS datasets
# for Master Thesis.
# To run the app, navigate to the project home directory and execute
# `streamlit run main_app.py`
#
# Author @ Sara Bonati
# Supervisor: Prof. Dr. Claudia Müller-Birn - Prof. Dr. Eirini Ntoutsi
# Project: Data Science MSc Master Thesis
#############################################

# general utility import
import numpy as np
import pandas as pd
import streamlit as st
import os
import glob
import json
from utils.eda_utils import plot_ml_results_spatial, plot_ml_results_temporal,\
    make_mapplot, map_plot_ml_results_spatial, roc_curve_sex, roc_curve_race

# directory management
wdir = os.getcwd()  # working directory
results_dir = r'C:\Users\sarab\Desktop\results2_download'
results_sampling_dir = r'C:\Users\sarab\Desktop\sampling_results'
# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, 'utils', 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())

#############################################
# App definition
#############################################

st.set_page_config(
    page_title="ML Analysis Results",
    page_icon="🤖",
    layout="wide"
)
st.markdown("# ML Analysis Results")
st.sidebar.markdown("# ML Analysis Results")
st.sidebar.markdown("In this page we show the results of the machine learning analysis applied to the classification "
                    "tasks "
            "across the spatial and temporal context. The goal is to compare the performance of normal classifiers, "
            "that do not receive constraints over the protected attributes VS the performance of fairness-aware "
            "classifiers.")

ml_form = st.form("ML Results form")
# get task, state and context to visualize
select_task = ml_form.selectbox('Which classification task do you want to focus on?', ['ACSEmployment'])
select_state = ml_form.selectbox('Which state used in training do you want to visualize?', task_infos['states'])
select_context = ml_form.selectbox('Which context do you want to focus on?', ['spatial', 'temporal'])
select_year = ml_form.selectbox('Which survey year (spatial context) or training year (temporal context) do you want to '
                         'use?', np.arange(min(task_infos['years']),max(task_infos['years'])+1))
ml_form_submitted = ml_form.form_submit_button("Submit")

if select_context == 'spatial':
    # select data paths according to year selection
    # ddir -> C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\thesis_code\results\ACSEmployment
    # or C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\thesis_code\results


    st.markdown("## Before sampling")

    # result paths (sklearn)
    results_sklearn_paths = glob.glob(
        os.path.join(results_dir,
                     select_task,
                     str(select_year),
                     'sklearn', select_state) + f'/{select_context}_{select_state}_test_all_*.csv')
    results_sklearn_paths.sort()
    #st.write(results_sklearn_paths)


    # result paths (aif360)
    results_aif360_paths = glob.glob(
        os.path.join(results_dir,
                     select_task, str(select_year), 'aif360', select_state) + f'/{select_context}'
                                                                              f'_{select_state}_test_all_*.csv')
    results_aif360_paths.sort()
    #st.write(results_aif360_paths)


    st.markdown(f"## Classifiers results in tabular form")
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.markdown(f"### Normal")

        which_clfier = st.selectbox('Which of the sklearn classifiers to visualize?',
                                    ('LinearSVC', 'LogReg', 'XGBoost'))

        which_path = [p for p in results_sklearn_paths if which_clfier in p]
        df_sklearn = pd.read_csv(which_path[0], header=0, sep=',')
        df_sklearn.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        st.write(f"Showing {which_clfier}")
        st.dataframe(df_sklearn, use_container_width=True)
    with col2:
        df_aif = pd.read_csv(results_aif360_paths[0], header=0, sep=',')
        df_aif.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        st.markdown(f"### Fairness-aware")
        st.write("Showing AdversarialDebiasing")
        st.dataframe(df_aif, use_container_width=True)

    st.markdown(f"## Figures")
    st.markdown(f"In the following figures we concentrate on the models' performance when using {select_state} as "
                f"training data, tested on all other us states. We first look at accuracy: accuracy results are "
                f"displayed below in a US map")

    # accuracy map plots
    map1, map2 = st.columns(2)

    with map1:

        which_path = [p for p in results_sklearn_paths if which_clfier in p]
        df_sklearn = pd.read_csv(which_path[0], header=0, sep=',')
        df_sklearn.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        df_sklearn = df_sklearn[~df_sklearn['state'].isin(['PR'])]
        fig_map = make_mapplot(df_sklearn, "accuracy", "Accuracy (normal classifiers)",
                               select_context,
                               "state", "blues")
        st.write("TODO: expand to show all classifiers with mapplot")
        st.plotly_chart(fig_map)

    with map2:
        df_aif = pd.read_csv(results_aif360_paths[0], header=0, sep=',')
        df_aif.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
        df_aif = df_aif[~df_aif['state'].isin(['PR'])]
        fig_map = make_mapplot(df_aif, "accuracy", "Accuracy (normal classifiers)",
                               select_context,
                               "state", "reds")
        st.write("TODO: expand to show all classifiers with mapplot")
        st.plotly_chart(fig_map)
    # fig = plot_ml_results_spatial(results_sklearn_paths)
    # st.plotly_chart(fig)

    st.markdown(f"What about the different machine learning models tested, will they show a difference in accuracy?"
                f"Let's take a closer look at these differences below. Each of the points next to the boxplots is one of the states used in testing.")
    fig_box = map_plot_ml_results_spatial(results_sklearn_paths,results_aif360_paths, 'accuracy')
    st.plotly_chart(fig_box)

    st.markdown(f"Accuracy is one of the most popular metrics, but it shouldn't be taken as the sole metric to judge "
                f"a model's performance. A point of interest in this work is the differential performance of "
                f"classifiers as a function of different protected attribute values. Let's plot ROC curves and see "
                f"for each protected attribute whether the false positive rate and true positive rates change "
                f"depending on attribute values.")

    st.pyplot(roc_curve_sex(select_state,select_year))
    st.pyplot(roc_curve_race(select_state, select_year))


    st.markdown(f"Next, we focus on fairness metrics. Here we display for our protected attributes, "
                f"SEX and RAC1P, the information across the US map for the "
                f"different metrics we collected. Note that: ")
    st.markdown(f"* Demographic parity difference (DPD) is defined as the difference between the largest and the "
                f"smallest group-level selection rate XX, across all values of the sensitive feature(s). The "
                f"demographic parity difference of 0 means that all groups have the same selection rate."
                f"For Demographic Parity Difference a value closer to 0 is better")
    st.markdown(f"* Equalized Odds Difference is collected here and stands for the greater of two metrics: "
                f"true_positive_rate_difference and false_positive_rate_difference. The former is the "
                f"difference between the largest and smallest of XX across all values of the sensitive feature(s). "
                f"The latter is defined similarly, but for XX. The equalized odds difference of 0 means that all "
                f"groups have the same true positive, true negative, false positive, and false negative rates.")


    # statistical parity + disparate impact ratio (sex)
    st.markdown(f"### DPD (SEX)")
    fig_box = map_plot_ml_results_spatial(results_sklearn_paths, results_aif360_paths, 'sex_dpd')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "sex_dpd", "DPD (SEX)", select_context,"state", "blues"))

    st.markdown(f"### EOD (SEX)")
    fig_box = map_plot_ml_results_spatial(results_sklearn_paths, results_aif360_paths, 'sex_eod')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "sex_eod", "EOD (SEX)", select_context,"state", "reds"))


    # statistical parity + disparate impact ratio (race)
    st.markdown(f"### DPD (RACE)")
    fig_box = map_plot_ml_results_spatial(results_sklearn_paths, results_aif360_paths, 'rac_dpd')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "rac_dpd", "DPD (RACE)", select_context,"state", "blues"))

    st.markdown(f"### EOD (RACE)")
    fig_box = map_plot_ml_results_spatial(results_sklearn_paths, results_aif360_paths, 'rac_eod')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "rac_eod", "EOD (RACE)", select_context,"state", "reds"))

if select_context == 'temporal':

    st.markdown(f"## Before sampling")
    st.markdown(f"## Classifiers results in tabular form")
    col1, col2 = st.columns(2, gap='large')
    # select data paths according to year selection
    # sklearn
    results_sklearn_paths = glob.glob(os.path.join(r'C:\Users\sarab\Desktop\results2_download', select_task,
                                                   str(select_year), 'sklearn',select_state,
                                                   f'{select_context}_{select_state}_test_{str(select_year)}.csv'))

    df_sklearn = pd.read_csv(results_sklearn_paths[0], sep=',')
    df_sklearn.rename(columns={'Unnamed: 0': 'classifier', 'Unnamed: 1': 'test_year'}, inplace=True)
    with col1:
        st.write(df_sklearn)

    # aif360
    results_aif360_paths = glob.glob(os.path.join(r'C:\Users\sarab\Desktop\results2_download', select_task,
                                                   str(select_year), 'aif360', select_state,
                                                   f'{select_context}_{select_state}_test_{str(select_year)}.csv'))

    df_aif = pd.read_csv(results_aif360_paths[0], sep=',')
    df_aif.rename(columns={'Unnamed: 0': 'classifier', 'Unnamed: 1': 'test_year'}, inplace=True)
    df_aif=df_aif[df_aif['classifier']!='ExponentiatedGradientReduction']
    with col2:
        st.write(df_aif)

    # accuracy
    st.markdown(f"### Predicitve Accuracy")
    fig = plot_ml_results_temporal(results_sklearn_paths,results_aif360_paths, select_year, 'accuracy')
    st.plotly_chart(fig)

    # statistical parity + disparate impact ratio (sex)
    st.markdown(f"### DPD (SEX)")
    fig_box = plot_ml_results_temporal(results_sklearn_paths, results_aif360_paths, select_year, 'sex_dpd')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "sex_dpd", "DPD (SEX)", select_context,"state", "blues"))

    st.markdown(f"### EOD (SEX)")
    fig_box = plot_ml_results_temporal(results_sklearn_paths, results_aif360_paths, select_year, 'sex_eod')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "sex_eod", "EOD (SEX)", select_context,"state", "reds"))

    # statistical parity + disparate impact ratio (race)
    st.markdown(f"### DPD (RACE)")
    fig_box = plot_ml_results_temporal(results_sklearn_paths, results_aif360_paths,select_year, 'rac_dpd')
    st.plotly_chart(fig_box)
    # st.plotly_chart(make_mapplot(df_sklearn, "rac_dpd", "DPD (RACE)", select_context,"state", "blues"))

    st.markdown(f"### EOD (RACE)")
    fig_box = plot_ml_results_temporal(results_sklearn_paths, results_aif360_paths,select_year, 'rac_eod')
    st.plotly_chart(fig_box)
