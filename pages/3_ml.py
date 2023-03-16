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
    make_mapplot, map_plot_ml_results_spatial, plot_roc_curve_spatial

# directory management
wdir = os.getcwd()  # working directory
ddir = os.path.join(os.path.split(os.path.split(wdir)[0])[0],"master_thesis", "fair_ml_thesis_data", "results")
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
)
st.markdown("# ML Analysis Results")
st.sidebar.markdown("# ML Analysis Results")
st.sidebar.markdown("In this page we show the results of the machine learning analysis applied to the classification "
                    "tasks "
            "across the spatial and temporal context. The goal is to compare the performance of normal classifiers, "
            "that do not receive constraints over the protected attributes VS the performance of fairness-aware "
            "classifiers. We expect to see a difference in performance accuracy, with normal classifiers having more "
            "room for improvement in the target class predictions and with fairness-aware classifiers showing better "
            "fairness metrics performance.")

ml_form = st.form("ML Results form")
# get task, state and context to visualize
select_task = ml_form.selectbox('Which classification task do you want to focus on?', task_infos['task_names'])
select_state = ml_form.selectbox('Which state used in training do you want to visualize?', task_infos['states'])
select_context = ml_form.selectbox('Which context do you want to focus on?', ['Spatial', 'Temporal'])
ml_form_submitted = ml_form.form_submit_button("Submit")

if select_context == 'Spatial':
    select_year = st.selectbox('Which year do you want to use?', np.arange(min(task_infos['years']),
                                                                           max(task_infos['years'])+1))
    # select data paths according to year selection
    # ddir -> C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\thesis_code\results\ACSEmployment
    # or C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\thesis_code\results
    # result paths (sklearn)
    results_sklearn_paths = glob.glob(
        os.path.join(r"C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\thesis_code\results",
                     select_task, str(select_year), 'sklearn', select_state) + f'/{select_state}_test_all_*.csv')
    results_sklearn_paths.sort()
    df_sklearn = pd.read_csv(results_sklearn_paths[0], header=0, sep=',')
    df_sklearn.rename(columns={'Unnamed: 0': 'state'}, inplace=True)

    # result paths (aif360)
    results_aif360_paths = glob.glob(
        os.path.join(r"C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\fair_ml_thesis_data\results",
                     select_task, str(select_year), 'aif360', select_state) + f'/{select_context}'
                                                                              f'_{select_state}_test_all_*.csv')
    results_aif360_paths.sort()
    df_aif = pd.read_csv(results_aif360_paths[0], header=0, sep=',')
    df_aif.rename(columns={'Unnamed: 0': 'state'}, inplace=True)

    st.markdown(f"## Classifiers results in tabular form")
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.markdown(f"### Normal")
        st.dataframe(df_sklearn, use_container_width=True)
    with col2:
        st.markdown(f"### Fairness-aware")
        st.dataframe(df_aif, use_container_width=True)

    st.markdown(f"## Figures")
    st.markdown(f"In the following figures we concentrate on the models' performance when using {select_state} as "
                f"training data, tested on all other us states. We first look at accuracy: accuracy results are "
                f"displayed below in a US map")

    # accuracy
    fig_map = make_mapplot(df_sklearn, "accuracy", "Accuracy (normal classifiers)",
                           select_context,
                           "state", "blues")
    st.write("TODO: expand to show all classifiers with mapplot")
    st.plotly_chart(fig_map)
    # fig = plot_ml_results_spatial(results_sklearn_paths)
    # st.plotly_chart(fig)

    st.markdown(f"What about the different machine learning models tested, will they show a difference in accuracy?"
                f"Let's take a closer look at these differences below.")
    fig_box = map_plot_ml_results_spatial(results_sklearn_paths,results_aif360_paths)
    st.plotly_chart(fig_box)

    st.markdown(f"Accuracy is one of the most popular metrics, but it shouldn't be taken as the sole metric to judge "
                f"a model's performance. A point of interest in this work is the differential performance of "
                f"classifiers as a function of different protected attribute values. Let's plot ROC curves and see "
                f"for each protected attribute whether the false positive rate and true positive rates change "
                f"depending on attribute values.")

    #plot_roc_curve_spatial(os.path.join(r"C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\fair_ml_thesis_data
    # \results",
    #                                   select_task, str(select_year), 'sklearn', select_state), 'SEX')

    st.markdown(f"Next, we focus on fairness metrics. Here we display for our protected attributes, "
                f"SEX and RAC1P, the information across the US map for the"
                f"different metrics we collected. Note that: ")
    st.markdown(f"* for Statistical Parity a value closer to 0 is better")
    st.markdown(f"* for Disparate Impact Ratio a higher value is better")

    col1, col2 = st.columns(2, gap='large')
    # statistical parity + disparate impact ratio (sex)
    with col1:
        st.markdown(f"### SPD (SEX)")
        st.plotly_chart(make_mapplot(df_sklearn, "sex_spd", "SPD (SEX)", select_context,"state", "blues"))
    with col2:
        st.markdown(f"### DIR (SEX)")
        st.plotly_chart(make_mapplot(df_sklearn, "sex_dir", "DIR (SEX)", select_context,"state", "reds"))

    col1, col2 = st.columns(2, gap='large')
    # statistical parity + disparate impact ratio (race)
    with col1:
        st.markdown(f"### SPD (RACE)")
        st.plotly_chart(make_mapplot(df_sklearn, "rac_spd", "SPD (RACE)", select_context,"state", "blues"))
    with col2:
        st.markdown(f"### DIR (RACE)")
        st.plotly_chart(make_mapplot(df_sklearn, "rac_dir", "DIR (RACE)", select_context,"state", "reds"))

if select_context == 'Temporal':
    select_year = st.slider('Which year do you want to use?', 2014, 2018)
    # select data paths according to year selection
    results_sklearn_paths = glob.glob(os.path.join(ddir, select_task, str(select_year), 'sklearn',select_state,
                                                   f'temporal_{select_state}_test_{str(select_year)}.csv'))

    df = pd.read_csv(results_sklearn_paths[0], sep=',')
    st.write(df)

    # accuracy
    fig = plot_ml_results_temporal(results_sklearn_paths[0])
    st.plotly_chart(fig)
