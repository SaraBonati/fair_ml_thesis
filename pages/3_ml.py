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
from utils.eda_utils import plot_ml_results_spatial, plot_ml_results_temporal, make_mapplot

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
    results_sklearn_paths = glob.glob(
        os.path.join(r"C:\Users\sarab\Desktop\Data_Science_MSc\master_thesis\thesis_code\results",
                     select_task, str(select_year), 'sklearn', select_state) + f'/{select_state}_test_all_*.csv')
    results_sklearn_paths.sort()

    df = pd.read_csv(results_sklearn_paths[0], header=0, sep=',')
    df.rename(columns={'Unnamed: 0': 'state'}, inplace=True)
    st.write(df)

    st.markdown(f"## Figure 1 - {select_state} as training data, results when tested on all other us states")

    # accuracy
    fig = plot_ml_results_spatial(results_sklearn_paths)
    st.plotly_chart(fig)

    # statistical parity + disparate impact ratio (sex)
    st.markdown(f"### SPD (SEX) - {select_state} as training data (closer to zero is better)")
    st.plotly_chart(make_mapplot(df, "sex_spd", "SPD (SEX)", select_context, "blues"))
    st.markdown(f"### DIR (SEX) - {select_state} as training data (higher is better)")
    st.plotly_chart(make_mapplot(df, "sex_dir", "DIR (SEX)", select_context, "blues"))

    # statistical parity + disparate impact ratio (race)
    st.markdown(f"### SPD (RACE) - {select_state} as training data (closer to zero is better)")
    st.plotly_chart(make_mapplot(df, "rac_spd", "SPD (RACE)", select_context, "reds"))
    st.markdown(f"### DIR (RACE) - {select_state} as training data (higher is better)")
    st.plotly_chart(make_mapplot(df, "rac_dir", "DIR (RACE)", select_context, "reds"))

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
