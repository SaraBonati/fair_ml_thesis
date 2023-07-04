# This script launches a Streamlit app designed ot help in
# the exploratory data analysis (EDA) process of the ACS PUMS datasets
# for Master Thesis.
# The EDA script shows plots for all US states (similar to figures in thesis pdf).
# To run the app, navigate to the project home directory and execute
# `streamlit run main_app.py`
#
# Author @ Sara Bonati
# Supervisor: Prof. Dr. Claudia Müller-Birn - Prof. Dr. Eirini Ntoutsi
# Project: Data Science MSc Master Thesis
#############################################

import json
import os
from utils.eda_utils import eda_metrics_usa, make_mapplot, make_demographic_plots, merge_dataframes_eda, \
    preprocess_healthinsurance, categorize, map_int_to_cat, make_protected_plots
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# directory management
wdir = os.getcwd()
ddir = os.path.join(os.path.split(wdir)[0], "fair_ml_thesis_data", "rawdata")

# tasks metadata (e,g, which columns are categorical, which column is the target etc..)
json_file_path = os.path.join(wdir, 'utils', 'tasks_metadata.json')
with open(json_file_path, 'r') as j:
    task_infos = json.loads(j.read())

# columns metadata (what do all numbers in categorical columns mean)
json_file_path = os.path.join(wdir, 'utils', 'cols_infos.json')
with open(json_file_path, 'r') as j:
    cols_infos = json.loads(j.read(), object_hook=lambda d: {int(k)
                                                             if k.lstrip('-').isdigit() else k: v for k, v in
                                                             d.items()})

#############################################
# App definition
#############################################

st.set_page_config(
    page_title="EDA",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
            # EDA of US Census dataset (all states)
            """)
st.sidebar.markdown("# EDA of dataset")
st.sidebar.markdown("In this section of the app you can view EDA plot for all US states in a specific survey year."
                    "Similar to what is represented in the thesis pdf, we show the proportion of SEX and RAC1P "
                    "survey units in the state samples, both standalone and in relation to the target class ESR.")

####################
task = 'ACSEmployment'
states = task_infos['states']


EDA_form = st.form("My form")
select_year = EDA_form.selectbox('Which year do you want to see?', np.arange(min(task_infos['years']),
                                                                             max(task_infos['years'])+1))
submitted = EDA_form.form_submit_button("Submit")

if submitted:

    data_paths = [os.path.join(ddir, str(select_year), '1-Year',f'{str(select_year)}_{state}_{task}.csv') for state in
                  states]
    data_merged = merge_dataframes_eda(data_paths, 'EDA')

    data_merged["RAC1P_r"] = data_merged['RAC1P']
    data_merged.loc[data_merged['RAC1P'] > 2, 'RAC1P'] = 3

    categorize(data_merged,
               task_infos['tasks'][task_infos['task_col_map'][task]]['cat_columns'] + ['RAC1P_r'])

    data_merged['RAC1P'] = data_merged['RAC1P'].map(cols_infos['RAC1P'])
    data_merged['RAC1P'] = data_merged['RAC1P'].map({'White':'White','Other':'Other','Black/African American':'Black'})
    data_merged['RAC1P_r'] = data_merged['RAC1P_r'].map(cols_infos['RAC1P_r'])
    data_merged['SEX'] = data_merged['SEX'].map(cols_infos['SEX'])
    data_merged['MAR'] = data_merged['MAR'].map(cols_infos['MAR'])
    data_merged['SCHL'] = data_merged['SCHL'].map(cols_infos['SCHL'])
    data_merged['NATIVITY'] = data_merged['NATIVITY'].map(cols_infos['NATIVITY'])
    data_merged['ESR'] = data_merged['ESR'].map(cols_infos['ESR'])

    st.dataframe(data_merged.iloc[:40,:])


    # normal plots (no ESR info)
    st.markdown('## SEX')
    sex_df = data_merged.groupby(by=['STATE'])['SEX'].value_counts(normalize=True).to_frame(
        'Proportion').reset_index()

    g = sns.catplot(data=sex_df, x="SEX", y='Proportion', col="STATE", col_wrap=7,
                    kind="bar",
                    errorbar=None,
                    height=6, palette={'Female': 'coral', 'Male': 'teal'}, aspect=.4)  # 4,0.6
    for ax in g.axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion")
        plt.subplots_adjust(bottom=0.5, hspace=0.5)

    st.pyplot(g)

    st.markdown('## RAC1P')
    rac_df = data_merged.groupby(by=['STATE'])['RAC1P'].value_counts(normalize=True).to_frame(
        'Proportion').reset_index()

    g = sns.catplot(data=rac_df, x="RAC1P", y='Proportion', col="STATE", col_wrap=7,
                    kind="bar",
                    errorbar=None,
                    height=6, palette={'White':'#476C9B','Black':'#984447','Other':'#469978'}, aspect=.4)  # 4,0.6
    for ax in g.axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion")
        plt.subplots_adjust(bottom=0.5, hspace=0.5)

    st.pyplot(g)


    # figures in relation to ESR
    st.markdown('## SEX')
    esr_df = data_merged.groupby(by=['STATE', 'ESR'])['SEX'].value_counts(normalize=True).to_frame(
        'Proportion').reset_index()
    g = sns.catplot(data=esr_df, x="ESR", y='Proportion', hue="SEX", col="STATE", col_wrap=7,
                    kind="bar",
                    errorbar=None,
                    height=6, palette={'Female': 'coral', 'Male': 'teal'}, aspect=.4)  # 4,0.6
    for ax in g.axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion")
        plt.subplots_adjust(bottom=0.5, hspace=0.5)
    st.pyplot(g)

    st.markdown('## RAC1P')
    esr_df2 = data_merged.groupby(by=['STATE', 'ESR'])['RAC1P'].value_counts(normalize=True).to_frame(
        'Proportion').reset_index()
    g = sns.catplot(data=esr_df2, x="ESR", y='Proportion', hue="RAC1P", col="STATE", col_wrap=7,
                    kind="bar",
                    errorbar=None,
                    height=6, palette={'White':'#476C9B','Black':'#984447','Other':'#469978'}, aspect=.4)  # 4,0.6
    for ax in g.axes:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("")
        ax.set_ylabel("Proportion")
        plt.subplots_adjust(bottom=0.5, hspace=0.5)
    st.pyplot(g)


    # fig = make_subplots(rows=8, cols=7, subplot_titles=states)
    #
    # for i in range(1,9,1):
    #     for j in range(1,8,1):
    #
    #         fig.add_trace(go.Bar(
    #                 x=sex_df['SEX'],
    #                 y=sex_df['Proportion'],
    #                 marker_color='indianred'),
    #                 row=i, col=j)

    # st.plotly_chart(fig)