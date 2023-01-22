# This script launches a Streamlit app designed ot help in
# the exploratory data analysis (EDA) process of the ACS PUMS datasets
# for Master Thesis.
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
    preprocess_healthinsurance, make_protected_plots
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

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
)

st.markdown("""
            # EDA of dataset
            In this section of the app you can perform some exploratory data analysis (EDA) on the US Census dataset.
            """)
st.sidebar.markdown("# EDA of dataset")

#############################################
# Select task state and year
#############################################

eda_form = st.form("My form")

select_task = eda_form.selectbox('Which classification task do you want to focus on?', task_infos['task_names'])
select_state = eda_form.selectbox('Which state do you want to see?', task_infos['states'])
select_year = eda_form.selectbox('Which year do you want to see?', np.arange(min(task_infos['years']),
                                                                             max(task_infos['years'])+1))
show_all_usa = eda_form.checkbox("Show EDA also for all US states")
submitted = eda_form.form_submit_button("Submit")

if submitted:

    # load data
    data = pd.read_csv(os.path.join(ddir, str(select_year), '1-Year',
                                    f'{str(select_year)}_{select_state}_{select_task}.csv'), sep=',')
    target_name = task_infos["tasks"][task_infos["task_col_map"][select_task]]["target"]

    # map int values to string values in order to make plots more understandable
    if select_task == "ACSHealthInsurance":
        data = preprocess_healthinsurance(data)
    data["RAC1P_r"] = data['RAC1P']
    data.loc[data['RAC1P_r'] > 2, 'RAC1P_r'] = 3

    data['RAC1P'] = data['RAC1P'].map(cols_infos['RAC1P'])
    data['RAC1P_r'] = data['RAC1P_r'].map(cols_infos['RAC1P_r'])
    data['SEX'] = data['SEX'].map(cols_infos['SEX'])
    data['MAR'] = data['MAR'].map(cols_infos['MAR'])
    data['SCHL'] = data['SCHL'].map(cols_infos['SCHL'])
    data['NATIVITY'] = data['NATIVITY'].map(cols_infos['NATIVITY'])
    data[target_name] = data[target_name].map(cols_infos[target_name])
    st.dataframe(data)

    data_merged = merge_dataframes_eda([os.path.join(ddir, str(y), '1-Year',
                                                     f'{str(y)}_{select_state}_{select_task}.csv') for y in
                                        task_infos["years"]],
                                       select_task)
    data_merged["RAC1P_r"] = data_merged['RAC1P']
    data_merged.loc[data_merged['RAC1P_r'] > 2, 'RAC1P_r'] = 3
    data_merged['RAC1P'] = data_merged['RAC1P'].map(cols_infos['RAC1P'])
    data_merged['RAC1P_r'] = data_merged['RAC1P_r'].map(cols_infos['RAC1P_r'])
    data_merged['SEX'] = data_merged['SEX'].map(cols_infos['SEX'])
    data_merged['MAR'] = data_merged['MAR'].map(cols_infos['MAR'])
    data_merged['SCHL'] = data_merged['SCHL'].map(cols_infos['SCHL'])
    data_merged['NATIVITY'] = data_merged['NATIVITY'].map(cols_infos['NATIVITY'])
    data_merged[target_name] = data_merged[target_name].map(cols_infos[target_name])

    #############################################
    # state-specific metrics
    #############################################
    st.markdown(f"### Metrics of {select_state} in {select_year}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N^ of Samples",
                value=len(data),
                delta_color='off')
    col2.metric(f"% of male samples",
                value=round((len(data[data['SEX'] == 'Male']) / len(data)) * 100, 2),
                delta_color='off')
    col3.metric(f"% of white samples",
                value=round((len(data[data['RAC1P'] == 'White']) / len(data)) * 100, 2),
                delta_color='off')
    col4.metric(f"% of foreign-born samples",
                value=round((len(data[data['NATIVITY'] == 'Foreign born']) / len(data)) * 100, 2),
                delta_color='off')

    #############################################
    # state-specific protected attributes info
    #############################################

    fig1, fig2 = make_protected_plots(data, data_merged, target_name)
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

    #############################################
    # state-specific demographic info
    #############################################
    target_name = task_infos["tasks"][task_infos["task_col_map"][select_task]]["target"]
    fig1, fig2, fig3, fig4, fig5, fig6 = make_demographic_plots(data, data_merged, target_name)

    # figure 1 and figure 2
    st.markdown(
        f"""## How does the distribution of {data[target_name].unique()[0]} VS {data[target_name].unique()[1]} in {select_state} change as a function of race and sex?""")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown(f"""## How does the distribution above evolve across the years in {select_state}? (Race recoded)""")
    st.plotly_chart(fig2, use_container_width=True)
    # figure 3 and figure 4
    st.markdown(f"## What is the age distribution in {select_state} as a function of race and sex?")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f"""## How does the age distribution above evolve across the years in {select_state}? (Race recoded)""")
    st.plotly_chart(fig4, use_container_width=True)
    # figure 5 and figure 6
    st.markdown(f"## What is the education status distribution in {select_state} as a function of race and sex?")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        f"## How does the education status distribution above evolve across the years in {select_state}? (Race recoded)")
    st.plotly_chart(fig6)

    #############################################
    # ALL US states figures
    #############################################
    if show_all_usa:
        st.markdown("## USA data (all states, might take a bit to load)")
        metrics = eda_metrics_usa(select_task)

        #############################################
        # map of attribute SEX and RAC1P across US states
        #############################################
        fig2 = make_mapplot(metrics, "disp_sex", "Disparate impact ratio (SEX)", "Temporal", "coolwarm")
        fig2.update_layout(title_font_family="Helvetica",
                           title_font_size=20,
                           title_font_color="black",
                           title_x=0.45)
        st.plotly_chart(fig2)

        fig3 = make_mapplot(metrics, "disp_race", "Disparate impact ratio (RACE)", "Temporal", "coolwarm")
        fig3.update_layout(title_font_family="Helvetica",
                           title_font_size=20,
                           title_font_color="black",
                           title_x=0.45)
        st.plotly_chart(fig3)
