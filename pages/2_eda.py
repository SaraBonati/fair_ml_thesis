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
from utils.eda_utils import eda_metrics_usa, make_mapplot, make_demographic_plots
# general utility import
import pandas as pd
import plotly.express as px
import streamlit as st

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

st.markdown("# EDA of dataset")
st.sidebar.markdown("# EDA of dataset")

#############################################
# Select task state and year
#############################################

eda_form = st.form("My form")

select_task = eda_form.selectbox('Which classification task do you want to focus on?', task_infos['task_names'])
select_state = eda_form.selectbox('Which state do you want to see?', task_infos['states'])
select_year = eda_form.slider('Which year do you want to see?', min(task_infos['years']), max(task_infos['years']))
show_all_usa = eda_form.checkbox("Show EDA also for all US states")
submitted = eda_form.form_submit_button("Submit")

if submitted:

    data = pd.read_csv(os.path.join(ddir, str(select_year), '1-Year',
                                    f'{str(select_year)}_{select_state}_{select_task}.csv'), sep=',')
    df2 = data
    df2['RAC1P'] = df2['RAC1P'].map(cols_infos['RAC1P'])
    df2['SEX'] = df2['SEX'].map(cols_infos['SEX'])
    df2['ESR'] = df2['ESR'].map(cols_infos['ESR'])
    st.dataframe(df2)

    #############################################
    # state-specific metrics
    #############################################
    col1, col2, col3 = st.columns(3)
    col1.metric("N^ of Samples",
                value=len(data),
                delta_color='off')
    col2.metric(f"% of male samples ({select_year})",
                value=round((len(data[data['SEX'] == 'Male']) / len(data)) * 100, 2),
                delta_color='off')
    col3.metric(f"% of white samples ({select_year})",
                value=round((len(data[data['RAC1P'] == 'White']) / len(data)) * 100, 2),
                delta_color='off')

    #############################################
    # state-specific demographic info
    #############################################
    target_name = task_infos["tasks"][task_infos["task_col_map"][select_task]]["target"]
    fig1, fig2 = make_demographic_plots(df2, target_name)

    # figure 1
    st.markdown(f"""## How does the distribution of {data[target_name].unique()[0]} VS {data[target_name].unique()[1]} 
                in {select_state} change as a function of race and sex?""")
    st.plotly_chart(fig1, use_container_width=True)
    # figure 2
    st.markdown(f"## What is the age distribution in {select_state} as a function of race and sex?")
    st.plotly_chart(fig2, use_container_width=True)
    # figure 3
    st.markdown(f"## What is the education status distribution in {select_state} as a function of race and sex?")
    selected_race = st.selectbox('Select an ethnicity:', list(cols_infos['RAC1P'].values()))
    if selected_race:
        fig3 = px.histogram(df2[df2['RAC1P'] == selected_race], x="RAC1P", y="SCHL", facet_row=target_name,
                            color='SEX', color_discrete_map={'Female': 'coral', 'Male': 'teal'}, barmode='group',
                            histfunc='count', width=800, height=800)
        st.plotly_chart(fig3, use_container_width=True)

    #############################################
    # ALL US states figures
    #############################################
    if show_all_usa:
        st.markdown("## USA data (all states, might take a bit to load)")
        metrics = eda_metrics_usa(select_task)

        #############################################
        # map of attribute SEX and RAC1P across US states
        #############################################
        fig2 = make_mapplot(metrics, "disp_sex", "Disparate impact ratio (SEX)","Temporal","coolwarm")
        fig2.update_layout(title_font_family="Helvetica",
                           title_font_size=20,
                           title_font_color="black",
                           title_x=0.45)
        st.plotly_chart(fig2)

        fig3 = make_mapplot(metrics, "disp_race", "Disparate impact ratio (RACE)","Temporal","coolwarm")
        fig3.update_layout(title_font_family="Helvetica",
                           title_font_size=20,
                           title_font_color="black",
                           title_x=0.45)
        st.plotly_chart(fig3)
