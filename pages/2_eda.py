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
    preprocess_healthinsurance, categorize, map_int_to_cat, make_protected_plots
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
            # EDA of US Census dataset
            """)
st.sidebar.markdown("# EDA of dataset")
st.sidebar.markdown("In this section of the app you can perform some exploratory data analysis (EDA) on the US Census dataset. "
                    "Data visualization can be a powerful tool to discover trends in the data, and here we leverage "
                    "it to show the distributions of target classes and protected attributes for a number of US "
                    "states and across the time period between 2014 and 2018 inclusive."
                    "Focus on the disparity of privileged and unprivileged features in the protected attributes, "
                    "how the disparity changes across states and years, and how these features are represented in a "
                    "balanced manner in the target class.")

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
                                    f'{str(select_year)}_{select_state}_{select_task}.csv'), sep=',', index_col=0)
    target_name = task_infos["tasks"][task_infos["task_col_map"][select_task]]["target"]

    # map int values to string values in order to make plots more understandable
    if select_task == "ACSHealthInsurance":
        data = preprocess_healthinsurance(data)

    # keep copy for more detailed plots
    data["RAC1P_r"] = data['RAC1P']
    # recode to be only white, black or other
    data.loc[data['RAC1P'] > 2, 'RAC1P'] = 3

    # turn to int and then category (single state-year data)
    categorize(data,
               task_infos['tasks'][task_infos['task_col_map'][select_task]]['cat_columns'] + ['RAC1P_r'])
    # now turn int values to labels for the categorical and target variables
    # data_2 = map_int_to_cat(data,
    #                       cols_infos,
    #                       task_infos['tasks'][task_infos['task_col_map'][select_task]]['cat_columns']
    #                       + ['RAC1P_r'])
    data[target_name] = data[target_name].map(cols_infos[target_name])

    # show dataframe in the app
    st.markdown(f"""This is the dataframe: """)
    st.dataframe(data)

    # load all data from a state for all years
    data_merged = merge_dataframes_eda([os.path.join(ddir, str(y), '1-Year',
                                                     f'{str(y)}_{select_state}_{select_task}.csv') for y in
                                        task_infos["years"]],
                                       select_task)
    data_merged["RAC1P_r"] = data_merged['RAC1P']
    data_merged.loc[data_merged['RAC1P_r'] > 2, 'RAC1P_r'] = 3

    # categorize data merged
    categorize(data_merged,
               task_infos['tasks'][task_infos['task_col_map'][select_task]]['cat_columns'])

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
    st.markdown(f"### Dataset Metrics")
    st.markdown(f"Below you can visualize some metrics on the dataset for state {select_state} in year {select_year}. "
                f"In the first row we see "
             "the number of samples in the dataset and the proportion of samples belonging to class 0 and class 1. "
             "It is generally desirable to have a balanced target variable.")

    col1, col2, col3 = st.columns(3)
    col1.metric("N^ of Samples",
                value=len(data),
                delta_color='off')
    target_proportion = data[target_name].value_counts(normalize=True).to_dict()
    col2.metric(f"% of samples in class 0",
                value=round(target_proportion[cols_infos[target_name][0]], 4),
                delta_color='off')
    col3.metric(f"% of samples in class 1",
                value=round(target_proportion[cols_infos[target_name][1]], 4),
                delta_color='off')

    st.write("The second row focuses on the protected attributes, and shows the proportion of samples in the dataset "
             "that belong to the privileged group of the protected attribute. for the variable SEX the privileged "
             "attribute is 1 (male) and for the variable RAC1P the privileged attribute is 1 (white)")
    col4, col5, col6 = st.columns(3)
    sex_proportion = data['SEX'].value_counts(normalize=True).to_dict()
    col4.metric(f"% of male samples",
                value=round(sex_proportion[1], 4),
                delta_color='off')
    race_proportion = data['RAC1P'].value_counts(normalize=True).to_dict()
    col5.metric(f"% of white samples",
                value=round(race_proportion[1], 4),
                delta_color='off')
    nativity_proportion = data['NATIVITY'].value_counts(normalize=True).to_dict()
    col6.metric(f"% of foreign-born samples",
                value=round(nativity_proportion[2], 4),
                delta_color='off')

    #############################################
    # state-specific protected attributes info
    #############################################
    fig1, fig2, fig3 = make_protected_plots(data, data_merged, target_name)
    col1, col2 = st.columns(2)
    st.plotly_chart(fig1)
    with col1:
        st.plotly_chart(fig2)
    with col2:
        st.plotly_chart(fig3)

    #############################################
    # state-specific demographic info
    #############################################
    target_name = task_infos["tasks"][task_infos["task_col_map"][select_task]]["target"]
    fig1, fig2, fig3, fig4, fig5, fig6 = make_demographic_plots(data, data_merged, target_name)

    # figure 1 and figure 2
    st.markdown(f"### State-specific target variable and protected attributes Info")
    st.markdown(f"Now we visualize the connections between the target variable and the protected attributes. Despite "
                f"having class balance across the overall dataset, the share of samples in class 0 or class 1 may be "
                f"different between samples with privileged and unprivileged protected attribute values. In other "
                f"words, How does the distribution of {data[target_name].unique()[0]} VS {data[target_name].unique()[1]}"
                f"in {select_state} change as a function of race and sex?")

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(f"### State-specific demographic and protected attributes Info")
    st.markdown(f"Next, we visualize more closely the relationship between the demographic attributes present in the "
                f"dataset and the protected attributes.")
    st.markdown(f"""How does the distribution above evolve across the years in {select_state}? (Race recoded)""")
    st.plotly_chart(fig2, use_container_width=True)
    # figure 3 and figure 4
    st.markdown(f"What is the age distribution in {select_state} as a function of race and sex?")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f"""How does the age distribution above evolve across the years in {select_state}? (Race recoded)""")
    st.plotly_chart(fig4, use_container_width=True)
    # figure 5 and figure 6
    st.markdown(f"What is the education status distribution in {select_state} as a function of race and sex?")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown(
        f"How does the education status distribution above evolve across the years in {select_state}? (Race recoded)")
    st.plotly_chart(fig6)

    #############################################
    # ALL US states figures
    #############################################
    if show_all_usa:

        st.markdown(f"### ALL US States Metrics")
        st.markdown(f"Below you can visualize some metrics on all US states data in year {select_year}. "
                    f"The metrics displayed are fairness metrics, and indicate "
                    f"NOTE: this data might take a bit to load")

        # load data file (or calculate on spot)
        metrics = eda_metrics_usa(select_task)

        #############################################
        # map of attribute SEX and RAC1P across US states
        #############################################
        fig2 = make_mapplot(metrics, "disp_sex", "Disparate impact ratio (SEX)", "Temporal", "state_code", "peach")
        fig2.update_layout(title_font_family="Helvetica",
                           title_font_size=20,
                           title_font_color="black",
                           title_x=0.45)
        st.plotly_chart(fig2)

        fig3 = make_mapplot(metrics, "disp_race", "Disparate impact ratio (RACE)", "Temporal", "state_code", "peach")
        fig3.update_layout(title_font_family="Helvetica",
                           title_font_size=20,
                           title_font_color="black",
                           title_x=0.45)
        st.plotly_chart(fig3)
