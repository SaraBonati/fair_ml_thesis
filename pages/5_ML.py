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
csv_results_dir = r'C:\Users\sarab\Desktop\csv_results'

#############################################
# App definition
#############################################

st.set_page_config(
    page_title="ML",
    page_icon="🤖",
    layout="wide"
)

st.markdown("# ML Analysis Results")
st.sidebar.markdown("# ML Analysis Results")
st.sidebar.markdown("In this page we show the results of the machine learning analysis applied to the classification "
                    "task "
            "across the spatial and temporal context. The goal is to compare the performance of normal classifiers, "
            "that do not receive constraints over the protected attributes VS the performance of fairness-aware "
            "classifiers.")

####################
task = 'ACSEmployment'
clfiers=['LinearSVC','LogReg','XGBoost','AdversarialDebiasing']
analysis_dict={'Yes': '(with domain knowledge)' , "No": '(without domain knowledge)'}

ml_form = st.form("ML Results form")
select_context = ml_form.selectbox('Which context do you want to focus on?', ['spatial', 'temporal'])
select_metric = ml_form.selectbox('Which metric do you want to visualize?', ['median_accuracy',
                                                                             'median_sex_dpd',
                                                                             'median_rac_dpd',
                                                                             'median_sex_eod',
                                                                             'median_rac_eod'])
ml_form_submitted = ml_form.form_submit_button("Submit")

if ml_form_submitted:
    if select_context == 'spatial':

        results_spatial1 = pd.read_csv(os.path.join(csv_results_dir,'spatial_results_normal.csv'),sep=',',header=0)
        results_spatial1['analysis'] = 'no_sampling'
        results_spatial2 = pd.read_csv(os.path.join(csv_results_dir,'spatial_results_sampling.csv'),sep=',',header=0)
        results_spatial2['analysis'] = 'sampling'
        results_spatial = pd.concat([results_spatial1, results_spatial2],ignore_index=True)

        # if select_analysis == 'No':
        #     results_spatial = pd.read_csv(os.path.join(csv_results_dir,'spatial_results_normal.csv'),sep=',',header=0)
        # else:
        #     results_spatial = pd.read_csv(os.path.join(csv_results_dir,'spatial_results_sampling.csv'),sep=',',header=0)

        st.markdown(f'## {select_metric}')
        g = px.box(results_spatial, x="classifier", y=select_metric, points="all", facet_row='year', color='analysis',
                   hover_data=['state',select_metric], height=1300)
        st.plotly_chart(g,use_container_width=True,height=1300)

    if select_context == 'temporal':

        results_temporal1 = pd.read_csv(os.path.join(csv_results_dir, 'results_temporal_normal.csv'), sep=',',
                                   header=0)
        results_temporal1['analysis'] = 'no_sampling'
        results_temporal2 = pd.read_csv(os.path.join(csv_results_dir, 'results_temporal_sampling.csv'), sep=',',
                                       header=0)
        results_temporal2['analysis'] = 'sampling'

        results_temporal = pd.concat([results_temporal1, results_temporal2], ignore_index=True)
        plot_results = results_temporal[~results_temporal['classifier'].isin([
            'ExponentiatedGradientReduction'])].groupby \
            (by=['analysis',"train_year", "classifier", "train_state"]).agg(median_accuracy=('accuracy', 'median'),
                                                                 median_sex_dpd=('sex_dpd','median'),
                                                                 median_rac_dpd=('rac_dpd','median'),
                                                                 median_sex_eod=('sex_eod','median'),
                                                                 median_rac_eod=('rac_eod','median')).reset_index()


        g = px.box(plot_results,
                   x="train_year",
                    y=select_metric,
                    color='classifier',
                    facet_row="analysis",
                    hover_data=['train_state',select_metric],
                    height=700
                    )
        st.plotly_chart(g, use_container_width=True, height=700)



        results_temporal20141 = pd.read_csv(os.path.join(csv_results_dir, 'results_temporal2014_normal.csv'),
                                           sep=',',
                                           header=0)
        results_temporal20141['analysis'] = 'no_sampling'

        results_temporal20142 = pd.read_csv(os.path.join(csv_results_dir, 'results_temporal2014_sampling.csv'),
                                                 sep=',',
                                                header=0)
        results_temporal20142['analysis'] = 'no_sampling'
        results_temporal2014 = pd.concat([results_temporal20141, results_temporal20142], ignore_index=True)
