import streamlit as st

st.set_page_config(
    page_title="The impact of spatial and temporal context in fairness-aware machine learning",
    page_icon="🏠",
)

st.markdown("# The impact of spatial and temporal context in fairness-aware machine learning")

st.markdown("""
Welcome!

This in an interactive application developed in Python using streamlit as part of the master thesis **The impact of 
spatial and temporal context in fairness-aware machine learning**.

This application provides different pages, listed on the sidebar to the left, to explore graphical and tabular 
data visualizations. Specifically:

- `eda` is a page to perform exploratory data analysis and visualize data for a single US state in the Census data
- `ml` is a page to explore the results obtained by traditional and fairness aware machine learning models when using a specifified US state or survey year in training
- `EDA` is a page created for the Colloquium presentation, and given a survey year presents some visualizations for all US states 
- `ML` is a page created for the Colloquium presentation, and displays a graphical overview of the median results of the machine learning models tested across all US states and different survey years


""")
