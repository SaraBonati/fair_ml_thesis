# MSc Data Science master thesis project 
# The impact of spatial and temporal context in fairness-aware  machine learning

This repository includes all scripts used in the MSc Data Science master thesis project (currently in progress).

## Virtual environment
```shell
python3 -m venv venv
source venv/bin/activate #(MacOS, Linux)
source venv/Scripts/Activate #(Windows)
python3 -m pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
```

## Repository organization
* `models` folder: includes classes to run multiple machine learning models (sklearn and aif360 models) on selected data
* `utils` folder: includes utility functions for plotting, downloading data
* `pages` folder: includes the pages making up the streamlit multipage app

## Running the app and the analysis

The American Community Survey (ACS) Public Use Microdata Sample (PUMS) files contain a wide range of both categorical and numerical features, for each US state and across multiple years ranging from 1994 to 2020. To make the data exploration process easier and more accessible, a EDA app built with the Python module `streamlit` can be used to check and plot data features.
Information on columns and recoding of columns can be found at `https://www.census.gov/programs-surveys/acs/microdata/documentation.html`

### Data download

To download the data and turn it into a task-specific dataset we use the methods in the script `utils/data_utils.py`

```python
# change to utils directory
cd utils 
# download raw ACS PUMS data
python data_utils.py --mode download
# use folktables to turn it into task specific data
python data_utils.py --mode rawdata_to_task
```

### ML

All machine learning preprocessing and analysis is conducted using `scikit-learn` and `aif360`. An app built with the Python module `streamlit` can be used to visualize the results of the nachine learning analysis.

To run the sklearn machine learning models:

```python
# change to models directory
cd models 
# run models in spatial context
python sklearn_models.py --mode spatial --task ACSEmployment
# run models in temporal context
python sklearn_models.py --mode temporal --task ACSEmployment
```

To run the aif360 machine learning models:

```python
# change to models directory
cd models 
# run models in spatial context
python aif360_models.py --mode spatial --task ACSEmployment
# run models in temporal context
python aif360_models.py --mode temporal --task ACSEmployment
```

### Streamlit
From the project home directory the streamlit app to visualize EDA plots and analysis results can be executed 
locally by using the command

```bash
streamlit run streamlit_app.py
```