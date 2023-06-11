## MSc Data Science master thesis project 

This repository includes all scripts used in the MSc Data Science master thesis project `The impact of spatial and 
temporal context in fairness-aware  machine learning` (currently in progress).

## Virtual environment
We use Python v3.8 within a virtual environment will all needed dependencies installed.
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
* `notebooks` folder: includes Jupyter notebooks used to create general figures and tables for the thesis document
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
# after download, use folktables to turn it into task specific data
python data_utils.py --mode rawdata_to_task
```

### ML

All preprocessing and analysis is conducted using `scikit-learn` and `aif360`. An app built with 
the Python module `streamlit` can be used to visualize the results of the machine learning analysis. Both analysis 
scripts are adapted to be run locally or on a cluster (in this case the FU Berlin HPC cluster).

To run the sklearn machine learning models locally:

```python
# change to models directory
cd models 
# run models in spatial context
python sklearn_models.py --mode spatial --task ACSEmployment --year 2014
# run models in temporal context
python sklearn_models.py --mode temporal --task ACSEmployment --year 2014
```

To run the aif360 machine learning models locally:

```python
# change to models directory
cd models 
# run models in spatial context
python aif360_models.py --mode spatial --task ACSEmployment --year 2014
# run models in temporal context
python aif360_models.py --mode temporal --task ACSEmployment --year 2014
```

To run the scripts in an HPC cluster setting use the bash script `fair_ml_cluster_job_array.sh` (for normal analyses)
. Next, the script and the sampled results be used in the analyses that use domain knowledge; these can be launched 
using the bash script `fair_ml_cluster_job_array_sampled.sh`. The scripts prepare a job array, submitting one 
slurm job for each combination of arguments in the script. Note that the scripts have to be modified accordingly 
depending on whether one wants to run sklearn or aif360 models, and depending on the type of analysis (spatial or 
temporal). 


```bash
# load Python module in cluster
module add Python/3.8.6-GCCcore-10.20.0
# change to repo directory (must clone or add scripts into cluster first)
cd fair_ml_thesis 
# run bash script that creates a .slurm job file and submits jobs to slurm scheduler
# (for normal analyses)
sbatch --array=1-n fair_ml_cluster_job_array.sh
# (to create sampled data with domain knowledge)
sbatch --array=1-n sample_cluster_job_array.sh
# (for analyses using domain knowledge)
sbatch --array=1-n fair_ml_cluster_job_array_sampled.sh
```


### Streamlit
From the project home directory the streamlit app to visualize EDA plots and analysis results can be executed 
locally by using the command

```bash
streamlit run streamlit_app.py
```