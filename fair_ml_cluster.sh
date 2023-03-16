#!/bin/bash

# This script submits jobs related to the master thesis project on the High Performance Computing resources
# of Freie Universitaet Berlin. Job submission is executed on HPC using a job wrapper for SLURM resource
# management system.
#
# Overview of methods:
# - sklearn: runs sklearn models on task data
# - aif: runs aif360 models on task data
# - clustering: runs clustering on task data
# - sklearn2: runs sklearn models on task data using clustering info
# - aif2: runs aif360 models on task data using clustering info
#
# PROJECT: Fair ML Thesis
# author @ Sara Bonati
#-----------------------------------------------------------------------------

# -------Directory and filename (fn) management-------------------------------
home_dir="/home/sarab23"
project_dir="${home_dir}/fair_ml"
data_dir="${project_dir}/fair_ml_thesis_data"
code_dir="${project_dir}/fair_ml_thesis"
log_dir="${project_dir}/fair_ml_thesis_logs"

# ---------------------------Worfklow settings--------------------------------
method="clustering"
task="ACSEmployment"

# -------Define computation parameters------------------------
n_cpus=2 # maximum number of cpus per process
mem=20GB # memory demand

if [[ "${method}" == "sklearn" ]]; then

  main_script_fn="${code_dir}/models/sklearn_models.py"

  # Create slurm job file
  echo "#!/bin/bash" >job.slurm
  # Name job file
  echo "#SBATCH --job-name fair_ml_thesis-${method}" >>job.slurm
  # Specify maximum run time
  echo "#SBATCH --time 20:00:00" >>job.slurm #24:00:00
  # Request cpus
  echo "#SBATCH --cpus-per-task ${n_cpus}" >>job.slurm
  # Specify RAM for operation
  echo "#SBATCH --mem ${mem}" >>job.slurm
  # Write output log to log directory
  echo "#SBATCH --output ${log_dir}/${method}/${method}_%j_$(date '+%Y_%m-%d_%H-%M-%S').out" >>job.slurm
  # Start in current directory:
  echo "#SBATCH --workdir ${code_dir}" >>job.slurm
  # Activate virtual environment on HPC
  echo "source venv/bin/activate" >>job.slurm
  # Call main python script
  echo "python3 ${main_script_fn} ${method}" >>job.slurm
  echo "python3 ${main_script_fn} ${method}"
  # Submit job to cluster and remove job
  echo "FAIR ML THESIS ${method} job in queue"
  echo ""
  sbatch job.slurm
  rm -f job.slurm

fi



