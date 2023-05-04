#!/bin/bash

# This script submits jobs related to the master thesis project on the High Performance Computing resources
# of Freie Universitaet Berlin. Job submission is executed on HPC using a job wrapper for SLURM resource
# management system.
#
# Overview of methods:
# - sklearn: runs sklearn models on task data
# - aif: runs aif360 models on task data
# - clustering: runs clustering on task data

# PROJECT: Fair ML Thesis
# author @ Sara Bonati
#-----------------------------------------------------------------------------

# -------Directory and filename (fn) management-------------------------------
home_dir="/home/sarab23/fair_ml"
data_dir="${home_dir}/fair_ml_thesis_data"
code_dir="${home_dir}/fair_ml_thesis"
log_dir="${home_dir}/fair_ml_thesis_logs"

# ---------------------------Worfklow settings--------------------------------
method="aif"
mode="temporal"
task="ACSEmployment"

# -------Define computation parameters------------------------
n_cpus=2 # maximum number of cpus per process
mem=5GB # memory demand

if [[ "${method}" == "sklearn" ]]; then

  main_script_fn="${code_dir}/models/sklearn_models_test.py"

  for y in $(seq 2014 1 2018); do

    # Create slurm job file
    echo "#!/bin/bash" >job.slurm
    # Name job file
    echo "#SBATCH --job-name fair_ml_thesis-${method}" >>job.slurm
    # specify quality of service
    echo "#SBATCH --qos standard" >>job.slurm
    # Specify maximum run time
    echo "#SBATCH --time 23:00:00" >>job.slurm #24:00:00
    # Request cpus
    echo "#SBATCH --cpus-per-task ${n_cpus}" >>job.slurm
    # Specify RAM for operation
    echo "#SBATCH --mem ${mem}" >>job.slurm
    # Write output log to log directory
    echo "#SBATCH --output ${log_dir}/${method}/${method}_${mode}_${y}_%j_$(date '+%Y_%m-%d_%H-%M-%S').out" >>job.slurm
    # Activate virtual environment on HPC
    echo "source venv/bin/activate" >>job.slurm
    # Call main python script
    echo "python ${main_script_fn} --mode ${mode} --task ${task} --year ${y}" >>job.slurm
    echo "python ${main_script_fn} ${method} ${mode} ${y}"
    # Submit job to cluster and remove job
    echo "FAIR ML THESIS ${method} ${mode} YEAR ${y} job in queue"
    echo ""
    sbatch job.slurm
    rm -f job.slurm
  done

fi

if [[ "${method}" == "aif" ]]; then

  main_script_fn="${code_dir}/models/aif360_models_test.py"

  for y in $(seq 2014 1 2018); do

    # Create slurm job file
    echo "#!/bin/bash" >job.slurm
    # Name job file
    echo "#SBATCH --job-name fair_ml_thesis-${method}" >>job.slurm
    # specify quality of service
    echo "#SBATCH --qos standard" >>job.slurm
    # Specify maximum run time
    echo "#SBATCH --time 23:59:00" >>job.slurm #24:00:00
    # Request cpus
    echo "#SBATCH --cpus-per-task ${n_cpus}" >>job.slurm
    # Specify RAM for operation
    echo "#SBATCH --mem ${mem}" >>job.slurm
    # Write output log to log directory
    echo "#SBATCH --output ${log_dir}/${method}/${method}_${mode}_${y}_%j_$(date '+%Y_%m-%d_%H-%M-%S').out" >>job.slurm
    # Activate virtual environment on HPC
    echo "source venv/bin/activate" >>job.slurm
    # Call main python script
    echo "python ${main_script_fn} --mode ${mode} --task ${task} --year ${y}" >>job.slurm
    echo "python ${main_script_fn} ${method} ${mode} ${y}"
    # Submit job to cluster and remove job
    echo "FAIR ML THESIS ${method} ${mode} YEAR ${y} job in queue"
    echo ""
    sbatch job.slurm
    rm -f job.slurm
  done

fi



