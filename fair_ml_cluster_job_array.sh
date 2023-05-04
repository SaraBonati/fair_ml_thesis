#!/bin/bash

#SBATCH --job-name=fair_ml_thesis
#SBATCH --mail-user=sarab23@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --output=/home/sarab23/fair_ml/fair_ml_thesis_logs/aif/test_all_temporal.%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=10:00:00
#SBATCH --qos=standard

# declare array containing script args combinations
declare -a combinations
index=0
for mode in 'temporal' #'spatial'
do
    for train in 'AL' 'AK' 'AZ' 'AR' 'CA' 'CO' 'CT' 'DE' 'FL' 'GA' 'HI' 'ID' 'IL' 'IN' 'IA' 'KS' 'KY' 'LA' 'ME' 'MD' 'MA' 'MI' 'MN' 'MS' 'MO' 'MT' 'NE' 'NV' 'NH' 'NJ' 'NM' 'NY' 'NC' 'ND' 'OH' 'OK' 'OR' 'PA' 'RI' 'SC' 'SD' 'TN' 'TX' 'UT' 'VT' 'VA' 'WA' 'WV' 'WI' 'WY'
    do
        for year in 2014 2015 2016 2017 2018
        do
            combinations[$index]="$mode $train $year"
            index=$((index + 1))
        done
    done
done

parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})
mode=${parameters[0]}
train=${parameters[1]}
year=${parameters[2]}

# setup python and activate venv
module add Python/3.8.6-GCCcore-10.2.0
source venv/bin/activate

# run script
python models/aif360_models_cluster_new.py  --mode ${mode} --train ${train} --year ${year}