#!/bin/bash

#SBATCH --job-name=new_clustering
#SBATCH --mail-user=sarab23@zedat.fu-berlin.de
#SBATCH --mail-type=end
#SBATCH --output=/home/sarab23/fair_ml/fair_ml_thesis_logs/sampling/new_clustering-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --qos=standard

declare -a combinations
index=0
for s in 'AL' 'AK' 'AZ' 'AR' 'CA' 'CO' 'CT' 'DE' 'FL' 'GA' 'HI' 'ID' 'IL' 'IN' 'IA' 'KS' 'KY' 'LA' 'ME' 'MD' 'MA' 'MI' 'MN' 'MS' 'MO' 'MT' 'NE' 'NV' 'NH' 'NJ' 'NM' 'NY' 'NC' 'ND' 'OH' 'OK' 'OR' 'PA' 'RI' 'SC' 'SD' 'TN' 'TX' 'UT' 'VT' 'VA' 'WA' 'WV' 'WI' 'WY'
do
    for y in 2014 2015 2016 2017 2018
    do

        combinations[$index]="$s $y"
        index=$((index + 1))

    done
done

parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})

state=${parameters[0]}
year=${parameters[1]}


# setup python and activate venv
module add Python/3.8.6-GCCcore-10.2.0
source venv/bin/activate

# run script
python models/sample.py  --state ${state} --year ${year}