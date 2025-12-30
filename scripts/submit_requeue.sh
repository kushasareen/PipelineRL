algo="vapo";
id=0;
hrs=12;
conf="math_trial";
# KUSHA: set your account in your .env
source .env

while getopts a:i:h:c: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
        i) id=${OPTARG};;
        h) hrs=${OPTARG};;
        c) conf=${OPTARG};;
    esac
done

echo "Starting job for $hrs hours"
echo "Starting job: $conf, $algo, $id"
NAME="${conf}_${algo}_${id}"

sbatch <<EOT
#!/bin/bash

#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --output="$SCRATCH/pipeline-rl/logs/%j_${NAME}.out"
#SBATCH --time=$hrs:00:00
#SBATCH --job-name=$NAME
#SBATCH --signal=B:TERM@120

echo "Job is running... Restart count: \${SLURM_RESTART_COUNT:-0}"

handle_timeout() {
    echo "---"
    echo "Job hitting time limit. Received SIGTERM."
    echo "Requeuing job \$SLURM_JOB_ID"
    echo "---"
    scontrol requeue "\$SLURM_JOB_ID"
    exit 0
}

trap 'handle_timeout' TERM

export HF_HOME="$SCRATCH/cache"
export NUM_GPUS=4

cd ~/PipelineRL

. tamia_activate.sh
source .env

srun --kill-on-bad-exit=1 bash scripts/run.sh -a $algo -c $conf -i $id
EOT
