#!/bin/bash
#SBATCH --job-name=p1_ncp
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=/n/home02/rsiegelmann/experiments/ncp_vs_mpc/logs/phase1/ncp_%a_%j.log

module purge
module load python
mamba deactivate
mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/gpt

echo "Phase 1 NCP Build: task ${SLURM_ARRAY_TASK_ID}, node ${SLURM_NODELIST}"
which python
nvidia-smi

cd /n/home02/rsiegelmann/experiments/ncp_vs_mpc

export PYTHONPATH="/n/home02/rsiegelmann/ncp:$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/run_ncp.py --index ${SLURM_ARRAY_TASK_ID} --phase 1
