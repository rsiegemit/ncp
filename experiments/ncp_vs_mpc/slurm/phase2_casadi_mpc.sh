#!/bin/bash
#SBATCH --job-name=p2_casadi
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-14
#SBATCH --output=/n/home02/rsiegelmann/experiments/ncp_vs_mpc/logs/phase2/casadi_%a_%j.log

module purge
module load python
mamba deactivate
mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/gpt

echo "Phase 2 CasADi-MPC: task ${SLURM_ARRAY_TASK_ID}, node ${SLURM_NODELIST}"
which python

python -c "import casadi" 2>/dev/null || pip install --user casadi

cd /n/home02/rsiegelmann/experiments/ncp_vs_mpc

export PYTHONPATH="/n/home02/rsiegelmann/ncp:$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH"

python src/run_casadi_mpc.py --index ${SLURM_ARRAY_TASK_ID} --phase 2
