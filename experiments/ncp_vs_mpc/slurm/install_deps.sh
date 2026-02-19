#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sompolinsky_lab
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/n/home02/rsiegelmann/experiments/ncp_vs_mpc/logs/install_deps_%j.log

# One-time: install CasADi and ensure ncp is available
module purge
module load python
mamba deactivate
mamba activate /n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/envs/gpt

which python
python --version

pip install --user casadi

# Verify imports
export PYTHONPATH="/n/home02/rsiegelmann/ncp:$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH"
python -c "import torch; print(f'PyTorch {torch.__version__} OK, CUDA={torch.cuda.is_available()}')"
python -c "import ncp; print('NCP package OK')"
python -c "import casadi; print(f'CasADi {casadi.__version__} OK')"
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"
python -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__} OK')"

echo "Dependencies installed successfully."
