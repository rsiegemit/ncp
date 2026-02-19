#!/bin/bash
# Submit Phase 2 (production) jobs with dependency chaining.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Submitting Phase 2 jobs ==="

NCP_JOB=$(sbatch --parsable phase2_ncp.sh)
echo "NCP build: job $NCP_JOB"

MPPI_JOB=$(sbatch --parsable phase2_mppi_mpc.sh)
echo "MPPI-MPC: job $MPPI_JOB"

CASADI_JOB=$(sbatch --parsable phase2_casadi_mpc.sh)
echo "CasADi-MPC: job $CASADI_JOB"

CL_JOB=$(sbatch --parsable --dependency=afterok:${NCP_JOB}:${MPPI_JOB}:${CASADI_JOB} phase2_closedloop.sh)
echo "Closed-loop: job $CL_JOB (depends on NCP, MPPI, CasADi)"

echo ""
echo "=== Phase 2 submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "After completion, run:"
echo "  python src/aggregate_results.py --phase 2"
echo "  python src/plot_results.py --phase 2"
