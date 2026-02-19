#!/bin/bash
# Submit Phase 1 jobs with dependency chaining.
# NCP + MPPI + CasADi run in parallel; closed-loop waits for all three.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Submitting Phase 1 jobs ==="

# Submit NCP build (produces ICs + NCP results)
NCP_JOB=$(sbatch --parsable phase1_ncp.sh)
echo "NCP build: job $NCP_JOB"

# Submit MPPI-MPC (can generate its own ICs if needed, but prefers existing)
MPPI_JOB=$(sbatch --parsable phase1_mppi_mpc.sh)
echo "MPPI-MPC: job $MPPI_JOB"

# Submit CasADi-MPC
CASADI_JOB=$(sbatch --parsable phase1_casadi_mpc.sh)
echo "CasADi-MPC: job $CASADI_JOB"

# Submit closed-loop evaluation — depends on all three completing
CL_JOB=$(sbatch --parsable --dependency=afterok:${NCP_JOB}:${MPPI_JOB}:${CASADI_JOB} phase1_closedloop.sh)
echo "Closed-loop: job $CL_JOB (depends on NCP, MPPI, CasADi)"

echo ""
echo "=== Phase 1 submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "After completion, run:"
echo "  python src/aggregate_results.py --phase 1"
echo "  python src/plot_results.py --phase 1"
