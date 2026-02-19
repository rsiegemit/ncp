"""Compute exponential convergence rates from closed-loop trajectory data.

For each converged trajectory, fit alpha such that ||x(t)|| <= e^{-alpha*t} ||x(0)||.
We estimate alpha via least-squares fit of log(||x(t)||) = log(||x(0)||) - alpha*t.
Higher alpha = faster exponential convergence.
"""
import json, torch, numpy as np
from pathlib import Path

cl_dir = Path("results/phase1/closed_loop")
controllers = ["ncp_native", "ncp_step", "ncp_nn_native", "ncp_nn_step", "mppi_mpc", "casadi_mpc"]
systems = ["pendulum", "van_der_pol", "duffing", "mountain_car"]

# Load dt for each system
dts = {}
for sys_name in systems:
    with open("configs/systems/{}.json".format(sys_name)) as f:
        cfg = json.load(f)
    dts[sys_name] = cfg['ncp']['dt']

for sys_name in systems:
    dt = dts[sys_name]
    print()
    print("=" * 110)
    print("  {} (dt={})  --  Exponential convergence rate: ||x(t)|| <= e^{{-alpha*t}} ||x(0)||".format(
        sys_name.upper(), dt))
    print("=" * 110)
    fmt_h = "  {:<18s} {:>7s} {:>8s} {:>9s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}"
    print(fmt_h.format("Controller", "Conv%", "a_mean", "a_median", "a_p25", "a_p75", "a_min",
                        "||x0||avg", "||xf||avg"))
    print("  " + "-" * 105)

    for ctrl in controllers:
        raw_path = cl_dir / "{}_{}_raw.pt".format(sys_name, ctrl)
        met_path = cl_dir / "{}_{}_metrics.json".format(sys_name, ctrl)
        if not raw_path.exists():
            print("  {:<18s} {:>7s}".format(ctrl, "pending"))
            continue

        with open(met_path) as f:
            m = json.load(f)
        data = torch.load(raw_path, weights_only=False)
        trajs = data['trajectories']
        converged = data['converged']

        alphas = []  # per-trajectory exponential rate
        x0_norms = []
        xf_norms = []

        for i, traj in enumerate(trajs):
            norms = traj.norm(dim=-1).numpy()
            x0_norms.append(norms[0])
            xf_norms.append(norms[-1])

            if not converged[i]:
                continue
            if len(norms) < 3 or norms[0] < 1e-6:
                continue

            # Time array
            t = np.arange(len(norms)) * dt

            # Filter to points where norm > 0 (avoid log(0))
            mask = norms > 1e-8
            if mask.sum() < 2:
                continue

            log_norms = np.log(norms[mask])
            t_valid = t[mask]

            # Least-squares fit: log(||x||) = b - alpha * t
            # Using polyfit degree 1: coeffs[0] = -alpha, coeffs[1] = b
            coeffs = np.polyfit(t_valid, log_norms, 1)
            alpha_fit = -coeffs[0]  # negate slope to get positive decay rate

            alphas.append(alpha_fit)

        if alphas:
            a = np.array(alphas)
            a_mean = np.mean(a)
            a_med = np.median(a)
            a_p25 = np.percentile(a, 25)
            a_p75 = np.percentile(a, 75)
            a_min = np.min(a)
        else:
            a_mean = a_med = a_p25 = a_p75 = a_min = float('nan')

        conv_pct = m['convergence_rate'] * 100
        x0_avg = np.mean(x0_norms)
        xf_avg = np.mean(xf_norms)

        fmt_r = "  {:<18s} {:>6.1f}% {:>8.4f} {:>9.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.3f} {:>10.4f}"
        print(fmt_r.format(ctrl, conv_pct, a_mean, a_med, a_p25, a_p75, a_min, x0_avg, xf_avg))

print()
print("=" * 110)
print("alpha > 0 = converging exponentially. Higher = faster. Units: 1/s (continuous time).")
print("Fit via least-squares on log(||x(t)||) vs t, only for converged trajectories.")
