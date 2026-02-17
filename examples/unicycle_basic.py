"""Basic unicycle NCP example.

Builds an NCP for a small unicycle instance and prints
the number of verified cells.
"""

import torch
import ncp


def main() -> None:
    result = ncp.NCPBuilder(
        env_factory=ncp.UnicycleEnv,
        state_spec=ncp.StateSpec(dim=3, wrap_dims=(2,)),
        control_spec=ncp.ControlSpec(
            dim=2,
            lower_bounds=torch.tensor([0.0, -1.0]),
            upper_bounds=torch.tensor([1.0, 1.0]),
        ),
        dynamics_fn=ncp.unicycle_derivatives,
        radius=3.0,
        epsilon=0.1,
        lipschitz=0.05,
        tau=1.0,
        dt=0.05,
        min_alpha=0.0001,
        max_splits=2,
        num_samples=500,
        batch_size=100,
    ).build()

    print(f"Verified cells: {result.verified_centers.shape[0]}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
