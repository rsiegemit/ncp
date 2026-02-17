"""Basic pendulum NCP example.

Builds an NCP for a small inverted pendulum instance and prints
the number of verified cells.
"""

import torch
import ncp


def main() -> None:
    result = ncp.NCPBuilder(
        env_factory=ncp.PendulumEnv,
        state_spec=ncp.StateSpec(dim=2, wrap_dims=(0,)),
        control_spec=ncp.ControlSpec(
            dim=1,
            lower_bounds=torch.tensor([-2.0]),
            upper_bounds=torch.tensor([2.0]),
        ),
        dynamics_fn=ncp.inverted_pendulum_2d_torch,
        radius=2.0,
        epsilon=0.1,
        lipschitz=5.0,
        tau=0.5,
        dt=0.1,
        min_alpha=0.01,
        max_splits=2,
        num_samples=500,
        batch_size=200,
    ).build()

    print(f"Verified cells: {result.verified_centers.shape[0]}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")

    # Use the controller
    controller = ncp.NCPController(result)
    test_state = torch.tensor([[0.5, 0.0]])
    ctrl, tau = controller.lookup(test_state)
    print(f"Control for state {test_state.tolist()}: tau={tau.item()}")


if __name__ == "__main__":
    main()
