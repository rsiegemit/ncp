"""Integration and edge-case tests for the full NCP pipeline."""

import torch
import pytest

import ncp


class TestPendulumPipeline:
    """End-to-end test: build NCP -> controller -> simulate."""

    @pytest.fixture(scope="class")
    def pendulum_result(self) -> ncp.NCPResult:
        return ncp.NCPBuilder(
            env_factory=ncp.PendulumEnv,
            state_spec=ncp.StateSpec(dim=2, wrap_dims=(0,)),
            control_spec=ncp.ControlSpec(
                dim=1,
                lower_bounds=torch.tensor([-2.0]),
                upper_bounds=torch.tensor([2.0]),
            ),
            dynamics_fn=ncp.inverted_pendulum_2d_torch,
            radius=1.0,
            epsilon=0.2,
            lipschitz=5.0,
            tau=0.5,
            dt=0.1,
            min_alpha=0.01,
            max_splits=2,
            num_samples=200,
            batch_size=100,
        ).build()

    def test_build_produces_verified_cells(
        self, pendulum_result: ncp.NCPResult
    ) -> None:
        r = pendulum_result
        assert r.verified_centers.shape[0] > 1
        assert r.verified_centers.shape[1] == 2
        assert r.verified_radii.shape[0] == r.verified_centers.shape[0]
        assert r.verified_controls.shape[0] == r.verified_centers.shape[0]
        assert r.verified_alphas.shape[0] == r.verified_centers.shape[0]
        assert r.verified_indices.shape[0] == r.verified_centers.shape[0]

    def test_controller_lookup(self, pendulum_result: ncp.NCPResult) -> None:
        controller = ncp.NCPController(pendulum_result)
        states = pendulum_result.verified_centers[1:4]
        ctrl, taus = controller.lookup(states)
        assert ctrl.shape[0] == states.shape[0]
        assert all(t > 0 for t in taus.tolist())

    def test_controller_outside_point(
        self, pendulum_result: ncp.NCPResult
    ) -> None:
        controller = ncp.NCPController(pendulum_result)
        outside = torch.tensor([[100.0, 100.0]])
        ctrl, tau = controller.lookup(outside)
        assert tau.item() == 0

    def test_simulate_runs(self, pendulum_result: ncp.NCPResult) -> None:
        controller = ncp.NCPController(pendulum_result)
        env = ncp.PendulumEnv(num_envs=2, dt=0.1, lipschitz=5.0)
        initial = pendulum_result.verified_centers[1:3].clone()
        sim_out = ncp.simulate(env, controller, initial, max_steps=20, precision=0.5)
        assert "trajectories" in sim_out
        assert "converged" in sim_out
        assert "steps" in sim_out
        assert sim_out["trajectories"].shape[0] == 2


class TestUnicyclePipeline:
    """End-to-end test for the 3-D unicycle system."""

    @pytest.fixture(scope="class")
    def unicycle_result(self) -> ncp.NCPResult:
        return ncp.NCPBuilder(
            env_factory=ncp.UnicycleEnv,
            state_spec=ncp.StateSpec(dim=3, wrap_dims=(2,)),
            control_spec=ncp.ControlSpec(
                dim=2,
                lower_bounds=torch.tensor([0.0, -1.0]),
                upper_bounds=torch.tensor([1.0, 1.0]),
            ),
            dynamics_fn=ncp.unicycle_derivatives,
            radius=1.0,
            epsilon=0.2,
            lipschitz=0.05,
            tau=1.0,
            dt=0.1,
            min_alpha=0.0001,
            max_splits=2,
            num_samples=200,
            batch_size=50,
        ).build()

    def test_build_shapes(self, unicycle_result: ncp.NCPResult) -> None:
        r = unicycle_result
        assert r.verified_centers.shape[1] == 3
        assert r.verified_controls.dim() == 3
        assert r.verified_controls.shape[2] == 2

    def test_controller_multidim_control(
        self, unicycle_result: ncp.NCPResult
    ) -> None:
        controller = ncp.NCPController(unicycle_result)
        states = unicycle_result.verified_centers[1:3]
        ctrl, taus = controller.lookup(states)
        assert ctrl.dim() == 3
        assert ctrl.shape[2] == 2


class TestCustomHighDimEnv:
    """Test that the framework works with arbitrary dynamics and no wrapping."""

    def test_4d_damped_linear(self) -> None:
        def damped_linear(
            x: torch.Tensor, u: torch.Tensor
        ) -> torch.Tensor:
            return -0.5 * x + u

        result = ncp.NCPBuilder(
            env_factory=lambda **kw: ncp.BaseEnv(
                state_spec=ncp.StateSpec(dim=4, wrap_dims=()),
                control_spec=ncp.ControlSpec(
                    dim=4,
                    lower_bounds=-torch.ones(4),
                    upper_bounds=torch.ones(4),
                ),
                dynamics_fn=damped_linear,
                **kw,
            ),
            state_spec=ncp.StateSpec(dim=4, wrap_dims=()),
            control_spec=ncp.ControlSpec(
                dim=4,
                lower_bounds=-torch.ones(4),
                upper_bounds=torch.ones(4),
            ),
            dynamics_fn=damped_linear,
            radius=0.5,
            epsilon=0.2,
            lipschitz=1.0,
            tau=1.0,
            dt=0.1,
            min_alpha=0.001,
            max_splits=1,
            num_samples=100,
            batch_size=200,
            angle_bound_dims=(),
        ).build()

        assert result.verified_centers.shape[1] == 4
        assert result.verified_controls.shape[2] == 4


class TestEdgeCases:
    """Edge case and correctness tests."""

    def test_wrap_angles_at_boundary(self) -> None:
        x = torch.tensor([[3.14, 0.5], [-3.14, -0.5], [6.28, 0.0]])
        wrapped = ncp.wrap_angles(x, dims=(0,))
        for i in range(x.shape[0]):
            v = wrapped[i, 0].item()
            assert -torch.pi < v <= torch.pi
            # Dimension 1 should be untouched
            assert wrapped[i, 1].item() == x[i, 1].item()

    def test_wrap_angles_no_dims(self) -> None:
        x = torch.tensor([[4.0, 5.0, 6.0]])
        wrapped = ncp.wrap_angles(x, dims=())
        assert torch.equal(wrapped, x)

    def test_subdivision_volume_3d(self) -> None:
        c = torch.tensor([[0.0, 0.0, 0.0]])
        r = torch.tensor([0.9])
        new_c, new_r = ncp.subdivide_cells(c, r, 3)
        assert new_c.shape == (27, 3)
        parent_vol = (2 * 0.9) ** 3
        child_vol = 27 * (2 * 0.3) ** 3
        assert abs(parent_vol - child_vol) < 1e-6

    def test_grid_1d_through_5d(self) -> None:
        for dim in [1, 2, 3, 4, 5]:
            pts, rad = ncp.generate_initial_grid(dim, 0.1, 0.5)
            if pts.dim() == 2:
                assert pts.shape[1] == dim
            assert len(rad) == len(pts)
            # No origin point
            if pts.dim() == 2:
                assert pts.abs().sum(dim=1).min() > 0

    def test_intersections_with_multiple_wrap_dims(self) -> None:
        c = torch.tensor([[3.0, -3.0, 0.0]])
        r = torch.tensor([0.2, 0.2, 0.2])
        qc = torch.tensor([[-3.0, 3.0, 0.0]])
        qr = torch.tensor([0.2, 0.2, 0.2])
        count_no = ncp.count_hypercube_intersections(c, r, qc, qr, wrap_dims=())
        count_yes = ncp.count_hypercube_intersections(
            c, r, qc, qr, wrap_dims=(0, 1)
        )
        assert count_no.item() == 0
        assert count_yes.item() == 1

    def test_eval_weights_vs_inf_norm(self) -> None:
        env_w = ncp.PendulumEnv(
            num_envs=1, dt=0.1, eval_weights=torch.tensor([1.0, 0.01])
        )
        d_weighted = env_w._distance(torch.tensor([[1.0, 10.0]]))
        env_inf = ncp.PendulumEnv(num_envs=1, dt=0.1)
        d_inf = env_inf._distance(torch.tensor([[1.0, 10.0]]))
        assert d_weighted.item() < d_inf.item()

    def test_find_path_mppi(self) -> None:
        env = ncp.PendulumEnv(num_envs=5, dt=0.1, lipschitz=5.0)
        seeds = torch.randn(5, 2) * 0.5
        winner, iters, taus = ncp.find_path(
            env,
            seeds=seeds,
            countermax=2,
            num_samples=50,
            time_steps=5,
            r=torch.ones(5) * 0.1,
        )
        assert winner.shape == (5, 5)
        assert iters.shape == (5,)
        assert taus.shape == (5,)

    def test_find_hypercube_intersections_structure(self) -> None:
        c = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        r = torch.tensor([0.5, 0.5, 0.5])
        qc = torch.tensor([[0.2, 0.0], [5.0, 5.0]])
        qr = torch.tensor([0.1, 0.1])
        lists = ncp.find_hypercube_intersections(c, r, qc, qr)
        assert len(lists) == 2
        assert len(lists[0]) >= 1
        assert len(lists[1]) == 0

    def test_bicycle_env_sample_trajectory(self) -> None:
        env = ncp.BicycleEnv(num_envs=5, dt=0.05, max_speed=1.0, max_steer=0.4)
        env.reset(torch.randn(5, 3) * 0.5)
        acts, traj, dist, taus = env.sample_trajectory(
            time_steps=10, num_samples=20
        )
        assert acts.shape == (5, 10, 2)
        assert traj.shape == (5, 11, 3)


class TestVisualization:
    """Smoke tests for visualization (non-interactive backend)."""

    @pytest.fixture(autouse=True)
    def _set_backend(self) -> None:
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_cells(self) -> None:
        centers = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
        radii = torch.tensor([0.2, 0.1])
        ax = ncp.plot_cells(centers, radii, xlim=(-1, 1), ylim=(-1, 1))
        assert ax is not None

    def test_plot_trajectories(self) -> None:
        traj = torch.randn(5, 10, 2)
        ax = ncp.plot_trajectories(traj, max_lines=3)
        assert ax is not None

    def test_plot_cells_custom_dims(self) -> None:
        centers = torch.randn(10, 4)
        radii = torch.rand(10) * 0.1
        ax = ncp.plot_cells(
            centers,
            radii,
            dims=(1, 3),
            pi_ticks=(False, False),
            title="Test",
        )
        assert ax is not None
