"""Comprehensive integration tests for all 12 new dynamical systems.

Each system is tested for:
  1. Env creation with correct specs
  2. Reset with deterministic seed
  3. Trajectory rollout shape and finiteness
  4. MPPI sample_trajectory shape
  5. Dynamics called through env._apply_dynamics
  6. No NaN / Inf in outputs after multiple Euler steps

Additionally, full NCPBuilder pipeline runs are tested on two
representative 2D systems (Van der Pol and Duffing).
"""

import torch
import pytest

import ncp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_env_suite(
    env_cls,
    state_dim: int,
    ctrl_dim: int,
    wrap_dims: tuple[int, ...],
    num_envs: int = 6,
    time_steps: int = 10,
    num_samples: int = 20,
    seed_scale: float = 0.5,
    **env_kwargs,
) -> None:
    """Run a standard suite of checks on an environment class."""
    env = env_cls(num_envs=num_envs, **env_kwargs)

    # --- Spec checks ---
    assert env.state_spec.dim == state_dim
    assert env.control_spec.dim == ctrl_dim
    assert env.state_spec.wrap_dims == wrap_dims
    assert env.states.shape == (num_envs, state_dim)

    # --- Reset ---
    seed = torch.randn(num_envs, state_dim) * seed_scale
    env.reset(seed)
    assert torch.equal(env.states, seed)

    # --- Trajectory rollout ---
    if ctrl_dim == 1:
        controls = torch.randn(num_envs, time_steps)
    else:
        controls = torch.randn(num_envs, time_steps, ctrl_dim)
    traj = env.trajectories(controls)
    assert traj.shape == (num_envs, time_steps + 1, state_dim)
    assert torch.isfinite(traj).all(), "Trajectory contains NaN/Inf"

    # --- MPPI sample_trajectory ---
    env.reset(torch.randn(num_envs, state_dim) * seed_scale)
    acts, sample_traj, dist, taus = env.sample_trajectory(
        time_steps=time_steps, num_samples=num_samples
    )
    if ctrl_dim == 1:
        assert acts.shape == (num_envs, time_steps)
    else:
        assert acts.shape == (num_envs, time_steps, ctrl_dim)
    assert sample_traj.shape == (num_envs, time_steps + 1, state_dim)
    assert dist.shape == (num_envs,)
    assert taus.shape == (num_envs,)
    assert torch.isfinite(sample_traj).all(), "MPPI trajectory contains NaN/Inf"
    assert torch.isfinite(dist).all(), "MPPI distances contain NaN/Inf"

    # --- Multi-step dynamics finiteness (50 Euler steps) ---
    env.reset(torch.randn(num_envs, state_dim) * seed_scale)
    x = env.states.clone()
    for _ in range(50):
        if ctrl_dim == 1:
            u = torch.zeros(num_envs, 1)
        else:
            u = torch.zeros(num_envs, ctrl_dim)
        x = env._apply_dynamics(x, u)
    assert torch.isfinite(x).all(), "State diverged to NaN/Inf in 50 zero-ctrl steps"


# ---------------------------------------------------------------------------
# Test classes: one per new system
# ---------------------------------------------------------------------------

class TestCartPoleIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.CartPoleEnv, 4, 1, ())

    def test_dynamics_kwargs_override(self) -> None:
        env = ncp.CartPoleEnv(num_envs=2, dynamics_kwargs={"m_c": 2.0, "m_p": 0.5})
        env.reset(torch.tensor([[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, -0.3, 0.0]]))
        controls = torch.zeros(2, 5)
        traj = env.trajectories(controls)
        assert torch.isfinite(traj).all()


class TestAcrobotIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.AcrobotEnv, 4, 1, (0, 1))

    def test_wrap_dims_applied_in_mppi(self) -> None:
        """sample_trajectory wraps angular dims; verify angles are in (-pi, pi]."""
        env = ncp.AcrobotEnv(num_envs=2)
        env.reset(torch.tensor([[4.0, -4.0, 0.0, 0.0], [7.0, 7.0, 0.0, 0.0]]))
        _, traj, _, _ = env.sample_trajectory(time_steps=3, num_samples=5)
        for t in range(traj.shape[1]):
            for d in [0, 1]:
                vals = traj[:, t, d]
                assert (vals > -torch.pi - 1e-6).all() and (vals <= torch.pi + 1e-6).all()


class TestMountainCarIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.MountainCarEnv, 2, 1, ())

    def test_gravity_hill_effect(self) -> None:
        """Applying force should change velocity differently at different positions."""
        env = ncp.MountainCarEnv(num_envs=2, dt=0.1)
        env.reset(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
        controls = torch.tensor([[1.0], [1.0]]).expand(2, 5)
        traj = env.trajectories(controls)
        # Velocities should differ due to cos(3*pos) term
        assert not torch.allclose(traj[0, -1, 1], traj[1, -1, 1])


class TestPendubotIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.PendubotEnv, 4, 1, (0, 1))


class TestFurutaIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.FurutaEnv, 4, 1, (0, 1), seed_scale=0.1, dt=0.005)


class TestBallBeamIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.BallBeamEnv, 4, 1, (), seed_scale=0.3)

    def test_zero_state_equilibrium(self) -> None:
        """Ball at center of level beam should be stable with no torque."""
        env = ncp.BallBeamEnv(num_envs=1)
        env.reset(torch.zeros(1, 4))
        controls = torch.zeros(1, 20)
        traj = env.trajectories(controls)
        assert torch.allclose(traj[0, -1], torch.zeros(4), atol=1e-4)


class TestQuadrotor2DIntegration:
    def test_full_suite(self) -> None:
        # Use small seed to avoid NaN from large angles
        _run_env_suite(ncp.Quadrotor2DEnv, 6, 2, (4,), seed_scale=0.3)

    def test_hover_trajectory(self) -> None:
        """Hover thrust at level attitude should keep position roughly stable."""
        m, g = 1.0, 9.81
        env = ncp.Quadrotor2DEnv(num_envs=1, mass=m, g=g, dt=0.01)
        env.reset(torch.zeros(1, 6))
        hover = torch.tensor([[[m * g / 2, m * g / 2]]]).expand(1, 50, 2)
        traj = env.trajectories(hover)
        # Should stay near origin
        assert traj[0, -1, :4].abs().max() < 0.1

    def test_custom_mass_forwarded_to_dynamics(self) -> None:
        """Non-default mass should be forwarded to dynamics_kwargs."""
        m, g = 2.0, 9.81
        env = ncp.Quadrotor2DEnv(num_envs=1, mass=m, g=g, dt=0.01)
        assert env.dynamics_kwargs["m"] == m
        assert env.dynamics_kwargs["g"] == g
        # Hover at custom mass should also stay stable
        env.reset(torch.zeros(1, 6))
        hover = torch.tensor([[[m * g / 2, m * g / 2]]]).expand(1, 50, 2)
        traj = env.trajectories(hover)
        assert traj[0, -1, :4].abs().max() < 0.1

    def test_explicit_dynamics_kwargs_not_overridden(self) -> None:
        """User-provided dynamics_kwargs should take precedence."""
        env = ncp.Quadrotor2DEnv(
            num_envs=1, mass=2.0, dynamics_kwargs={"m": 3.0}
        )
        assert env.dynamics_kwargs["m"] == 3.0  # explicit wins
        assert env.dynamics_kwargs["g"] == 9.81  # default filled in


class TestTwoLinkArmIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.TwoLinkArmEnv, 4, 2, (0, 1))


class TestCSTRIntegration:
    def test_full_suite(self) -> None:
        # CSTR needs positive concentrations and moderate temperatures
        # (high T causes thermal runaway; use near-steady conditions)
        env = ncp.CSTREnv(num_envs=4, dt=0.001)
        seed = torch.tensor([[8.0, 310.0], [9.0, 305.0], [7.5, 315.0], [8.5, 308.0]])
        env.reset(seed)
        controls = torch.zeros(4, 10)
        traj = env.trajectories(controls)
        assert traj.shape == (4, 11, 2)
        assert torch.isfinite(traj).all()

    def test_mppi(self) -> None:
        env = ncp.CSTREnv(num_envs=3, dt=0.001)
        env.reset(torch.tensor([[8.0, 310.0], [9.0, 305.0], [7.5, 315.0]]))
        acts, traj, dist, taus = env.sample_trajectory(time_steps=5, num_samples=10)
        assert acts.shape == (3, 5)
        assert torch.isfinite(traj).all()


class TestVanDerPolIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.VanDerPolEnv, 2, 1, ())

    def test_limit_cycle(self) -> None:
        """Without control, VdP should evolve and stay finite."""
        env = ncp.VanDerPolEnv(num_envs=3, dt=0.01)
        env.reset(torch.tensor([[2.0, 0.0], [0.1, 0.0], [-1.5, 1.0]]))
        controls = torch.zeros(3, 200)
        traj = env.trajectories(controls)
        assert torch.isfinite(traj).all()
        # Should not blow up — bounded limit cycle
        assert traj.abs().max() < 50.0


class TestDuffingIntegration:
    def test_full_suite(self) -> None:
        _run_env_suite(ncp.DuffingEnv, 2, 1, ())


class TestLotkaVolterraIntegration:
    def test_full_suite(self) -> None:
        # LV needs positive populations
        env = ncp.LotkaVolterraEnv(num_envs=4)
        seed = torch.tensor([[2.0, 1.0], [1.0, 2.0], [3.0, 0.5], [0.5, 3.0]])
        env.reset(seed)
        controls = torch.zeros(4, 10)
        traj = env.trajectories(controls)
        assert traj.shape == (4, 11, 2)
        assert torch.isfinite(traj).all()

    def test_mppi(self) -> None:
        env = ncp.LotkaVolterraEnv(num_envs=3)
        env.reset(torch.tensor([[2.0, 1.0], [1.5, 1.5], [1.0, 2.0]]))
        acts, traj, dist, taus = env.sample_trajectory(time_steps=5, num_samples=10)
        assert acts.shape == (3, 5)
        assert torch.isfinite(traj).all()


# ---------------------------------------------------------------------------
# Full NCPBuilder pipeline tests on 2D systems
# ---------------------------------------------------------------------------

class TestVanDerPolPipeline:
    """End-to-end: build NCP for Van der Pol, lookup controller, simulate."""

    @pytest.fixture(scope="class")
    def vdp_result(self) -> ncp.NCPResult:
        return ncp.NCPBuilder(
            env_factory=ncp.VanDerPolEnv,
            state_spec=ncp.StateSpec(dim=2, wrap_dims=()),
            control_spec=ncp.ControlSpec(
                dim=1,
                lower_bounds=torch.tensor([-5.0]),
                upper_bounds=torch.tensor([5.0]),
            ),
            dynamics_fn=ncp.van_der_pol_dynamics,
            radius=1.0,
            epsilon=0.2,
            lipschitz=5.0,
            tau=0.5,
            dt=0.05,
            min_alpha=0.01,
            max_splits=2,
            num_samples=200,
            batch_size=100,
        ).build()

    def test_verified_cells(self, vdp_result: ncp.NCPResult) -> None:
        assert vdp_result.verified_centers.shape[0] > 0
        assert vdp_result.verified_centers.shape[1] == 2

    def test_controller_lookup(self, vdp_result: ncp.NCPResult) -> None:
        controller = ncp.NCPController(vdp_result)
        states = vdp_result.verified_centers[:3]
        ctrl, taus = controller.lookup(states)
        assert ctrl.shape[0] == states.shape[0]

    def test_simulate(self, vdp_result: ncp.NCPResult) -> None:
        controller = ncp.NCPController(vdp_result)
        env = ncp.VanDerPolEnv(num_envs=2, dt=0.05, lipschitz=5.0)
        initial = vdp_result.verified_centers[:2].clone()
        sim_out = ncp.simulate(env, controller, initial, max_steps=20, precision=0.5)
        assert "trajectories" in sim_out
        assert sim_out["trajectories"].shape[0] == 2


class TestDuffingPipeline:
    """End-to-end: build NCP for Duffing oscillator."""

    @pytest.fixture(scope="class")
    def duffing_result(self) -> ncp.NCPResult:
        return ncp.NCPBuilder(
            env_factory=ncp.DuffingEnv,
            state_spec=ncp.StateSpec(dim=2, wrap_dims=()),
            control_spec=ncp.ControlSpec(
                dim=1,
                lower_bounds=torch.tensor([-10.0]),
                upper_bounds=torch.tensor([10.0]),
            ),
            dynamics_fn=ncp.duffing_dynamics,
            radius=1.0,
            epsilon=0.2,
            lipschitz=5.0,
            tau=0.5,
            dt=0.05,
            min_alpha=0.01,
            max_splits=2,
            num_samples=200,
            batch_size=100,
        ).build()

    def test_verified_cells(self, duffing_result: ncp.NCPResult) -> None:
        assert duffing_result.verified_centers.shape[0] > 0

    def test_simulate(self, duffing_result: ncp.NCPResult) -> None:
        controller = ncp.NCPController(duffing_result)
        env = ncp.DuffingEnv(num_envs=2, dt=0.05, lipschitz=5.0)
        initial = duffing_result.verified_centers[:2].clone()
        sim_out = ncp.simulate(env, controller, initial, max_steps=20, precision=0.5)
        assert sim_out["trajectories"].shape[0] == 2


class TestMountainCarPipeline:
    """End-to-end: build NCP for Mountain Car."""

    @pytest.fixture(scope="class")
    def mcar_result(self) -> ncp.NCPResult:
        return ncp.NCPBuilder(
            env_factory=ncp.MountainCarEnv,
            state_spec=ncp.StateSpec(dim=2, wrap_dims=()),
            control_spec=ncp.ControlSpec(
                dim=1,
                lower_bounds=torch.tensor([-1.0]),
                upper_bounds=torch.tensor([1.0]),
            ),
            dynamics_fn=ncp.mountain_car_dynamics,
            radius=0.5,
            epsilon=0.15,
            lipschitz=5.0,
            tau=0.5,
            dt=0.1,
            min_alpha=0.01,
            max_splits=2,
            num_samples=200,
            batch_size=100,
        ).build()

    def test_verified_cells(self, mcar_result: ncp.NCPResult) -> None:
        assert mcar_result.verified_centers.shape[0] > 0
        assert mcar_result.verified_centers.shape[1] == 2

    def test_controller(self, mcar_result: ncp.NCPResult) -> None:
        controller = ncp.NCPController(mcar_result)
        states = mcar_result.verified_centers[:2]
        ctrl, taus = controller.lookup(states)
        assert ctrl.shape[0] == 2


# ---------------------------------------------------------------------------
# Cross-system import / export test
# ---------------------------------------------------------------------------

class TestAllExports:
    """Verify every new dynamics function and env class is accessible via ncp.*."""

    DYNAMICS = [
        "cartpole_dynamics",
        "acrobot_dynamics",
        "mountain_car_dynamics",
        "pendubot_dynamics",
        "furuta_dynamics",
        "ball_beam_dynamics",
        "quadrotor_2d_dynamics",
        "two_link_arm_dynamics",
        "cstr_dynamics",
        "van_der_pol_dynamics",
        "duffing_dynamics",
        "lotka_volterra_dynamics",
    ]

    ENVS = [
        "CartPoleEnv",
        "AcrobotEnv",
        "MountainCarEnv",
        "PendubotEnv",
        "FurutaEnv",
        "BallBeamEnv",
        "Quadrotor2DEnv",
        "TwoLinkArmEnv",
        "CSTREnv",
        "VanDerPolEnv",
        "DuffingEnv",
        "LotkaVolterraEnv",
    ]

    @pytest.mark.parametrize("name", DYNAMICS)
    def test_dynamics_export(self, name: str) -> None:
        fn = getattr(ncp, name)
        assert callable(fn)

    @pytest.mark.parametrize("name", ENVS)
    def test_env_export(self, name: str) -> None:
        cls = getattr(ncp, name)
        assert issubclass(cls, ncp.BaseEnv)
