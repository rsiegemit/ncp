"""JSON config loader with phase merge support."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class NCPConfig:
    radius: float = 1.0
    epsilon: float = 0.2
    lipschitz: float = 5.0
    tau: float = 0.5
    dt: float = 0.1
    min_alpha: float = 0.01
    max_splits: int = 2
    num_samples: int = 200
    batch_size: int = 100
    reuse: bool = False


@dataclass
class MPCConfig:
    horizon_steps: int | None = None  # Derived from tau/dt if None
    num_samples: int = 1000  # MPPI samples (GPU)
    sigma: list[float] = field(default_factory=lambda: [1.0])  # per-control noise std
    temperature: float = 1.0  # MPPI temperature (lambda)
    Q_diag: list[float] = field(default_factory=lambda: [1.0])  # per-state cost weights
    R_diag: list[float] = field(default_factory=lambda: [0.01])  # per-control cost weights


@dataclass
class CasADiConfig:
    horizon_steps: int | None = None  # Derived from tau/dt if None
    Q_diag: list[float] = field(default_factory=lambda: [1.0])  # per-state cost weights
    R_diag: list[float] = field(default_factory=lambda: [0.01])  # per-control cost weights
    terminal_weight: float = 10.0
    max_iter: int = 200
    tol: float = 1e-6


@dataclass
class SimConfig:
    max_steps: int = 100
    precision: float = 0.05


@dataclass
class ICConfig:
    grid_per_dim: int = 5
    num_random: int = 10
    seed: int = 42


@dataclass
class SystemConfig:
    system_name: str = ""
    ncp: NCPConfig = field(default_factory=NCPConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    casadi: CasADiConfig = field(default_factory=CasADiConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    ic: ICConfig = field(default_factory=ICConfig)

    def __post_init__(self):
        # Derive horizon_steps from tau/dt if not explicitly set
        time_steps = int(self.ncp.tau / self.ncp.dt)
        if self.mpc.horizon_steps is None:
            self.mpc.horizon_steps = time_steps
        if self.casadi.horizon_steps is None:
            self.casadi.horizon_steps = time_steps


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_config(d: dict) -> SystemConfig:
    """Convert nested dict to SystemConfig."""
    ncp_d = d.get("ncp", {})
    mpc_d = d.get("mpc", {})
    casadi_d = d.get("casadi", {})
    sim_d = d.get("sim", {})
    ic_d = d.get("ic", {})

    return SystemConfig(
        system_name=d.get("system_name", ""),
        ncp=NCPConfig(**ncp_d),
        mpc=MPCConfig(**mpc_d),
        casadi=CasADiConfig(**casadi_d),
        sim=SimConfig(**sim_d),
        ic=ICConfig(**ic_d),
    )


def load_config(system_name: str, phase: int) -> SystemConfig:
    """Load system base config and merge with phase overrides."""
    base_path = CONFIG_DIR / "systems" / f"{system_name}.json"
    phase_path = CONFIG_DIR / f"phase{phase}" / f"{system_name}.json"

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path) as f:
        base = json.load(f)

    if phase_path.exists():
        with open(phase_path) as f:
            override = json.load(f)
        merged = _deep_merge(base, override)
    else:
        merged = base

    merged["system_name"] = system_name
    return _dict_to_config(merged)
