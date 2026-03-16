"""Heston stochastic volatility model with Euler discretization."""

import numpy as np
from numpy.typing import NDArray

from .base import Dynamics


class Heston(Dynamics):
    """Heston stochastic volatility model.

    dS = (r - q) S dt + sqrt(V) S dW_S
    dV = kappa (theta - V) dt + xi sqrt(V) dW_V
    corr(dW_S, dW_V) = rho

    Uses the full-truncation Euler scheme (Lord, Koekkoek & Van Dijk, 2010)
    for the variance process to prevent negative variance.

    Parameters
    ----------
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    V0 : float
        Initial variance.
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run variance level.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between spot and variance Brownians.
    """

    def __init__(
        self,
        r: float = 0.05,
        q: float = 0.0,
        V0: float = 0.04,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
    ):
        self.r = r
        self.q = q
        self.V0 = V0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    @property
    def n_factors(self) -> int:
        return 2

    def drift(self, t: float, S: NDArray) -> NDArray:
        return (self.r - self.q) * S

    def diffusion(self, t: float, S: NDArray) -> NDArray:
        # This returns sigma*S but sigma isn't constant in Heston.
        # For the abstract interface; actual step uses variance state.
        raise NotImplementedError("Use step() directly for Heston.")

    def step(
        self,
        t: float,
        dt: float,
        S: NDArray,
        dW: NDArray,
        **state: NDArray,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        V = state["variance"]
        dW_S = dW[:, 0]
        dW_V = dW[:, 1]

        # Full truncation: use max(V, 0) in diffusion terms
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)

        # Variance process (Euler with full truncation)
        V_new = V + self.kappa * (self.theta - V_pos) * dt + self.xi * sqrt_V * dW_V
        V_new = np.maximum(V_new, 0.0)

        # Spot process (log-Euler for positivity)
        log_drift = (self.r - self.q - 0.5 * V_pos) * dt
        S_new = S * np.exp(log_drift + sqrt_V * dW_S)

        return S_new, {"variance": V_new}

    def sample_paths(
        self,
        S0: float,
        t_grid: NDArray,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> dict[str, NDArray]:
        n_steps = len(t_grid) - 1
        dt = np.diff(t_grid)

        n_sim = n_paths // 2 if antithetic else n_paths

        # Generate correlated Brownian increments
        # Z1, Z2 independent standard normals
        Z1 = rng.standard_normal((n_sim, n_steps))
        Z2 = rng.standard_normal((n_sim, n_steps))

        if antithetic:
            Z1 = np.concatenate([Z1, -Z1], axis=0)
            Z2 = np.concatenate([Z2, -Z2], axis=0)

        actual_paths = Z1.shape[0]

        # Correlate: W_S = Z1, W_V = rho*Z1 + sqrt(1-rho^2)*Z2
        sqrt_dt = np.sqrt(dt)  # (n_steps,)
        dW_S = Z1 * sqrt_dt[np.newaxis, :]
        dW_V = (self.rho * Z1 + np.sqrt(1.0 - self.rho**2) * Z2) * sqrt_dt[np.newaxis, :]

        # Pre-allocate
        spot = np.empty((actual_paths, n_steps + 1))
        variance = np.empty((actual_paths, n_steps + 1))
        spot[:, 0] = S0
        variance[:, 0] = self.V0

        # Step through time (vectorized across paths)
        for i in range(n_steps):
            dW = np.stack([dW_S[:, i], dW_V[:, i]], axis=1)
            S_new, state_new = self.step(
                t_grid[i], dt[i], spot[:, i], dW, variance=variance[:, i]
            )
            spot[:, i + 1] = S_new
            variance[:, i + 1] = state_new["variance"]

        return {"spot": spot, "variance": variance}
