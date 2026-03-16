"""Geometric Brownian Motion (Black-Scholes dynamics) with exact sampling."""

import numpy as np
from numpy.typing import NDArray

from .base import Dynamics


class GBM(Dynamics):
    """Geometric Brownian Motion: dS = (r - q) S dt + sigma S dW.

    Uses exact log-normal sampling (not Euler), so there is zero
    discretization error regardless of time step size.

    Parameters
    ----------
    r : float
        Risk-free rate (annualized, continuous compounding).
    q : float
        Continuous dividend yield.
    sigma : float
        Constant volatility.
    """

    def __init__(self, r: float = 0.05, q: float = 0.0, sigma: float = 0.20) -> None:
        self.r = r
        self.q = q
        self.sigma = sigma

    def drift(self, t: float, S: NDArray) -> NDArray:
        return (self.r - self.q) * S

    def diffusion(self, t: float, S: NDArray) -> NDArray:
        return self.sigma * S

    def step(
        self,
        t: float,
        dt: float,
        S: NDArray,
        dW: NDArray,
        **state: NDArray,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        # Exact log-normal step: S(t+dt) = S(t) * exp((r-q-0.5*σ²)dt + σ*dW)
        drift_term = (self.r - self.q - 0.5 * self.sigma**2) * dt
        S_new = S * np.exp(drift_term + self.sigma * dW)
        return S_new, {}

    def sample_paths(
        self,
        S0: float,
        t_grid: NDArray,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> dict[str, NDArray]:
        n_steps = len(t_grid) - 1
        dt = np.diff(t_grid)  # (n_steps,)

        # Number of base paths to simulate before optional antithetic pairing.
        n_sim = (n_paths + 1) // 2 if antithetic else n_paths

        # Generate all increments at once: (n_sim, n_steps)
        Z = rng.standard_normal((n_sim, n_steps))

        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)[:n_paths]

        # Exact log-normal: log(S(t+dt)/S(t)) = (r-q-σ²/2)dt + σ√dt Z
        drift_inc = (self.r - self.q - 0.5 * self.sigma**2) * dt  # (n_steps,)
        diff_inc = self.sigma * np.sqrt(dt) * Z  # (n_paths, n_steps)
        log_increments = drift_inc[np.newaxis, :] + diff_inc

        # Cumulative sum of log increments, prepend 0 for t=0
        log_S = np.concatenate(
            [np.zeros((log_increments.shape[0], 1)), np.cumsum(log_increments, axis=1)],
            axis=1,
        )
        paths = S0 * np.exp(log_S)  # (n_paths, n_steps+1)

        return {"spot": paths}
