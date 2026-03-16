"""Abstract base class for spot price dynamics (SDE specifications)."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Dynamics(ABC):
    """Base class for stochastic dynamics.

    Every dynamics object specifies:
    - drift and diffusion coefficients of the SDE
    - a step method (Euler or exact) for path generation
    - a vectorized sample_paths method that produces (n_paths, n_steps+1) arrays

    Additional state variables (e.g. variance in Heston) are carried as
    extra arrays alongside the spot path.
    """

    @abstractmethod
    def drift(self, t: float, S: NDArray) -> NDArray:
        """Instantaneous drift coefficient.

        Parameters
        ----------
        t : float
            Current time.
        S : NDArray, shape (n_paths,)
            Current spot prices.

        Returns
        -------
        NDArray, shape (n_paths,)
        """

    @abstractmethod
    def diffusion(self, t: float, S: NDArray) -> NDArray:
        """Instantaneous diffusion coefficient.

        Parameters
        ----------
        t : float
            Current time.
        S : NDArray, shape (n_paths,)
            Current spot prices.

        Returns
        -------
        NDArray, shape (n_paths,)
        """

    @abstractmethod
    def step(
        self,
        t: float,
        dt: float,
        S: NDArray,
        dW: NDArray,
        **state: NDArray,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Advance the process by one time step.

        Parameters
        ----------
        t : float
            Current time.
        dt : float
            Time increment.
        S : NDArray, shape (n_paths,)
            Current spot prices.
        dW : NDArray, shape (n_paths,) or (n_paths, n_factors)
            Brownian increments (already scaled by sqrt(dt)).
        **state
            Additional state arrays (e.g. variance for Heston).

        Returns
        -------
        S_new : NDArray, shape (n_paths,)
        state_new : dict[str, NDArray]
        """

    @abstractmethod
    def sample_paths(
        self,
        S0: float,
        t_grid: NDArray,
        n_paths: int,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> dict[str, NDArray]:
        """Generate Monte Carlo paths over the given time grid.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        t_grid : NDArray, shape (n_steps+1,)
            Time points including t=0.
        n_paths : int
            Number of paths to simulate.
        rng : numpy.random.Generator
            Random number generator.
        antithetic : bool
            If True, generate antithetic paths for variance reduction.

        Returns
        -------
        dict with at least 'spot': NDArray of shape (n_paths, n_steps+1).
        May include additional state arrays (e.g. 'variance' for Heston).
        """

    @property
    def n_factors(self) -> int:
        """Number of Brownian factors driving the process."""
        return 1
