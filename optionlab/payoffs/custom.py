"""Lambda-based custom payoffs — pass any function of paths."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .base import Payoff


class CustomPayoff(Payoff):
    """Custom payoff defined by an arbitrary function of paths.

    This lets you price any path-dependent claim without writing a new class.

    Parameters
    ----------
    func : callable
        Function with signature (paths: NDArray, t_grid: NDArray) -> NDArray
        that returns the undiscounted payoff for each path.
    T : float
        Time to expiry in years.
    name : str
        Optional human-readable name.

    Examples
    --------
    >>> # Asian call option (arithmetic average)
    >>> asian_call = CustomPayoff(
    ...     func=lambda paths, t: np.maximum(paths.mean(axis=1) - 100, 0),
    ...     T=1.0,
    ...     name="Asian Call K=100",
    ... )

    >>> # Digital call paying $1 if S_T > K
    >>> digital = CustomPayoff(
    ...     func=lambda paths, t: (paths[:, -1] > 100).astype(float),
    ...     T=1.0,
    ...     name="Digital Call K=100",
    ... )
    """

    def __init__(
        self,
        func: Callable[[NDArray, NDArray], NDArray],
        T: float,
        name: str = "CustomPayoff",
    ):
        self._func = func
        self.T = T
        self.name = name

    @property
    def expiry(self) -> float:
        return self.T

    def cashflows(self, paths: NDArray, t_grid: NDArray) -> NDArray:
        return self._func(paths, t_grid)

    def __repr__(self) -> str:
        return f"CustomPayoff(name='{self.name}', T={self.T})"
