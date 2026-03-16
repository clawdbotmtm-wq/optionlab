"""Abstract base class for payoff definitions."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Payoff(ABC):
    """Base class for contingent claim payoffs.

    A payoff is a function of simulated paths that returns the discounted
    cashflow for each path. The engine handles discounting; the payoff
    returns undiscounted terminal or path-dependent cashflows.
    """

    @abstractmethod
    def cashflows(
        self,
        paths: NDArray,
        t_grid: NDArray,
    ) -> NDArray:
        """Compute the payoff for each simulated path.

        Parameters
        ----------
        paths : NDArray, shape (n_paths, n_steps+1)
            Simulated spot price paths.
        t_grid : NDArray, shape (n_steps+1,)
            Time grid corresponding to path columns.

        Returns
        -------
        NDArray, shape (n_paths,)
            Undiscounted payoff for each path.
        """

    @property
    @abstractmethod
    def expiry(self) -> float:
        """Time to expiry in years."""
