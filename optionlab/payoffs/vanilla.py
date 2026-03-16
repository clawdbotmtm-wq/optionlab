"""European vanilla call and put payoffs."""

from collections.abc import Mapping
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .base import Payoff


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class EuropeanOption(Payoff):
    """European vanilla option payoff.

    Parameters
    ----------
    strike : float
        Strike price.
    T : float
        Time to expiry in years.
    option_type : OptionType or str
        'call' or 'put'.
    """

    def __init__(
        self,
        strike: float,
        T: float,
        option_type: str | OptionType = "call",
    ) -> None:
        self.strike = strike
        self.T = T
        self.option_type = OptionType(option_type)

    @property
    def expiry(self) -> float:
        """Time to expiry in years."""
        return self.T

    def cashflows(
        self,
        paths: NDArray,
        t_grid: NDArray,
        state_paths: Mapping[str, NDArray] | None = None,
    ) -> NDArray:
        """Terminal payoff evaluated at expiry."""
        S_T = paths[:, -1]
        if self.option_type == OptionType.CALL:
            return np.maximum(S_T - self.strike, 0.0)
        else:
            return np.maximum(self.strike - S_T, 0.0)

    def __repr__(self) -> str:
        return (
            f"EuropeanOption(strike={self.strike}, T={self.T}, "
            f"type={self.option_type.value})"
        )
