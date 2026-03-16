"""Discount and forward rate curves."""

import numpy as np
from numpy.typing import NDArray


class DiscountCurve:
    """Flat or term-structure discount curve with continuous compounding.

    For the initial scaffold, this implements a flat rate curve.
    Extend to bootstrapped curves by overriding discount_factor().

    Parameters
    ----------
    rate : float
        Continuously compounded risk-free rate.
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def discount_factor(self, T: float | NDArray) -> float | NDArray:
        """Discount factor D(0, T) = exp(-r * T)."""
        return np.exp(-self.rate * np.asarray(T))

    def forward_rate(self, T1: float, T2: float) -> float:
        """Continuously compounded forward rate between T1 and T2."""
        if T2 <= T1:
            raise ValueError(f"T2 ({T2}) must be greater than T1 ({T1})")
        return (self.rate * T2 - self.rate * T1) / (T2 - T1)

    def __repr__(self) -> str:
        return f"DiscountCurve(rate={self.rate})"


class ForwardCurve:
    """Forward price curve for the underlying.

    Computes forward prices from spot, rates, and dividend yield.

    Parameters
    ----------
    spot : float
        Current spot price.
    rate : float
        Risk-free rate (continuous compounding).
    dividend_yield : float
        Continuous dividend yield.
    """

    def __init__(self, spot: float, rate: float, dividend_yield: float = 0.0) -> None:
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

    def forward(self, T: float | NDArray) -> float | NDArray:
        """Forward price F(0, T) = S * exp((r - q) * T)."""
        T = np.asarray(T)
        return self.spot * np.exp((self.rate - self.dividend_yield) * T)

    def __repr__(self) -> str:
        return (
            f"ForwardCurve(spot={self.spot}, rate={self.rate}, "
            f"q={self.dividend_yield})"
        )
