"""Implied volatility solver using Brent's method."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from ..payoffs.vanilla import OptionType


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str | OptionType = "call",
) -> float:
    """Black-Scholes closed-form price for a European option.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate (continuous compounding).
    sigma : float
        Volatility.
    q : float
        Continuous dividend yield.
    option_type : str or OptionType
        'call' or 'put'.

    Returns
    -------
    float
        Option price.
    """
    option_type = OptionType(option_type)

    if T <= 0 or sigma <= 0:
        # Intrinsic value at expiry
        F = S * np.exp((r - q) * T)
        if option_type == OptionType.CALL:
            return max(F - K, 0.0) * np.exp(-r * T)
        else:
            return max(K - F, 0.0) * np.exp(-r * T)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == OptionType.CALL:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: str | OptionType = "call",
    bounds: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Invert the Black-Scholes formula to find implied volatility.

    Uses Brent's method (scipy.optimize.brentq) for robust root-finding.

    Parameters
    ----------
    price : float
        Observed (or model) option price.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    option_type : str or OptionType
        'call' or 'put'.
    bounds : tuple
        Search interval for volatility (low, high).

    Returns
    -------
    float
        Implied volatility, or NaN if no solution found.
    """
    option_type = OptionType(option_type)

    def objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - price

    try:
        return brentq(objective, bounds[0], bounds[1], xtol=1e-12, maxiter=200)
    except ValueError:
        return float("nan")
