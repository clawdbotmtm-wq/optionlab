"""Black-Scholes analytical pricing engine."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..market.data import MarketData
from ..payoffs.vanilla import EuropeanOption, OptionType
from ..surface.implied_vol import implied_vol


@dataclass
class AnalyticalResult:
    """Result from analytical pricing."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    iv: float


def bs_price(
    option: EuropeanOption,
    market: MarketData,
    sigma: float,
) -> AnalyticalResult:
    """Price a European option using the Black-Scholes formula.

    Returns price and all first-order Greeks in closed form.

    Parameters
    ----------
    option : EuropeanOption
        The option to price.
    market : MarketData
        Market data (spot, rates).
    sigma : float
        Volatility to use.

    Returns
    -------
    AnalyticalResult
        Contains price, delta, gamma, vega, theta, rho, and implied vol.
    """
    S = market.spot
    K = option.strike
    T = option.expiry
    r = market.discount_curve.rate
    q = market.forward_curve.dividend_yield
    is_call = option.option_type == OptionType.CALL

    if T <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return AnalyticalResult(
            price=intrinsic, delta=1.0 if is_call and S > K else 0.0,
            gamma=0.0, vega=0.0, theta=0.0, rho=0.0, iv=float("nan"),
        )

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Price
    if is_call:
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    # Greeks
    n_d1 = norm.pdf(d1)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)

    if is_call:
        delta = exp_qT * norm.cdf(d1)
        theta = (
            -S * exp_qT * n_d1 * sigma / (2 * sqrt_T)
            - r * K * exp_rT * norm.cdf(d2)
            + q * S * exp_qT * norm.cdf(d1)
        )
        rho_val = K * T * exp_rT * norm.cdf(d2)
    else:
        delta = -exp_qT * norm.cdf(-d1)
        theta = (
            -S * exp_qT * n_d1 * sigma / (2 * sqrt_T)
            + r * K * exp_rT * norm.cdf(-d2)
            - q * S * exp_qT * norm.cdf(-d1)
        )
        rho_val = -K * T * exp_rT * norm.cdf(-d2)

    gamma = exp_qT * n_d1 / (S * sigma * sqrt_T)
    vega = S * exp_qT * n_d1 * sqrt_T / 100  # per 1% vol move

    # IV (should round-trip to sigma for BS)
    iv_val = implied_vol(price, S, K, T, r, q, option.option_type)

    return AnalyticalResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta / 365,  # per calendar day
        rho=rho_val / 100,  # per 1% rate move
        iv=iv_val,
    )
