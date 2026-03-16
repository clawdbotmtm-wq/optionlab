"""Bump-and-reprice Greeks — universal finite-difference approach."""

from copy import copy
from dataclasses import dataclass, is_dataclass, replace
from math import sqrt
from typing import Any, TypeVar, cast

from ..dynamics.base import Dynamics
from ..engine.monte_carlo import MonteCarloEngine
from ..market.data import MarketData
from ..payoffs.base import Payoff

_T = TypeVar("_T")


@dataclass
class Greeks:
    """First- and second-order Greeks via bump-and-reprice."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _clone_with_updates(obj: _T, /, **updates: Any) -> _T:
    """Clone an object and update selected attributes."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return cast(_T, replace(obj, **updates))

    cloned = copy(obj)
    for attr, value in updates.items():
        if not hasattr(cloned, attr):
            raise AttributeError(
                f"Cannot bump missing attribute '{attr}' on {type(cloned).__name__}."
            )
        setattr(cloned, attr, value)
    return cast(_T, cloned)


def _reprice(
    engine: MonteCarloEngine,
    payoff: Payoff,
    dynamics: Dynamics,
    market: MarketData,
) -> float:
    """Helper to get just the price from an engine run."""
    return engine.price(payoff, dynamics, market).price


def compute_greeks(
    engine: MonteCarloEngine,
    payoff: Payoff,
    dynamics: Dynamics,
    market: MarketData,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    time_bump: float = 1.0 / 365.0,
    rate_bump: float = 0.0001,
) -> Greeks:
    """Compute Greeks via central finite differences (bump-and-reprice).

    This is the universal approach that works for any dynamics/payoff
    combination. Bumps are relative for spot, absolute for vol/rate/time.

    Parameters
    ----------
    engine : MonteCarloEngine
        The pricing engine to use.
    payoff : Payoff
        The option to price.
    dynamics : Dynamics
        The stochastic process.
    market : MarketData
        Market data.
    spot_bump : float
        Relative spot bump (fraction of spot). Default 1%.
    vol_bump : float
        Absolute volatility bump. Default 1%.
    time_bump : float
        Time decrement for theta. Default 1 day.
    rate_bump : float
        Absolute rate bump for rho. Default 1bp.

    Returns
    -------
    Greeks
        Delta, gamma, vega, theta, rho.
    """
    S = market.spot
    h_s = spot_bump * S  # absolute spot bump

    base_price = _reprice(engine, payoff, dynamics, market)

    # Delta and Gamma: bump spot
    market_up = market.bump_spot(+h_s)
    market_dn = market.bump_spot(-h_s)
    price_up = _reprice(engine, payoff, dynamics, market_up)
    price_dn = _reprice(engine, payoff, dynamics, market_dn)

    delta = (price_up - price_dn) / (2 * h_s)
    gamma = (price_up - 2 * base_price + price_dn) / (h_s**2)

    # Vega: bump volatility (requires dynamics with sigma attribute)
    vega = 0.0
    if hasattr(dynamics, "sigma"):
        sigma_orig = float(getattr(dynamics, "sigma"))
        dynamics_vup = _clone_with_updates(dynamics, sigma=sigma_orig + vol_bump)
        dynamics_vdn = _clone_with_updates(dynamics, sigma=sigma_orig - vol_bump)
        price_vup = _reprice(engine, payoff, dynamics_vup, market)
        price_vdn = _reprice(engine, payoff, dynamics_vdn, market)
        vega = (price_vup - price_vdn) / (2 * vol_bump) * 0.01  # per 1% vol
    elif hasattr(dynamics, "V0"):
        # For Heston, bump initial variance
        V0_orig = float(getattr(dynamics, "V0"))
        vol_level = sqrt(max(V0_orig, 0.0))
        bump_size = 2.0 * vol_level * vol_bump
        if bump_size > 0.0:
            V0_up = V0_orig + bump_size
            V0_dn = max(V0_orig - bump_size, 0.0)
            dynamics_vup = _clone_with_updates(dynamics, V0=V0_up)
            dynamics_vdn = _clone_with_updates(dynamics, V0=V0_dn)
            price_vup = _reprice(engine, payoff, dynamics_vup, market)
            price_vdn = _reprice(engine, payoff, dynamics_vdn, market)
            vega = (price_vup - price_vdn) / (V0_up - V0_dn) * 0.01

    # Theta: reprice with shorter expiry
    theta = 0.0
    if hasattr(payoff, "T") and payoff.T > time_bump:
        payoff_short = _clone_with_updates(payoff, T=payoff.T - time_bump)
        price_short = _reprice(engine, payoff_short, dynamics, market)
        theta = (price_short - base_price) / time_bump * (-1.0 / 365.0)

    # Rho: bump risk-free rate
    rho = 0.0
    if hasattr(dynamics, "r"):
        r_orig = float(getattr(dynamics, "r"))
        dynamics_rup = _clone_with_updates(dynamics, r=r_orig + rate_bump)
        market_rup = MarketData.from_flat(
            S, r_orig + rate_bump, market.forward_curve.dividend_yield,
        )
        price_rup = _reprice(engine, payoff, dynamics_rup, market_rup)

        dynamics_rdn = _clone_with_updates(dynamics, r=r_orig - rate_bump)
        market_rdn = MarketData.from_flat(
            S, r_orig - rate_bump, market.forward_curve.dividend_yield,
        )
        price_rdn = _reprice(engine, payoff, dynamics_rdn, market_rdn)
        rho = (price_rup - price_rdn) / (2 * rate_bump) * 0.01  # per 1% rate

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
