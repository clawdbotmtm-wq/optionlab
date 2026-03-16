"""Bump-and-reprice Greeks — universal finite-difference approach."""

from dataclasses import dataclass

from ..dynamics.base import Dynamics
from ..dynamics.gbm import GBM
from ..engine.monte_carlo import MonteCarloEngine, PricingResult
from ..market.data import MarketData
from ..payoffs.base import Payoff


@dataclass
class Greeks:
    """First- and second-order Greeks via bump-and-reprice."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


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
        sigma_orig = dynamics.sigma
        dynamics.sigma = sigma_orig + vol_bump
        price_vup = _reprice(engine, payoff, dynamics, market)
        dynamics.sigma = sigma_orig - vol_bump
        price_vdn = _reprice(engine, payoff, dynamics, market)
        dynamics.sigma = sigma_orig  # restore
        vega = (price_vup - price_vdn) / (2 * vol_bump) * 0.01  # per 1% vol
    elif hasattr(dynamics, "V0"):
        # For Heston, bump initial variance
        V0_orig = dynamics.V0
        dynamics.V0 = V0_orig + vol_bump * 2 * (V0_orig**0.5)
        price_vup = _reprice(engine, payoff, dynamics, market)
        dynamics.V0 = V0_orig - vol_bump * 2 * (V0_orig**0.5)
        price_vdn = _reprice(engine, payoff, dynamics, market)
        dynamics.V0 = V0_orig
        bump_size = vol_bump * 2 * (V0_orig**0.5)
        vega = (price_vup - price_vdn) / (2 * bump_size) * 0.01

    # Theta: reprice with shorter expiry
    theta = 0.0
    if hasattr(payoff, "T") and payoff.T > time_bump:
        orig_T = payoff.T
        payoff.T = orig_T - time_bump
        price_short = _reprice(engine, payoff, dynamics, market)
        payoff.T = orig_T  # restore
        theta = (price_short - base_price) / time_bump * (-1.0 / 365.0)

    # Rho: bump risk-free rate
    rho = 0.0
    if hasattr(dynamics, "r"):
        r_orig = dynamics.r
        dynamics.r = r_orig + rate_bump
        market_rup = MarketData.from_flat(
            S, r_orig + rate_bump, market.forward_curve.dividend_yield,
        )
        price_rup = _reprice(engine, payoff, dynamics, market_rup)
        dynamics.r = r_orig - rate_bump
        market_rdn = MarketData.from_flat(
            S, r_orig - rate_bump, market.forward_curve.dividend_yield,
        )
        price_rdn = _reprice(engine, payoff, dynamics, market_rdn)
        dynamics.r = r_orig  # restore
        rho = (price_rup - price_rdn) / (2 * rate_bump) * 0.01  # per 1% rate

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
