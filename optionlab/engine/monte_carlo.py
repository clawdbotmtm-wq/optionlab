"""Monte Carlo pricing engine with antithetic variates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..dynamics.base import Dynamics
from ..market.data import MarketData
from ..payoffs.base import Payoff
from ..surface.implied_vol import implied_vol


@dataclass
class PricingResult:
    """Result from Monte Carlo pricing.

    Attributes
    ----------
    price : float
        Discounted option price (mean of payoffs).
    stderr : float
        Standard error of the price estimate.
    iv : float
        Black-Scholes implied volatility of the price.
    paths : NDArray | None
        Simulated spot paths, if retained.
    """

    price: float
    stderr: float
    iv: float
    paths: NDArray | None = None


class MonteCarloEngine:
    """Vectorized Monte Carlo pricing engine.

    Connects dynamics (path generation) to payoffs (path consumption)
    with variance reduction via antithetic variates.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps in discretization.
    seed : int | None
        Random seed for reproducibility.
    antithetic : bool
        Use antithetic variates for variance reduction.
    store_paths : bool
        Whether to retain paths in the result.
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int | None = 42,
        antithetic: bool = True,
        store_paths: bool = False,
    ):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.antithetic = antithetic
        self.store_paths = store_paths

    def price(
        self,
        payoff: Payoff,
        dynamics: Dynamics,
        market: MarketData,
    ) -> PricingResult:
        """Price a contingent claim via Monte Carlo simulation.

        Parameters
        ----------
        payoff : Payoff
            The option or derivative to price.
        dynamics : Dynamics
            The stochastic process for the underlying.
        market : MarketData
            Market data (spot, curves).

        Returns
        -------
        PricingResult
            Contains price, standard error, implied vol, and optionally paths.
        """
        rng = np.random.default_rng(self.seed)
        T = payoff.expiry
        t_grid = np.linspace(0.0, T, self.n_steps + 1)

        # Generate paths (vectorized across all paths simultaneously)
        path_data = dynamics.sample_paths(
            S0=market.spot,
            t_grid=t_grid,
            n_paths=self.n_paths,
            rng=rng,
            antithetic=self.antithetic,
        )
        spot_paths = path_data["spot"]

        # Compute payoffs (vectorized across all paths)
        cf = payoff.cashflows(spot_paths, t_grid)

        # Discount to present value
        df = market.discount_curve.discount_factor(T)
        pv = cf * df

        # Statistics
        price = float(np.mean(pv))
        stderr = float(np.std(pv, ddof=1) / np.sqrt(len(pv)))

        # Convert to implied vol
        r = market.discount_curve.rate
        q = market.forward_curve.dividend_yield

        # Try to extract strike and option type for IV computation
        iv_val = float("nan")
        if hasattr(payoff, "strike") and hasattr(payoff, "option_type"):
            iv_val = implied_vol(
                price, market.spot, payoff.strike, T, r, q,
                payoff.option_type,
            )

        return PricingResult(
            price=price,
            stderr=stderr,
            iv=iv_val,
            paths=spot_paths if self.store_paths else None,
        )
