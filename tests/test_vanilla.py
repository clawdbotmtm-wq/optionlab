"""Tests for vanilla European option pricing — MC vs Black-Scholes."""

import numpy as np
import pytest

import optionlab as ol


class TestBlackScholesAnalytical:
    """Test the analytical Black-Scholes implementation."""

    def test_call_put_parity(self):
        """Call - Put = S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
        market = ol.MarketData.from_flat(S, r, q)

        call = ol.bs_price(ol.EuropeanOption(K, T, "call"), market, sigma)
        put = ol.bs_price(ol.EuropeanOption(K, T, "put"), market, sigma)

        parity_lhs = call.price - put.price
        parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(parity_lhs - parity_rhs) < 1e-10

    def test_atm_call_price(self):
        """ATM call price should be roughly S * sigma * sqrt(T) * 0.4."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        market = ol.MarketData.from_flat(S, r)
        result = ol.bs_price(ol.EuropeanOption(K, T, "call"), market, sigma)
        # BS ATM call ≈ 10.45 for these params
        assert 10.0 < result.price < 11.0

    def test_iv_roundtrip(self):
        """Implied vol should recover the input vol."""
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.03, 0.25
        market = ol.MarketData.from_flat(S, r)
        result = ol.bs_price(ol.EuropeanOption(K, T, "call"), market, sigma)
        assert abs(result.iv - sigma) < 1e-8

    def test_deep_itm_put_intrinsic(self):
        """Deep ITM put should be close to intrinsic value."""
        S, K, T, r, sigma = 50.0, 100.0, 0.01, 0.05, 0.20
        market = ol.MarketData.from_flat(S, r)
        result = ol.bs_price(ol.EuropeanOption(K, T, "put"), market, sigma)
        intrinsic = (K - S) * np.exp(-r * T)
        assert abs(result.price - intrinsic) < 0.5


class TestMonteCarloGBM:
    """Test MC engine against Black-Scholes for GBM dynamics."""

    def test_mc_call_converges_to_bs(self):
        """MC call price should be within 3 stderr of BS analytical."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        market = ol.MarketData.from_flat(S, r)
        option = ol.EuropeanOption(K, T, "call")

        analytical = ol.bs_price(option, market, sigma)
        dynamics = ol.GBM(r=r, sigma=sigma)
        engine = ol.MonteCarloEngine(n_paths=200_000, n_steps=1, seed=42, antithetic=True)
        mc = engine.price(option, dynamics, market)

        assert abs(mc.price - analytical.price) < 3 * mc.stderr

    def test_mc_put_converges_to_bs(self):
        """MC put price should converge to BS."""
        S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.03, 0.30
        market = ol.MarketData.from_flat(S, r)
        option = ol.EuropeanOption(K, T, "put")

        analytical = ol.bs_price(option, market, sigma)
        dynamics = ol.GBM(r=r, sigma=sigma)
        engine = ol.MonteCarloEngine(n_paths=200_000, n_steps=1, seed=123, antithetic=True)
        mc = engine.price(option, dynamics, market)

        assert abs(mc.price - analytical.price) < 3 * mc.stderr

    def test_antithetic_reduces_variance(self):
        """Antithetic variates should reduce standard error."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        market = ol.MarketData.from_flat(S, r)
        option = ol.EuropeanOption(K, T, "call")
        dynamics = ol.GBM(r=r, sigma=sigma)

        mc_plain = ol.MonteCarloEngine(
            n_paths=50_000, n_steps=1, seed=42, antithetic=False,
        ).price(option, dynamics, market)

        mc_anti = ol.MonteCarloEngine(
            n_paths=50_000, n_steps=1, seed=42, antithetic=True,
        ).price(option, dynamics, market)

        assert mc_anti.stderr < mc_plain.stderr


class TestImpliedVol:
    """Test the IV solver."""

    def test_brent_solver(self):
        """IV solver should invert BS price accurately."""
        from optionlab.surface.implied_vol import black_scholes_price, implied_vol

        S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.05, 0.25
        price = black_scholes_price(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r)
        assert abs(iv - sigma) < 1e-10

    def test_put_iv(self):
        """IV solver should work for puts too."""
        from optionlab.surface.implied_vol import black_scholes_price, implied_vol

        S, K, T, r, sigma = 100.0, 90.0, 0.5, 0.03, 0.30
        price = black_scholes_price(S, K, T, r, sigma, option_type="put")
        iv = implied_vol(price, S, K, T, r, option_type="put")
        assert abs(iv - sigma) < 1e-10


class TestCustomPayoff:
    """Test custom payoff functionality."""

    def test_asian_call(self):
        """Custom payoff for an arithmetic Asian call should run without error."""
        K = 100.0
        asian = ol.CustomPayoff(
            func=lambda paths, t: np.maximum(paths.mean(axis=1) - K, 0),
            T=1.0,
            name="Asian Call",
        )
        market = ol.MarketData.from_flat(100.0, 0.05)
        dynamics = ol.GBM(r=0.05, sigma=0.20)
        engine = ol.MonteCarloEngine(n_paths=50_000, n_steps=252, seed=42)
        result = engine.price(asian, dynamics, market)

        # Asian call should be cheaper than vanilla call
        vanilla = ol.EuropeanOption(K, 1.0, "call")
        vanilla_price = engine.price(vanilla, dynamics, market).price
        assert result.price > 0
        assert result.price < vanilla_price


class TestHestonSmoke:
    """Smoke tests for Heston dynamics."""

    def test_heston_paths_shape(self):
        """Heston should produce correctly shaped path arrays."""
        dynamics = ol.Heston(r=0.05, V0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        rng = np.random.default_rng(42)
        t_grid = np.linspace(0, 1, 253)
        paths = dynamics.sample_paths(100.0, t_grid, 1000, rng)

        assert paths["spot"].shape == (1000, 253)
        assert paths["variance"].shape == (1000, 253)
        assert np.all(paths["spot"] > 0)
        assert np.all(paths["variance"] >= 0)

    def test_heston_prices_otm_put(self):
        """Heston should produce a positive price for an OTM put."""
        market = ol.MarketData.from_flat(100.0, 0.05)
        dynamics = ol.Heston(r=0.05, V0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        option = ol.EuropeanOption(strike=90.0, T=0.5, option_type="put")
        engine = ol.MonteCarloEngine(n_paths=50_000, n_steps=126, seed=42, antithetic=True)
        result = engine.price(option, dynamics, market)

        assert result.price > 0
        assert result.stderr > 0
