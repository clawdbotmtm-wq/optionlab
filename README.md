# optionlab

Flexible Monte Carlo options pricing engine. Simulation-first, with pluggable dynamics and payoffs.

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import optionlab as ol

# Market data
market = ol.MarketData.from_flat(spot=100, rate=0.05)

# European call option
option = ol.EuropeanOption(strike=100, T=1.0, option_type="call")

# Black-Scholes analytical price
result = ol.bs_price(option, market, sigma=0.20)
print(f"BS Price: {result.price:.4f}, Delta: {result.delta:.4f}")

# Monte Carlo under GBM
dynamics = ol.GBM(r=0.05, sigma=0.20)
engine = ol.MonteCarloEngine(n_paths=100_000, antithetic=True)
mc = engine.price(option, dynamics, market)
print(f"MC Price: {mc.price:.4f} ± {mc.stderr:.4f}")

# Heston stochastic vol
heston = ol.Heston(r=0.05, V0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
mc_heston = engine.price(option, heston, market)
print(f"Heston Price: {mc_heston.price:.4f}, IV: {mc_heston.iv:.4f}")

# Custom payoff (Asian call)
import numpy as np
asian = ol.CustomPayoff(
    func=lambda paths, t: np.maximum(paths.mean(axis=1) - 100, 0),
    T=1.0,
)
mc_asian = engine.price(asian, dynamics, market)
print(f"Asian Call: {mc_asian.price:.4f}")
```

## Architecture

- **dynamics/** — SDE specifications (GBM, Heston). Produce paths.
- **payoffs/** — Contingent claims (vanilla, custom). Consume paths.
- **engine/** — Monte Carlo pricer, analytical BS, bump-and-reprice Greeks.
- **surface/** — Implied volatility solver (Brent's method).
- **market/** — Discount curves, forward curves, market data container.

## Examples

```bash
python examples/01_vanilla_bs.py    # MC vs analytical comparison
python examples/02_heston_smile.py  # Heston implied vol smile
```

## Tests

```bash
pytest tests/ -v
```
