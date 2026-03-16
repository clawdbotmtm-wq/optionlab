# optionlab — Flexible Options Pricing Engine

## Philosophy
- Industry-focused, not academic toy. Inspired by Taleb's Dynamic Hedging, Gatheral's Volatility Surface, Rebonato's practical approach.
- Simulation-first: Monte Carlo as the universal pricer, with analytical shortcuts where available.
- Pluggable dynamics: swap spot processes without touching payoff logic.
- Pluggable payoffs: define any contingent claim as a function of paths.
- Greeks via finite difference on any pricer (bump-and-reprice), plus pathwise where available.
- Output layer always converts to implied vol for comparison across models.

## Architecture

```
optionlab/
├── dynamics/          # Spot process specifications
│   ├── base.py        # Abstract base: defines drift/diffusion/jump interface
│   ├── gbm.py         # Geometric Brownian Motion (Black-Scholes)
│   ├── heston.py      # Heston stochastic vol
│   ├── sabr.py        # SABR
│   ├── local_vol.py   # Dupire local vol (from surface)
│   ├── jump_diffusion.py  # Merton, Kou
│   └── custom.py      # User-defined SDE
│
├── payoffs/           # Contingent claim definitions
│   ├── base.py        # Abstract payoff: function of paths → cashflows
│   ├── vanilla.py     # European/American calls/puts
│   ├── barrier.py     # Knock-in/out, digital
│   ├── asian.py       # Arithmetic/geometric average
│   ├── compound.py    # Options on options
│   ├── real_options.py # Mortgage refi, exercise decisions
│   └── custom.py      # Lambda-based custom payoffs
│
├── engine/            # Pricing engines
│   ├── monte_carlo.py # Core MC engine with variance reduction
│   ├── analytical.py  # Closed-form where available (BS, Heston CF)
│   ├── pde.py         # Finite difference PDE solver (future)
│   └── greeks.py      # Finite-difference Greeks wrapper
│
├── surface/           # Vol surface tools
│   ├── implied_vol.py # Price → IV conversion (Newton/Brent)
│   ├── svi.py         # SVI parameterization
│   ├── sabr_vol.py    # SABR vol formulas
│   └── surface.py     # Full surface object (strike × expiry)
│
├── market/            # Market data containers
│   ├── curves.py      # Discount curves, forward curves
│   └── data.py        # Market snapshot (spot, vol surface, rates)
│
├── utils/             # Helpers
│   ├── stats.py       # MC diagnostics, convergence
│   └── time.py        # Day count, schedule generation
│
├── examples/          # Worked examples
│   ├── 01_vanilla_bs.py
│   ├── 02_heston_smile.py
│   ├── 03_mortgage_refi.py
│   └── 04_barrier_local_vol.py
│
└── tests/
```

## Key Design Decisions

### 1. Dynamics as SDE specifications
Every dynamics class defines:
- `drift(t, S, **state) → float`
- `diffusion(t, S, **state) → float`  
- `step(t, dt, S, **state, rng) → (S_new, **state_new)` — Euler or exact
- `sample_paths(t_grid, n_paths, rng) → array[n_paths, n_steps]`

Additional state (e.g., variance in Heston) carried as named arrays.

### 2. Payoffs as path functionals
Every payoff defines:
- `cashflows(paths, t_grid, **kwargs) → array[n_paths]` — terminal or path-dependent
- `exercise_decisions(paths, t_grid) → array` — for American/Bermudan (LSM)

### 3. Greeks via bump-and-reprice
Universal approach:
```python
def delta(pricer, payoff, dynamics, market, bump=0.01):
    price_up = pricer.price(payoff, dynamics, market.bump_spot(+bump))
    price_dn = pricer.price(payoff, dynamics, market.bump_spot(-bump))
    return (price_up - price_dn) / (2 * bump * market.spot)
```

### 4. IV output layer
Every price gets converted to Black-Scholes implied vol for comparability:
```python
result = pricer.price(payoff, dynamics, market)
result.price     # dollar price
result.iv        # implied vol
result.greeks    # dict of Greeks
```

## Language & Dependencies
- Python 3.11+
- numpy, scipy (core numerics)
- Optional: numba (JIT for MC inner loops), matplotlib (viz)
- No heavy frameworks — keep it lean and hackable

## What This Is NOT
- Not a trading system
- Not a market data platform  
- Not trying to be QuantLib (too heavy, too C++)
- Not a backtester
