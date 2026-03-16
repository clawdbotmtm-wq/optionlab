"""Example 3: Pricing a mortgage refinance option.

A mortgage borrower holds an embedded American call option on interest rates.
They can refinance at any point, paying closing costs to get a lower rate.
This is equivalent to a callable bond from the lender's perspective.

The "underlying" is the mortgage rate (or equivalently, the 10Y Treasury).
The "strike" is the breakeven rate where PV savings > refi costs.
The "payoff" at exercise is: PV(old payments) - PV(new payments) - closing_costs.

We model rates using a mean-reverting process (Vasicek/CIR) and simulate
the optimal exercise boundary via Longstaff-Schwartz regression.

Current market data (March 2026):
- 30Y mortgage rate: 6.11%
- 10Y Treasury: 4.13%
- Your existing mortgage: 5.25% on $5M, ~28 years remaining
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from numpy.typing import NDArray


# ============================================================
# MORTGAGE CASH FLOW HELPERS
# ============================================================

def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    """Fixed monthly payment for a standard amortizing mortgage."""
    r = annual_rate / 12
    n = years * 12
    if r < 1e-8:
        return principal / n
    return principal * r * (1 + r)**n / ((1 + r)**n - 1)


def remaining_balance(principal: float, annual_rate: float, total_years: int,
                       years_elapsed: float) -> float:
    """Outstanding balance after years_elapsed of payments."""
    r = annual_rate / 12
    n = total_years * 12
    p = years_elapsed * 12
    if r < 1e-8:
        return principal * (1 - p/n)
    pmt = monthly_payment(principal, annual_rate, total_years)
    return principal * (1 + r)**p - pmt * ((1 + r)**p - 1) / r


def pv_remaining_payments(balance, annual_rate, remaining_years, discount_rate):
    """Present value of remaining mortgage payments discounted at discount_rate.
    
    All args can be scalar or array (vectorized across paths).
    """
    annual_rate = np.asarray(annual_rate, dtype=float)
    discount_rate = np.asarray(discount_rate, dtype=float)
    balance = np.asarray(balance, dtype=float)
    
    r_pay = annual_rate / 12
    r_disc = discount_rate / 12
    n = int(remaining_years * 12)
    if n <= 0:
        return np.zeros_like(discount_rate)
    
    # Monthly payment
    pmt = np.where(r_pay > 1e-8,
                   balance * r_pay * (1 + r_pay)**n / ((1 + r_pay)**n - 1),
                   balance / n)
    
    # PV of annuity
    pv = np.where(r_disc > 1e-8,
                  pmt * (1 - (1 + r_disc)**(-n)) / r_disc,
                  pmt * n)
    return pv


# ============================================================
# RATE DYNAMICS: CIR (Cox-Ingersoll-Ross) MODEL
# ============================================================

class CIRDynamics:
    """Cox-Ingersoll-Ross mean-reverting rate process.
    
    dr = kappa * (theta - r) * dt + xi * sqrt(r) * dW
    
    Parameters calibrated to current market:
    - theta: long-run mean rate (where rates revert to)
    - kappa: speed of mean reversion
    - xi: volatility of rates
    - r0: current rate level
    """
    
    def __init__(self, r0: float, kappa: float, theta: float, xi: float):
        self.r0 = r0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
    
    def sample_paths(self, t_grid: NDArray, n_paths: int, 
                     rng: np.random.Generator) -> NDArray:
        """Simulate rate paths using full-truncation Euler."""
        n_steps = len(t_grid) - 1
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0
        
        for i in range(n_steps):
            dt = t_grid[i + 1] - t_grid[i]
            r = rates[:, i]
            r_pos = np.maximum(r, 0)  # truncation for CIR
            
            dW = rng.standard_normal(n_paths) * np.sqrt(dt)
            dr = self.kappa * (self.theta - r_pos) * dt + self.xi * np.sqrt(r_pos) * dW
            rates[:, i + 1] = np.maximum(r + dr, 0.001)  # floor at 0.1%
        
        return rates


# ============================================================
# MORTGAGE REFI OPTION: LONGSTAFF-SCHWARTZ
# ============================================================

def price_refi_option(
    # Existing mortgage
    original_principal: float,
    original_rate: float,
    original_term_years: int,
    years_elapsed: float,
    
    # Refi costs
    closing_cost_pct: float,  # as fraction of balance
    closing_cost_fixed: float,  # fixed dollar amount
    
    # Rate dynamics
    rate_dynamics: CIRDynamics,
    
    # Simulation params
    n_paths: int = 100_000,
    n_steps_per_year: int = 12,  # monthly exercise opportunities
    horizon_years: float = 10.0,  # max time to consider refi
    seed: int = 42,
) -> dict:
    """Price the embedded refinance option using Longstaff-Schwartz.
    
    Returns dict with option value, optimal exercise boundary, and diagnostics.
    """
    rng = np.random.default_rng(seed)
    
    remaining_term = original_term_years - years_elapsed
    current_balance = remaining_balance(original_principal, original_rate, 
                                         original_term_years, years_elapsed)
    
    # Time grid (monthly)
    n_steps = int(horizon_years * n_steps_per_year)
    t_grid = np.linspace(0, horizon_years, n_steps + 1)
    
    # Simulate rate paths
    rate_paths = rate_dynamics.sample_paths(t_grid, n_paths, rng)
    
    # Convert Treasury rate to mortgage rate (spread ≈ 1.9% currently)
    mortgage_spread = original_rate - rate_dynamics.r0  # implied spread at inception
    # Use a more realistic fixed spread
    mortgage_spread = 1.90 / 100  # typical 30Y mortgage - 10Y Treasury spread
    mortgage_rate_paths = rate_paths + mortgage_spread
    
    print(f"Current balance: ${current_balance:,.0f}")
    print(f"Remaining term: {remaining_term:.1f} years")
    print(f"Monthly payment: ${monthly_payment(current_balance, original_rate, int(remaining_term)):,.0f}")
    print(f"Rate paths: mean={rate_paths[:, -1].mean()*100:.2f}%, "
          f"std={rate_paths[:, -1].std()*100:.2f}%")
    
    # ============================================================
    # LONGSTAFF-SCHWARTZ BACKWARD INDUCTION
    # ============================================================
    
    # Exercise value at each time step:
    # PV(old payments at old rate) - PV(new payments at new rate) - closing costs
    # where new rate = simulated mortgage rate at that time
    
    # Cash flows matrix (what you receive if you exercise at time t)
    exercise_value = np.zeros((n_paths, n_steps + 1))
    
    for j in range(1, n_steps + 1):
        t = t_grid[j]
        elapsed_at_t = years_elapsed + t
        remaining_at_t = max(original_term_years - elapsed_at_t, 0)
        
        if remaining_at_t < 0.5:  # less than 6 months left, not worth refi
            continue
        
        # Balance at time t (under original mortgage)
        bal_t = remaining_balance(original_principal, original_rate,
                                   original_term_years, elapsed_at_t)
        
        # New mortgage rate at time t
        new_rate = mortgage_rate_paths[:, j]
        
        # Closing costs
        closing_costs = closing_cost_pct * bal_t + closing_cost_fixed
        
        # Savings: PV of old payments - PV of new payments
        # Both discounted at the new (market) rate
        pv_old = pv_remaining_payments(bal_t, original_rate, remaining_at_t, new_rate)
        pv_new = pv_remaining_payments(bal_t, new_rate, remaining_at_t, new_rate)
        
        # Exercise value = savings - costs (only if positive)
        exercise_value[:, j] = np.maximum(pv_old - pv_new - closing_costs, 0)
    
    # Backward induction (LSM)
    # Start from the end and work backwards
    cashflow = exercise_value[:, -1].copy()  # terminal exercise value
    exercise_time = np.full(n_paths, n_steps, dtype=int)
    optimal_boundary = np.full(n_steps + 1, np.nan)
    
    for j in range(n_steps - 1, 0, -1):
        t = t_grid[j]
        dt_to_next = t_grid[j + 1] - t_grid[j]
        
        # Discount factor from j+1 to j (using simulated rate)
        df = np.exp(-rate_paths[:, j] * dt_to_next)
        
        # Continuation value (discounted future cashflow)
        continuation = cashflow * df
        
        # Identify in-the-money paths (positive exercise value)
        ev = exercise_value[:, j]
        itm = ev > 0
        
        if itm.sum() < 10:
            continue
        
        # Regression: continuation value ~ f(rate, rate^2, rate^3)
        r_itm = rate_paths[itm, j]
        X = np.column_stack([r_itm, r_itm**2, r_itm**3])
        y = continuation[itm]
        
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            continuation_hat = X @ coeffs
        except np.linalg.LinAlgError:
            continue
        
        # Exercise if exercise value > estimated continuation
        exercise_mask = np.zeros(n_paths, dtype=bool)
        exercise_mask[itm] = ev[itm] > continuation_hat
        
        # Update cashflows
        cashflow[exercise_mask] = ev[exercise_mask]
        exercise_time[exercise_mask] = j
        
        # Record optimal boundary (rate threshold)
        if exercise_mask.any():
            optimal_boundary[j] = rate_paths[exercise_mask, j].max()
    
    # Discount cashflows to time 0
    pv_cashflows = np.zeros(n_paths)
    for i in range(n_paths):
        j = exercise_time[i]
        if exercise_value[i, j] > 0:
            # Discount from exercise time to 0
            cum_rate = np.trapezoid(rate_paths[i, :j+1], t_grid[:j+1])
            pv_cashflows[i] = cashflow[i] * np.exp(-cum_rate)
    
    option_value = float(np.mean(pv_cashflows))
    option_stderr = float(np.std(pv_cashflows, ddof=1) / np.sqrt(n_paths))
    
    # Exercise statistics
    exercised = exercise_value[np.arange(n_paths), exercise_time] > 0
    exercise_frac = exercised.mean()
    mean_exercise_time = t_grid[exercise_time[exercised]].mean() if exercised.any() else np.nan
    mean_exercise_rate = rate_paths[exercised, exercise_time[exercised]].mean() if exercised.any() else np.nan
    
    return {
        'option_value': option_value,
        'stderr': option_stderr,
        'current_balance': current_balance,
        'exercise_fraction': exercise_frac,
        'mean_exercise_time_years': mean_exercise_time,
        'mean_exercise_rate': mean_exercise_rate,
        'optimal_boundary': optimal_boundary,
        'rate_paths': rate_paths,
        't_grid': t_grid,
        'exercise_times': exercise_time,
    }


# ============================================================
# MAIN: PRICE JAMES'S MORTGAGE REFI OPTION
# ============================================================

def main():
    print("=" * 70)
    print("MORTGAGE REFINANCE OPTION — Current Market Conditions")
    print("=" * 70)
    
    # James's mortgage
    original_principal = 5_000_000
    original_rate = 0.0525      # 5.25%
    original_term = 30           # 30-year fixed
    years_elapsed = 2.0          # ~2 years in
    
    # Refi costs (typical for jumbo)
    closing_cost_pct = 0.005     # 0.5% of balance (points)
    closing_cost_fixed = 15_000  # appraisal, title, legal, etc.
    
    # CIR rate dynamics calibrated to current market
    # 10Y Treasury = 4.13%, mortgage = 6.11%
    # Implied vol from swaption market: ~80-100 bps/year
    rate_model = CIRDynamics(
        r0=0.0413,      # current 10Y Treasury
        kappa=0.5,       # mean reversion speed (half-life ~1.4 years)
        theta=0.04,      # long-run mean (roughly neutral rate)
        xi=0.06,         # vol of rates (~80bp annual vol at current level)
    )
    
    print(f"\nMortgage: ${original_principal/1e6:.0f}M at {original_rate*100:.2f}%, "
          f"{original_term}Y fixed, {years_elapsed:.0f} years elapsed")
    print(f"Current 30Y mortgage rate: 6.11%")
    print(f"Current 10Y Treasury: 4.13%")
    print(f"Refi costs: {closing_cost_pct*100:.1f}% + ${closing_cost_fixed:,}")
    
    # Price the option
    result = price_refi_option(
        original_principal=original_principal,
        original_rate=original_rate,
        original_term_years=original_term,
        years_elapsed=years_elapsed,
        closing_cost_pct=closing_cost_pct,
        closing_cost_fixed=closing_cost_fixed,
        rate_dynamics=rate_model,
        n_paths=200_000,
        n_steps_per_year=12,
        horizon_years=10.0,
        seed=42,
    )
    
    print(f"\n{'─' * 50}")
    print(f"RESULTS")
    print(f"{'─' * 50}")
    print(f"  Refi option value:    ${result['option_value']:>12,.0f} ± ${result['stderr']:>8,.0f}")
    print(f"  As % of balance:      {result['option_value']/result['current_balance']*100:.2f}%")
    print(f"  Exercise probability: {result['exercise_fraction']*100:.1f}%")
    print(f"  Mean exercise time:   {result['mean_exercise_time_years']:.1f} years")
    print(f"  Mean rate at exercise:{result['mean_exercise_rate']*100:.2f}% (10Y)")
    print(f"  Implied mortgage at ex:{(result['mean_exercise_rate']+0.019)*100:.2f}%")
    
    # What rate makes refi worthwhile?
    balance = result['current_balance']
    closing_costs = closing_cost_pct * balance + closing_cost_fixed
    old_pmt = monthly_payment(balance, original_rate, int(original_term - years_elapsed))
    
    print(f"\n{'─' * 50}")
    print(f"BREAKEVEN ANALYSIS")
    print(f"{'─' * 50}")
    print(f"  Current monthly payment: ${old_pmt:,.0f}")
    print(f"  Closing costs: ${closing_costs:,.0f}")
    
    # Find breakeven rate
    for test_rate in np.arange(0.01, original_rate, 0.0025):
        new_pmt = monthly_payment(balance, test_rate, int(original_term - years_elapsed))
        monthly_savings = old_pmt - new_pmt
        if monthly_savings > 0:
            breakeven_months = closing_costs / monthly_savings
            pv_savings = pv_remaining_payments(balance, original_rate, original_term - years_elapsed, test_rate) - \
                         pv_remaining_payments(balance, test_rate, original_term - years_elapsed, test_rate)
            if pv_savings > closing_costs:
                print(f"  At {test_rate*100:.2f}%: save ${monthly_savings:,.0f}/mo, "
                      f"breakeven {breakeven_months:.0f} months, PV savings ${pv_savings:,.0f}")
    
    # Sensitivity to rate vol
    print(f"\n{'─' * 50}")
    print(f"SENSITIVITY: Option value vs rate volatility")
    print(f"{'─' * 50}")
    for xi in [0.03, 0.04, 0.06, 0.08, 0.10]:
        rm = CIRDynamics(r0=0.0413, kappa=0.5, theta=0.04, xi=xi)
        r = price_refi_option(
            original_principal, original_rate, original_term, years_elapsed,
            closing_cost_pct, closing_cost_fixed, rm,
            n_paths=50_000, seed=42,
        )
        print(f"  xi={xi:.2f}: value=${r['option_value']:>10,.0f}, "
              f"exercise={r['exercise_fraction']*100:.0f}%")


if __name__ == "__main__":
    main()
