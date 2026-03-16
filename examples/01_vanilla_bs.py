"""Example 1: Price a European call under GBM, compare MC to Black-Scholes analytical."""

import optionlab as ol


def main():
    # Market setup
    spot = 100.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    market = ol.MarketData.from_flat(spot, r, q)

    # Option: European call, K=100, T=1Y
    option = ol.EuropeanOption(strike=100.0, T=1.0, option_type="call")

    # --- Analytical (Black-Scholes) ---
    analytical = ol.bs_price(option, market, sigma)
    print("=== Black-Scholes Analytical ===")
    print(f"  Price:  {analytical.price:.4f}")
    print(f"  Delta:  {analytical.delta:.4f}")
    print(f"  Gamma:  {analytical.gamma:.4f}")
    print(f"  Vega:   {analytical.vega:.4f}")
    print(f"  Theta:  {analytical.theta:.6f}")
    print(f"  Rho:    {analytical.rho:.4f}")
    print(f"  IV:     {analytical.iv:.4f}")

    # --- Monte Carlo ---
    dynamics = ol.GBM(r=r, q=q, sigma=sigma)
    engine = ol.MonteCarloEngine(n_paths=200_000, n_steps=252, seed=42, antithetic=True)
    mc_result = engine.price(option, dynamics, market)

    print("\n=== Monte Carlo (200K paths, antithetic) ===")
    print(f"  Price:  {mc_result.price:.4f} ± {mc_result.stderr:.4f}")
    print(f"  IV:     {mc_result.iv:.4f}")

    # --- MC Greeks (bump-and-reprice) ---
    greeks = ol.compute_greeks(engine, option, dynamics, market)
    print("\n=== MC Greeks (bump-and-reprice) ===")
    print(f"  Delta:  {greeks.delta:.4f}")
    print(f"  Gamma:  {greeks.gamma:.4f}")
    print(f"  Vega:   {greeks.vega:.4f}")
    print(f"  Theta:  {greeks.theta:.6f}")
    print(f"  Rho:    {greeks.rho:.4f}")

    # --- Comparison ---
    diff = abs(mc_result.price - analytical.price)
    print(f"\n  MC vs BS price difference: {diff:.4f}")
    print(f"  Within 2 stderr: {diff < 2 * mc_result.stderr}")


if __name__ == "__main__":
    main()
