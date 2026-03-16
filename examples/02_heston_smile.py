"""Example 2: Price across strikes under Heston, plot the implied volatility smile."""

import numpy as np

import optionlab as ol


def main():
    # Market setup
    spot = 100.0
    r = 0.05
    q = 0.0
    market = ol.MarketData.from_flat(spot, r, q)

    # Heston parameters (typical equity-like)
    dynamics = ol.Heston(
        r=r, q=q,
        V0=0.04,       # initial variance (σ=20%)
        kappa=2.0,      # mean-reversion speed
        theta=0.04,     # long-run variance
        xi=0.5,         # vol-of-vol
        rho=-0.7,       # spot-vol correlation (negative skew)
    )

    engine = ol.MonteCarloEngine(
        n_paths=100_000,
        n_steps=252,
        seed=42,
        antithetic=True,
    )

    # Strike range: 70% to 130% of spot
    strikes = np.linspace(70, 130, 13)
    T = 0.5  # 6 months

    print(f"{'Strike':>8} {'Price':>10} {'Stderr':>10} {'IV':>10}")
    print("-" * 42)

    ivs = []
    for K in strikes:
        option = ol.EuropeanOption(strike=K, T=T, option_type="call")
        result = engine.price(option, dynamics, market)
        ivs.append(result.iv)
        print(f"{K:8.1f} {result.price:10.4f} {result.stderr:10.4f} {result.iv:10.4f}")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(strikes, np.array(ivs) * 100, "o-", linewidth=2, markersize=6)
        ax.set_xlabel("Strike", fontsize=12)
        ax.set_ylabel("Implied Volatility (%)", fontsize=12)
        ax.set_title("Heston Implied Volatility Smile (T=0.5Y)", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("heston_smile.png", dpi=150)
        print("\nPlot saved to heston_smile.png")
    except ImportError:
        print("\nInstall matplotlib to generate the smile plot.")


if __name__ == "__main__":
    main()
