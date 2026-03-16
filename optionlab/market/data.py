"""Market data container."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .curves import DiscountCurve, ForwardCurve


@dataclass
class MarketData:
    """Snapshot of market data needed for pricing.

    Parameters
    ----------
    spot : float
        Current spot price.
    discount_curve : DiscountCurve
        Risk-free discount curve.
    forward_curve : ForwardCurve
        Forward price curve.
    """

    spot: float
    discount_curve: DiscountCurve
    forward_curve: ForwardCurve

    @classmethod
    def from_flat(cls, spot: float, rate: float, dividend_yield: float = 0.0) -> MarketData:
        """Construct MarketData from flat rate and dividend yield."""
        return cls(
            spot=spot,
            discount_curve=DiscountCurve(rate),
            forward_curve=ForwardCurve(spot, rate, dividend_yield),
        )

    def bump_spot(self, bump: float) -> MarketData:
        """Return a new MarketData with spot bumped by an absolute amount."""
        new_spot = self.spot + bump
        return replace(
            self,
            spot=new_spot,
            forward_curve=ForwardCurve(
                new_spot,
                self.forward_curve.rate,
                self.forward_curve.dividend_yield,
            ),
        )
