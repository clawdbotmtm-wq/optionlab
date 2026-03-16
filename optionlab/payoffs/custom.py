"""Lambda-based custom payoffs — pass any function of paths."""

import inspect
from collections.abc import Callable
from collections.abc import Mapping
from typing import Literal

from numpy.typing import NDArray

from .base import Payoff


StateCallStyle = Literal["none", "positional", "keyword"]
CustomPayoffFunc = Callable[[NDArray, NDArray], NDArray] | Callable[
    [NDArray, NDArray, Mapping[str, NDArray]],
    NDArray,
]


def _resolve_state_call_style(func: Callable[..., NDArray]) -> StateCallStyle:
    """Determine whether a callable accepts extra state paths."""
    sig = inspect.signature(func)
    sentinel = object()

    try:
        sig.bind(sentinel, sentinel, sentinel)
    except TypeError:
        pass
    else:
        return "positional"

    try:
        sig.bind(sentinel, sentinel, state_paths=sentinel)
    except TypeError:
        return "none"

    return "keyword"


class CustomPayoff(Payoff):
    """Custom payoff defined by an arbitrary function of paths.

    This lets you price any path-dependent claim without writing a new class.

    Parameters
    ----------
    func : callable
        Function returning undiscounted payoff per path.
        Supported signatures:
        - (paths: NDArray, t_grid: NDArray) -> NDArray
        - (paths: NDArray, t_grid: NDArray, state_paths: Mapping[str, NDArray])
          -> NDArray
    T : float
        Time to expiry in years.
    name : str
        Optional human-readable name.

    Examples
    --------
    >>> # Asian call option (arithmetic average)
    >>> asian_call = CustomPayoff(
    ...     func=lambda paths, t: np.maximum(paths.mean(axis=1) - 100, 0),
    ...     T=1.0,
    ...     name="Asian Call K=100",
    ... )

    >>> # Digital call paying $1 if S_T > K
    >>> digital = CustomPayoff(
    ...     func=lambda paths, t: (paths[:, -1] > 100).astype(float),
    ...     T=1.0,
    ...     name="Digital Call K=100",
    ... )
    """

    def __init__(
        self,
        func: CustomPayoffFunc,
        T: float,
        name: str = "CustomPayoff",
    ) -> None:
        self._func = func
        self._state_call_style = _resolve_state_call_style(func)
        self.T = T
        self.name = name

    @property
    def expiry(self) -> float:
        """Time to expiry in years."""
        return self.T

    def cashflows(
        self,
        paths: NDArray,
        t_grid: NDArray,
        state_paths: Mapping[str, NDArray] | None = None,
    ) -> NDArray:
        """Compute cashflows using the wrapped callable."""
        state_paths = {} if state_paths is None else state_paths

        if self._state_call_style == "positional":
            return self._func(paths, t_grid, state_paths)
        if self._state_call_style == "keyword":
            return self._func(paths, t_grid, state_paths=state_paths)
        return self._func(paths, t_grid)

    def __repr__(self) -> str:
        return f"CustomPayoff(name='{self.name}', T={self.T})"
