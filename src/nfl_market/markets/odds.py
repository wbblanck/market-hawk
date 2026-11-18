import math


def _check_american(odds) -> int:
    if isinstance(odds, bool):
        raise ValueError("American odds must be numeric")
    try:
        value = float(odds)
    except (TypeError, ValueError) as e:
        raise ValueError(f"American odds must be numeric, got {odds!r}") from e
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"American odds must be a whole number, got {odds!r}")
    value = int(value)
    if value == 0:
        raise ValueError("American odds cannot be zero")
    if -100 < value < 100:
        raise ValueError(f"American odds must be <= -100 or >= +100, got {value}")
    return value


def is_valid_american(odds) -> bool:
    try:
        _check_american(odds)
        return True
    except ValueError:
        return False


def american_to_decimal(odds: int) -> float:
    odds = _check_american(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def implied_probability(odds: int) -> float:
    odds = _check_american(odds)
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def devig_two_way(a: int, b: int) -> tuple[float, float]:
    """Multiplicative (proportional) devig of a two-way market."""
    pa, pb = implied_probability(a), implied_probability(b)
    return pa / (pa + pb), pb / (pa + pb)
