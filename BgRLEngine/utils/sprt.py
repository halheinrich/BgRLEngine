"""Sequential Probability Ratio Test (SPRT) for level promotion.

Tests H₀: p ≤ p₀ (not ready) vs H₁: p ≥ p₁ (ready to promote).
"""

from __future__ import annotations

import math


class SPRTResult:
    """Result constants for SPRT."""
    CONTINUE = "continue"
    ACCEPT = "accept"    # promote: checkpoint is good enough
    REJECT = "reject"    # don't promote: need more training


def sprt_test(
    wins: int,
    games: int,
    p0: float = 0.70,
    p1: float = 0.76,
    alpha: float = 0.05,
    beta: float = 0.10,
    hard_cap: int = 2000,
) -> str:
    """Run Sequential Probability Ratio Test.

    Args:
        wins: number of wins so far.
        games: total games played so far.
        p0: null hypothesis win rate.
        p1: alternative hypothesis win rate.
        alpha: false promotion rate.
        beta: missed promotion rate.
        hard_cap: maximum games before forced reject.

    Returns:
        SPRTResult: "accept", "reject", or "continue".
    """
    if games >= hard_cap:
        return SPRTResult.REJECT

    if games == 0:
        return SPRTResult.CONTINUE

    # Log-likelihood ratio boundaries
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))

    # Cumulative log-likelihood ratio
    losses = games - wins
    if p0 == 0 or p1 == 0 or p0 == 1 or p1 == 1:
        return SPRTResult.CONTINUE

    llr = (
        wins * math.log(p1 / p0)
        + losses * math.log((1 - p1) / (1 - p0))
    )

    if llr >= upper:
        return SPRTResult.ACCEPT
    elif llr <= lower:
        return SPRTResult.REJECT
    else:
        return SPRTResult.CONTINUE
