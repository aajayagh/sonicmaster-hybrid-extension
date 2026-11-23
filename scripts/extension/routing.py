"""Degradation routing for 10_extension."""
from __future__ import annotations

from dataclasses import dataclass

P_TYPES = frozenset({
    "volume", "stereo", "comp", "punch",
})

R_TYPES = frozenset({
    "clarity", "boom", "airy", "bright", "dark", "mud", "warm", "vocal",
    "mic",
    "small", "big", "mix", "real", "noise",
})

S_ONLY_TYPES = frozenset({
    "clip", "xband",
})


@dataclass(frozen=True)
class RouteDecision:
    mode: str  # "P", "R", "R_then_P", "fallback_s"
    p_types: tuple[str, ...]
    r_types: tuple[str, ...]
    unknown_types: tuple[str, ...]


def route_degradations(degradation_types: list[str] | tuple[str, ...]) -> RouteDecision:
    cleaned = [t.strip().lower() for t in degradation_types if t and t.strip()]
    if not cleaned:
        return RouteDecision(mode="fallback_s", p_types=(), r_types=(), unknown_types=())

    p_list = tuple(t for t in cleaned if t in P_TYPES)
    r_list = tuple(t for t in cleaned if t in R_TYPES)
    s_list = tuple(t for t in cleaned if t in S_ONLY_TYPES)
    unk = tuple(t for t in cleaned if t not in P_TYPES and t not in R_TYPES and t not in S_ONLY_TYPES)

    if unk:
        return RouteDecision(mode="fallback_s", p_types=p_list, r_types=r_list, unknown_types=unk)
    if s_list:
        return RouteDecision(mode="S", p_types=(), r_types=(), unknown_types=())
    if r_list and p_list:
        return RouteDecision(mode="R_then_P", p_types=p_list, r_types=r_list, unknown_types=())
    if r_list:
        return RouteDecision(mode="R", p_types=(), r_types=r_list, unknown_types=())
    return RouteDecision(mode="P", p_types=p_list, r_types=(), unknown_types=())
