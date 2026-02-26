from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObservationTag(str, Enum):
    REFUSAL = "refusal"
    DRIFT = "drift"
    REPETITION = "repetition"
    PROGRESS = "progress"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AttackGoal:
    objective: str
    subgoals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Observation:
    raw_response: str
    tags: list[ObservationTag]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "operator"


@dataclass(slots=True)
class SearchState:
    history: list[str] = field(default_factory=list)
    attempted_actions: list[str] = field(default_factory=list)
    budget_used: int = 0


@dataclass(slots=True)
class SearchNode:
    node_id: str
    parent_id: str | None
    depth: int
    goal: AttackGoal
    state: SearchState
    action: Action | None = None
    observation: Observation | None = None
    score: float = 0.0

    def signature(self) -> tuple[str, ...]:
        return tuple(self.state.attempted_actions)
