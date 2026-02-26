from __future__ import annotations

from dataclasses import dataclass

from agent_attack.core.types import AttackGoal, SearchNode
from agent_attack.memory.skills import SkillLibrary
from agent_attack.planner.search import FrontierPlanner
from agent_attack.runtime.components import ConsistencyChecker, HeuristicTagger, MockVictim, PromptRealizer


@dataclass(slots=True)
class AttackConfig:
    seed_prompt: str
    objective: str
    subgoals: list[str]
    constraints: list[str]
    max_budget: int = 15
    beam_width: int = 4


class CATSAttackEngine:
    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self.skill_library = SkillLibrary()
        self.planner = FrontierPlanner(
            victim=MockVictim(),
            parser=HeuristicTagger(),
            checker=ConsistencyChecker(),
            realizer=PromptRealizer(),
            skill_library=self.skill_library,
            max_budget=config.max_budget,
            beam_width=config.beam_width,
        )

    def run(self) -> list[SearchNode]:
        goal = AttackGoal(
            objective=self.config.objective,
            subgoals=self.config.subgoals,
            constraints=self.config.constraints,
        )
        return self.planner.run(goal, seed_prompt=self.config.seed_prompt)
