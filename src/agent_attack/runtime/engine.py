from __future__ import annotations

from dataclasses import dataclass

from agent_attack.core.types import AttackGoal, SearchNode
from agent_attack.memory.skills import SkillLibrary
from agent_attack.planner.search import FrontierPlanner
from agent_attack.runtime.components import PromptRealizer
from agent_attack.runtime.judge import LLMPromptJudge
from agent_attack.runtime.model_clients import ClientConfig, HTTPModelClient
from agent_attack.skills.attack_techniques import TechniqueLibrary


@dataclass(slots=True)
class AttackConfig:
    seed_prompt: str
    objective: str
    subgoals: list[str]
    constraints: list[str]
    target_provider: str
    target_model: str
    judge_provider: str
    judge_model: str
    target_api_key: str | None = None
    target_base_url: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    max_budget: int = 15
    beam_width: int = 4


class CATSAttackEngine:
    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self.skill_library = SkillLibrary()
        self.technique_library = TechniqueLibrary()

        target_client = HTTPModelClient(
            ClientConfig(
                provider=config.target_provider,
                model=config.target_model,
                api_key=config.target_api_key,
                base_url=config.target_base_url,
            )
        )
        judge_client = HTTPModelClient(
            ClientConfig(
                provider=config.judge_provider,
                model=config.judge_model,
                api_key=config.judge_api_key,
                base_url=config.judge_base_url,
                temperature=0.0,
            )
        )
        judge = LLMPromptJudge(judge_client)
        self.planner = FrontierPlanner(
            victim=target_client,
            parser=judge,
            checker=judge,
            realizer=PromptRealizer(),
            skill_library=self.skill_library,
            technique_library=self.technique_library,
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
