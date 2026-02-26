from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

from agent_attack.core.types import Action, ObservationTag, SearchNode


@dataclass(slots=True)
class Skill:
    name: str
    preconditions: list[ObservationTag] = field(default_factory=list)
    policy_steps: list[str] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)
    termination: str = "on_progress_or_refusal"
    usage_scores: list[float] = field(default_factory=list)

    def to_action(self) -> Action:
        return Action(
            name=self.name,
            payload={"steps": self.policy_steps, **self.parameters},
            source="skill",
        )


class SkillLibrary:
    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {
            "gentle_context_bridge": Skill(
                name="gentle_context_bridge",
                preconditions=[ObservationTag.REFUSAL],
                policy_steps=["acknowledge", "shift_context", "retry_subgoal"],
                parameters={"directness": 0.3},
            )
        }

    def suggest(self, node: SearchNode) -> list[Action]:
        if not node.observation:
            return []
        tags = set(node.observation.tags)
        matches = [s for s in self.skills.values() if not s.preconditions or tags.intersection(s.preconditions)]
        ranked = sorted(matches, key=self._avg_score, reverse=True)
        return [s.to_action() for s in ranked[:2]]

    def observe_transition(self, parent: SearchNode, child: SearchNode) -> None:
        if not child.action or child.action.source != "skill":
            return
        skill = self.skills.get(child.action.name)
        if not skill:
            return
        score = 1.0 if child.observation and ObservationTag.PROGRESS in child.observation.tags else 0.0
        skill.usage_scores.append(score)
        if skill.usage_scores:
            skill.parameters["directness"] = max(0.1, min(0.9, 1 - fmean(skill.usage_scores) * 0.5))

    @staticmethod
    def _avg_score(skill: Skill) -> float:
        if not skill.usage_scores:
            return 0.5
        return fmean(skill.usage_scores)
