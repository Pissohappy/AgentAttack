from __future__ import annotations

from agent_attack.core.interfaces import ActionRealizer
from agent_attack.core.types import Action, SearchNode


class PromptRealizer(ActionRealizer):
    def to_prompt(self, node: SearchNode, action: Action) -> str:
        return (
            f"Goal: {node.goal.objective}\n"
            f"Subgoals: {node.goal.subgoals}\n"
            f"Constraints: {node.goal.constraints}\n"
            f"Action: {action.name}\n"
            f"Action payload: {action.payload}\n"
            "Generate the next response while maintaining conversation consistency."
        )
