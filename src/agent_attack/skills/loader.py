from __future__ import annotations

import json
from pathlib import Path

from agent_attack.memory.skills import Skill


class SkillStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, skill: Skill) -> Path:
        path = self.root / f"{skill.name}.json"
        payload = {
            "name": skill.name,
            "preconditions": [tag.value for tag in skill.preconditions],
            "policy_steps": skill.policy_steps,
            "parameters": skill.parameters,
            "termination": skill.termination,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
