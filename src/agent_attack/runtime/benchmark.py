from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_attack.runtime.engine import AttackConfig, CATSAttackEngine
from agent_attack.runtime.model_clients import ClientConfig, HTTPModelClient


@dataclass(slots=True)
class BenchmarkItem:
    sample_id: int
    original_prompt: str
    image_path: str | None
    style: str | None
    main_category: str | None
    subcategory: str | None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "BenchmarkItem":
        return cls(
            sample_id=int(row.get("id", -1)),
            original_prompt=str(row.get("original_prompt", "")),
            image_path=row.get("image_path"),
            style=row.get("style"),
            main_category=row.get("main_category"),
            subcategory=row.get("subcategory"),
        )


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Benchmark file must be a JSON array")
    return [BenchmarkItem.from_dict(item) for item in raw]


def run_baseline_single_turn(items: list[BenchmarkItem], *, provider: str, model: str, api_key: str | None = None, base_url: str | None = None) -> list[dict[str, Any]]:
    victim = HTTPModelClient(
        ClientConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    )
    outputs: list[dict[str, Any]] = []
    for item in items:
        response = victim.respond(item.original_prompt)
        outputs.append(
            {
                "id": item.sample_id,
                "mode": "baseline_single_turn",
                "original_prompt": item.original_prompt,
                "response": response,
                "meta": {
                    "main_category": item.main_category,
                    "subcategory": item.subcategory,
                    "style": item.style,
                    "image_path": item.image_path,
                },
            }
        )
    return outputs


def run_agent_attack(items: list[BenchmarkItem], config: AttackConfig) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for item in items:
        per_item_cfg = AttackConfig(
            seed_prompt=item.original_prompt,
            objective=config.objective,
            subgoals=config.subgoals,
            constraints=config.constraints,
            target_provider=config.target_provider,
            target_model=config.target_model,
            judge_provider=config.judge_provider,
            judge_model=config.judge_model,
            target_api_key=config.target_api_key,
            target_base_url=config.target_base_url,
            judge_api_key=config.judge_api_key,
            judge_base_url=config.judge_base_url,
            max_budget=config.max_budget,
            beam_width=config.beam_width,
        )
        engine = CATSAttackEngine(per_item_cfg)
        nodes = engine.run()
        outputs.append(
            {
                "id": item.sample_id,
                "mode": "agent_attack",
                "original_prompt": item.original_prompt,
                "trajectory": [
                    {
                        "node_id": n.node_id,
                        "depth": n.depth,
                        "action": n.action.name if n.action else "root",
                        "action_source": n.action.source if n.action else "root",
                        "score": n.score,
                        "tags": [t.value for t in n.observation.tags] if n.observation else [],
                        "response": n.observation.raw_response if n.observation else "",
                    }
                    for n in nodes
                ],
                "meta": {
                    "main_category": item.main_category,
                    "subcategory": item.subcategory,
                    "style": item.style,
                    "image_path": item.image_path,
                },
            }
        )
    return outputs


def dump_results(path: str | Path, data: list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
