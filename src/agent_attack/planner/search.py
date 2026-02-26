from __future__ import annotations

from dataclasses import replace
from heapq import heappop, heappush
from itertools import count
from uuid import uuid4

from agent_attack.core.interfaces import ActionRealizer, Checker, ParserTagger, VictimModel
from agent_attack.core.types import Action, AttackGoal, SearchNode, SearchState
from agent_attack.memory.skills import SkillLibrary


class FrontierPlanner:
    """Best-first planner with explicit backtracking via frontier."""

    def __init__(
        self,
        victim: VictimModel,
        parser: ParserTagger,
        checker: Checker,
        realizer: ActionRealizer,
        skill_library: SkillLibrary,
        max_budget: int = 20,
        beam_width: int = 5,
    ) -> None:
        self.victim = victim
        self.parser = parser
        self.checker = checker
        self.realizer = realizer
        self.skill_library = skill_library
        self.max_budget = max_budget
        self.beam_width = beam_width
        self._counter = count()

    def run(self, goal: AttackGoal, seed_prompt: str) -> list[SearchNode]:
        root = SearchNode(
            node_id=str(uuid4()),
            parent_id=None,
            depth=0,
            goal=goal,
            state=SearchState(history=[seed_prompt]),
        )
        frontier: list[tuple[float, int, SearchNode]] = []
        heappush(frontier, (-root.score, next(self._counter), root))
        explored: list[SearchNode] = [root]

        while frontier and len(explored) < self.max_budget:
            _, _, node = heappop(frontier)
            for child in self._expand(node):
                explored.append(child)
                if self.checker.should_prune(child):
                    continue
                heappush(frontier, (-child.score, next(self._counter), child))

            if len(frontier) > self.beam_width:
                frontier = sorted(frontier, key=lambda x: x[0])[: self.beam_width]

        return explored

    def _expand(self, node: SearchNode) -> list[SearchNode]:
        actions = self._candidate_actions(node)
        children: list[SearchNode] = []
        for action in actions:
            prompt = self.realizer.to_prompt(node, action)
            response = self.victim.respond(prompt)
            obs = self.parser.parse(response, node)

            state = replace(node.state)
            state.history = [*node.state.history, prompt, response]
            state.attempted_actions = [*node.state.attempted_actions, action.name]
            state.budget_used += 1

            child = SearchNode(
                node_id=str(uuid4()),
                parent_id=node.node_id,
                depth=node.depth + 1,
                goal=node.goal,
                state=state,
                action=action,
                observation=obs,
            )
            child.score = self.checker.score(node, child)
            self.skill_library.observe_transition(node, child)
            children.append(child)
        return children

    def _candidate_actions(self, node: SearchNode) -> list[Action]:
        operators = [
            Action(name="reframe_goal", payload={"style": "indirect"}),
            Action(name="ask_clarification", payload={"tone": "neutral"}),
        ]
        skills = self.skill_library.suggest(node)
        return operators + skills
