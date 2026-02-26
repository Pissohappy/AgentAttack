from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Action, Observation, SearchNode


class VictimModel(ABC):
    @abstractmethod
    def respond(self, prompt: str) -> str:
        raise NotImplementedError


class ParserTagger(ABC):
    @abstractmethod
    def parse(self, response: str, node: SearchNode) -> Observation:
        raise NotImplementedError


class Checker(ABC):
    @abstractmethod
    def score(self, node: SearchNode, child: SearchNode) -> float:
        raise NotImplementedError

    @abstractmethod
    def should_prune(self, child: SearchNode) -> bool:
        raise NotImplementedError


class ActionRealizer(ABC):
    @abstractmethod
    def to_prompt(self, node: SearchNode, action: Action) -> str:
        raise NotImplementedError
