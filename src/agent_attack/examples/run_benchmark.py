from __future__ import annotations

import argparse

from agent_attack.runtime.benchmark import dump_results, load_benchmark, run_agent_attack, run_baseline_single_turn
from agent_attack.runtime.engine import AttackConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AgentAttack benchmark in baseline or agent mode")
    parser.add_argument("--dataset", required=True, help="Path to benchmark JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument("--mode", choices=["baseline", "agent"], required=True)

    parser.add_argument("--target-provider", required=True, choices=["vllm", "openai", "gemini", "anthropic"])
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-api-key", default=None)
    parser.add_argument("--target-base-url", default=None)

    parser.add_argument("--judge-provider", choices=["vllm", "openai", "gemini", "anthropic"], default="openai")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-base-url", default=None)

    parser.add_argument("--objective", default="Probe policy consistency against unsafe intent over multiple turns.")
    parser.add_argument("--subgoals", nargs="*", default=["trigger refusal", "recover", "measure drift"])
    parser.add_argument("--constraints", nargs="*", default=["no harmful content", "high-level only"])
    parser.add_argument("--max-budget", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    items = load_benchmark(args.dataset)

    if args.mode == "baseline":
        outputs = run_baseline_single_turn(
            items,
            provider=args.target_provider,
            model=args.target_model,
            api_key=args.target_api_key,
            base_url=args.target_base_url,
        )
    else:
        cfg = AttackConfig(
            seed_prompt="dataset_prompt_will_override",
            objective=args.objective,
            subgoals=list(args.subgoals),
            constraints=list(args.constraints),
            target_provider=args.target_provider,
            target_model=args.target_model,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            target_api_key=args.target_api_key,
            target_base_url=args.target_base_url,
            judge_api_key=args.judge_api_key,
            judge_base_url=args.judge_base_url,
            max_budget=args.max_budget,
            beam_width=args.beam_width,
        )
        outputs = run_agent_attack(items, cfg)

    dump_results(args.output, outputs)
    print(f"done: {len(outputs)} samples -> {args.output}")


if __name__ == "__main__":
    main()
