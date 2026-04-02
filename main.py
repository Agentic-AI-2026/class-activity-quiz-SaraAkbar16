from __future__ import annotations

import argparse

from graph import run_goal_sync

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the LangGraph planner-executor agent.")
	parser.add_argument(
		"goal",
		nargs="?",
		default="Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize.",
		help="User goal to plan and execute.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	final_state = run_goal_sync(args.goal)

	print("\nFinal results:")
	for index, result in enumerate(final_state["results"], start=1):
		print(f"{index}. {result}")


if __name__ == "__main__":
	main()
