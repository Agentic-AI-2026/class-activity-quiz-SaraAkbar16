from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import graph  # noqa: E402


class FakeMessage:
	def __init__(self, content: str):
		self.content = content


class FakeLLM:
	def __init__(self) -> None:
		self.calls: list[list[object]] = []

	def invoke(self, messages):
		self.calls.append(messages)
		if messages and isinstance(messages[0], graph.SystemMessage):
			plan = [
				{
					"step": 1,
					"description": "Calculate the tables and chairs needed for 150 people.",
					"tool": "calculator",
					"args": {"expression": "150 / 6"},
				},
				{
					"step": 2,
					"description": "Find the average ticket price for a similar outdoor event.",
					"tool": "search_web",
					"args": {"query": "average ticket price outdoor event"},
				},
				{
					"step": 3,
					"description": "Check the weather in Karachi for the event date.",
					"tool": "get_current_weather",
					"args": {"city": "Karachi"},
				},
				{
					"step": 4,
					"description": "Summarize the full event plan.",
					"tool": None,
					"args": None,
				},
			]
			return FakeMessage(json.dumps(plan))

		context = messages[-1].content if messages else ""
		return FakeMessage(f"SYNTHESIS: {context.splitlines()[0] if context else 'done'}")


class FakeTool:
	def __init__(self, name: str, result_fn):
		self.name = name
		self.calls: list[dict[str, object]] = []
		self._result_fn = result_fn

	async def ainvoke(self, args):
		self.calls.append(args)
		return self._result_fn(args)


class GraphSmokeTest(unittest.TestCase):
	def test_run_goal_sync(self) -> None:
		fake_llm = FakeLLM()
		calculator_tool = FakeTool(
			"calculator",
			lambda args: f"{args['expression']} = 25",
		)
		search_tool = FakeTool(
			"search_web",
			lambda args: "Average ticket price: $30",
		)
		weather_tool = FakeTool(
			"get_current_weather",
			lambda args: "Current weather in Karachi: Sunny, 30 C",
		)

		async def fake_get_mcp_tools():
			return {
				"calculator": calculator_tool,
				"search_web": search_tool,
				"get_current_weather": weather_tool,
			}

		with patch.object(graph, "get_llm", return_value=fake_llm), patch.object(
			graph, "get_mcp_tools", side_effect=fake_get_mcp_tools
		):
			final_state = graph.run_goal_sync(
				"Plan an outdoor event for 150 people: calculate tables/chairs, find average ticket price, check weather, and summarize."
			)

		self.assertEqual(final_state["current_step"], 4)
		self.assertEqual(len(final_state["plan"]), 4)
		self.assertEqual(len(final_state["results"]), 4)
		self.assertEqual(calculator_tool.calls[0], {"expression": "150 / 6"})
		self.assertEqual(search_tool.calls[0], {"query": "average ticket price outdoor event"})
		self.assertEqual(weather_tool.calls[0], {"city": "Karachi"})
		self.assertEqual(len(fake_llm.calls), 2)
		self.assertIn("SYNTHESIS", final_state["results"][-1])


if __name__ == "__main__":
	unittest.main(verbosity=2)

