from __future__ import annotations

import asyncio
import math
import json
import os
import re
import sys
from typing import Any, TypedDict

import requests

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

try:
	from api_keys import ANTHROPIC_API_KEY, GEMINI_API_KEY
except Exception:
	ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
	GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools in this workspace and their argument names:
  - calculator(expression: str)
  - add(a: float, b: float)
  - subtract(a: float, b: float)
  - multiply(a: float, b: float)
  - divide(a: float, b: float)
  - power(base: float, exponent: float)
  - square_root(number: float)
  - search_web(query: str)
  - search_news(query: str)
  - get_current_weather(city: str)
  - get_weather_forecast(city: str, days: int)

Use null for tool/args on synthesis or writing steps.
Use calculator for arithmetic, search_web for lookups, and get_current_weather for current weather.
Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ALIASES = {
	"fetch_wikipedia": "search_web",
	"fetch_data_source": "search_web",
	"get_weather": "get_current_weather",
}

EXPECTED_ARG_ORDER = {
	"calculator": ("expression",),
	"add": ("a", "b"),
	"subtract": ("a", "b"),
	"multiply": ("a", "b"),
	"divide": ("a", "b"),
	"power": ("base", "exponent"),
	"square_root": ("number",),
	"search_web": ("query",),
	"search_news": ("query",),
	"get_current_weather": ("city",),
	"get_weather_forecast": ("city", "days"),
}

CITY_COORDS = {
	"london": (51.5074, -0.1278),
	"paris": (48.8566, 2.3522),
	"new york": (40.7128, -74.0060),
	"tokyo": (35.6762, 139.6503),
	"karachi": (24.8607, 67.0011),
	"lahore": (31.5204, 74.3587),
	"islamabad": (33.6844, 73.0479),
	"rawalpindi": (33.5651, 73.0169),
	"dubai": (25.2048, 55.2708),
	"berlin": (52.5200, 13.4050),
	"sydney": (-33.8688, 151.2093),
	"chicago": (41.8781, -87.6298),
}


class GraphState(TypedDict):
	# goal: original user request for the full run
	goal: str
	# plan: human-readable step descriptions shown in state
	plan: list[str]
	# step_details: structured steps used by executor logic
	step_details: list[dict[str, Any]]
	# current_step: zero-based pointer to the next step to run
	current_step: int
	# results: accumulated outputs in execution order
	results: list[str]


_mcp_client = MultiServerMCPClient(
	{
		"math": {
			"command": sys.executable,
			"args": [os.path.join(os.path.dirname(__file__), "Tools", "math_server.py")],
			"transport": "stdio",
		},
		"search": {
			"command": sys.executable,
			"args": [os.path.join(os.path.dirname(__file__), "Tools", "search_server.py")],
			"transport": "stdio",
		},
		"weather": {
			"url": "http://localhost:8000/mcp",
			"transport": "streamable_http",
		},
	}
)

_tool_cache: dict[str, Any] | None = None
_llm_cache: Any | None = None


class LocalTool:
	def __init__(self, name: str, func):
		self.name = name
		self._func = func

	async def ainvoke(self, args: dict[str, Any]) -> Any:
		return self._func(args)


def safe_calculate(expression: str) -> str:
	allowed_globals = {
		"__builtins__": {},
		"sqrt": math.sqrt,
		"log": math.log,
		"log2": math.log2,
		"log10": math.log10,
		"sin": math.sin,
		"cos": math.cos,
		"tan": math.tan,
		"ceil": math.ceil,
		"floor": math.floor,
		"pi": math.pi,
		"e": math.e,
		"abs": abs,
		"round": round,
		"pow": pow,
	}
	try:
		result = eval(expression, allowed_globals)
		return f"{expression} = {round(float(result), 6)}"
	except Exception as exc:
		return f"Error evaluating '{expression}': {exc}"


def weather_condition_from_code(code: int) -> str:
	if code < 3:
		return "Sunny"
	if code < 50:
		return "Cloudy"
	return "Rainy"


def local_current_weather(city: str) -> str:
	city_key = city.lower().strip()
	coords = CITY_COORDS.get(city_key)
	if not coords:
		available = ", ".join(c.title() for c in CITY_COORDS)
		return f"City '{city}' not found. Available cities: {available}"

	lat, lon = coords
	url = (
		f"https://api.open-meteo.com/v1/forecast"
		f"?latitude={lat}&longitude={lon}"
		f"&current_weather=true"
		f"&hourly=relativehumidity_2m,apparent_temperature"
	)
	try:
		data = requests.get(url, timeout=8).json()
		cw = data.get("current_weather", {})
		temp = cw.get("temperature", "N/A")
		wind = cw.get("windspeed", "N/A")
		wcode = int(cw.get("weathercode", 0))
		humidity = data.get("hourly", {}).get("relativehumidity_2m", ["N/A"])[0]
		feels = data.get("hourly", {}).get("apparent_temperature", ["N/A"])[0]
		cond = weather_condition_from_code(wcode)
		return (
			f"Current weather in {city_key.title()}:\n"
			f"  Condition : {cond}\n"
			f"  Temp      : {temp} C\n"
			f"  Feels like: {feels} C\n"
			f"  Wind      : {wind} km/h\n"
			f"  Humidity  : {humidity}%"
		)
	except requests.Timeout:
		return f"Weather API timed out for '{city}'"
	except Exception as exc:
		return f"Weather API error: {exc}"


def local_weather_forecast(city: str, days: int) -> str:
	if days < 1 or days > 7:
		return "Please provide a number of days between 1 and 7."

	city_key = city.lower().strip()
	coords = CITY_COORDS.get(city_key)
	if not coords:
		return f"City '{city}' not found."

	lat, lon = coords
	url = (
		f"https://api.open-meteo.com/v1/forecast"
		f"?latitude={lat}&longitude={lon}"
		f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
		f"&forecast_days={days}"
		f"&timezone=auto"
	)
	try:
		data = requests.get(url, timeout=8).json()
		daily = data.get("daily", {})
		dates = daily.get("time", [])
		max_temps = daily.get("temperature_2m_max", [])
		min_temps = daily.get("temperature_2m_min", [])
		wcodes = daily.get("weathercode", [])

		lines = [f"Forecast for {city_key.title()} ({days} days):"]
		for index in range(min(days, len(dates))):
			wcode = int(wcodes[index]) if index < len(wcodes) else 0
			cond = weather_condition_from_code(wcode)
			lines.append(
				f"  {dates[index]} : {cond}, High {max_temps[index]}C / Low {min_temps[index]}C"
			)
		return "\n".join(lines)
	except Exception as exc:
		return f"Forecast error: {exc}"


def build_local_tool_map() -> dict[str, Any]:
	return {
		"calculator": LocalTool("calculator", lambda args: safe_calculate(str(args.get("expression", "")))),
		"add": LocalTool("add", lambda args: args.get("a", 0) + args.get("b", 0)),
		"subtract": LocalTool("subtract", lambda args: args.get("a", 0) - args.get("b", 0)),
		"multiply": LocalTool("multiply", lambda args: args.get("a", 0) * args.get("b", 0)),
		"divide": LocalTool(
			"divide",
			lambda args: "Error: Cannot divide by zero"
			if args.get("b", 0) == 0
			else round(args.get("a", 0) / args.get("b", 1), 6),
		),
		"power": LocalTool("power", lambda args: round(args.get("base", 0) ** args.get("exponent", 0), 6)),
		"square_root": LocalTool(
			"square_root",
			lambda args: "Error: Cannot take square root of a negative number"
			if args.get("number", 0) < 0
			else str(round(math.sqrt(args.get("number", 0)), 6)),
		),
		"search_web": LocalTool(
			"search_web",
			lambda args: f"Offline web lookup for '{args.get('query', '')}': live search is unavailable.",
		),
		"search_news": LocalTool(
			"search_news",
			lambda args: f"Offline news lookup for '{args.get('query', '')}': live search is unavailable.",
		),
		"get_current_weather": LocalTool(
			"get_current_weather",
			lambda args: local_current_weather(str(args.get("city", "Karachi"))),
		),
		"get_weather_forecast": LocalTool(
			"get_weather_forecast",
			lambda args: local_weather_forecast(str(args.get("city", "Karachi")), int(args.get("days", 3))),
		),
	}


def infer_city_from_goal(goal: str) -> str | None:
	known_cities = [
		"karachi",
		"lahore",
		"islamabad",
		"rawalpindi",
		"dubai",
		"berlin",
		"sydney",
		"chicago",
		"london",
		"paris",
		"tokyo",
		"new york",
	]
	goal_lower = goal.lower()
	for city in known_cities:
		if city in goal_lower:
			return city.title()
	return None


def build_offline_plan(goal: str) -> list[dict[str, Any]]:
	goal_lower = goal.lower()
	steps: list[dict[str, Any]] = []
	step_number = 1

	if any(keyword in goal_lower for keyword in ("calculate", "tables", "chairs", "math", "sum", "number")):
		people_match = re.search(r"(\d+)\s*people", goal_lower)
		people_count = int(people_match.group(1)) if people_match else 150
		steps.append(
			{
				"step": step_number,
				"description": f"Calculate tables and chairs needed for {people_count} people.",
				"tool": "calculator",
				"args": {"expression": f"{people_count} / 6"},
			}
		)
		step_number += 1

	if any(keyword in goal_lower for keyword in ("ticket", "price", "average", "cost", "budget")):
		steps.append(
			{
				"step": step_number,
				"description": "Find the average ticket price for a similar outdoor event.",
				"tool": "search_web",
				"args": {"query": "average ticket price outdoor event"},
			}
		)
		step_number += 1

	if "weather" in goal_lower:
		city = infer_city_from_goal(goal) or os.getenv("DEFAULT_CITY", "Karachi")
		steps.append(
			{
				"step": step_number,
				"description": f"Check the current weather in {city} for the event.",
				"tool": "get_current_weather",
				"args": {"city": city},
			}
		)
		step_number += 1

	steps.append(
		{
			"step": step_number,
			"description": "Summarize the event plan using the collected results.",
			"tool": None,
			"args": None,
		}
	)
	return steps


def offline_synthesis(step_description: str, context: str) -> str:
	context_lines = [line.strip() for line in context.splitlines() if line.strip()]
	if not context_lines:
		return f"{step_description}"
	joined = " | ".join(context_lines[-3:])
	return f"{step_description} Summary: {joined}"


def stringify_tool_result(result: Any) -> str:
	if isinstance(result, str):
		return result
	if hasattr(result, "content"):
		content = result.content
		return content if isinstance(content, str) else str(content)
	if isinstance(result, list):
		parts: list[str] = []
		for item in result:
			if isinstance(item, dict) and "text" in item:
				parts.append(str(item["text"]))
			elif hasattr(item, "content"):
				parts.append(str(item.content))
			else:
				parts.append(str(item))
		return "\n".join(parts)
	if isinstance(result, dict) and "text" in result:
		return str(result["text"])
	return str(result)


def safe_llm_text(messages: list[Any], *, purpose: str, goal: str = "", context: str = "") -> str:
	try:
		llm = get_llm()
		response = llm.invoke(messages)
		return response.content if hasattr(response, "content") else str(response)
	except Exception as exc:
		if purpose == "plan":
			print(f"Planner LLM unavailable, using offline fallback: {exc}")
			return json.dumps(build_offline_plan(goal))
		if purpose == "synthesis":
			print(f"Synthesis LLM unavailable, using offline fallback: {exc}")
			return offline_synthesis(goal, context)
		raise


def get_llm() -> Any:
	"""Create a small, dependency-tolerant LLM wrapper."""
	global _llm_cache
	if _llm_cache is not None:
		return _llm_cache

	model_name = os.getenv("OLLAMA_MODEL", "llama3.1")

	try:
		from langchain_ollama import ChatOllama

		_llm_cache = ChatOllama(model=model_name, temperature=0)
		return _llm_cache
	except Exception:
		pass

	anthropic_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
	if anthropic_key:
		try:
			from langchain_anthropic import ChatAnthropic

			_llm_cache = ChatAnthropic(
				model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
				temperature=0,
			)
			return _llm_cache
		except Exception:
			pass

	gemini_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
	if gemini_key:
		try:
			from langchain_google_genai import ChatGoogleGenerativeAI

			_llm_cache = ChatGoogleGenerativeAI(
				model=os.getenv("GEMINI_MODEL", os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")),
				temperature=0,
			)
			return _llm_cache
		except Exception:
			pass

	raise RuntimeError(
		"No supported LLM backend is available. Set OLLAMA_MODEL with Ollama running, "
		"or configure ANTHROPIC_API_KEY / GEMINI_API_KEY."
	)


async def get_mcp_tools() -> dict[str, Any]:
	global _tool_cache
	# Reuse cached tools so each loop iteration does not reconnect to MCP.
	if _tool_cache is not None:
		return _tool_cache

	tools: list[Any] = []
	for server_name in ("math", "search", "weather"):
		try:
			server_tools = await _mcp_client.get_tools(server_name=server_name)
		except Exception as exc:
			print(f"Skipping MCP server '{server_name}': {exc}")
			continue
		tools.extend(server_tools)

	tool_map = {tool.name: tool for tool in tools}
	# Keep local tools as a safety net when MCP servers are unavailable.
	for name, local_tool in build_local_tool_map().items():
		tool_map.setdefault(name, local_tool)
	_tool_cache = tool_map
	return _tool_cache


def extract_json_array(text: Any) -> list[dict[str, Any]]:
	if not isinstance(text, str):
		if hasattr(text, "content") and isinstance(text.content, str):
			text = text.content
		else:
			text = str(text)

	cleaned = re.sub(r"```json|```", "", text).strip()
	parsed = json.loads(cleaned)
	if isinstance(parsed, dict) and "plan" in parsed:
		parsed = parsed["plan"]
	if not isinstance(parsed, list):
		raise ValueError("Planner output must be a JSON array of steps.")

	normalized: list[dict[str, Any]] = []
	for index, raw_step in enumerate(parsed, start=1):
		if not isinstance(raw_step, dict):
			raise ValueError("Each plan step must be an object.")
		normalized.append(
			{
				"step": int(raw_step.get("step", index)),
				"description": str(raw_step.get("description", "")),
				"tool": raw_step.get("tool"),
				"args": raw_step.get("args"),
			}
		)
	normalized.sort(key=lambda item: item["step"])
	return normalized


def resolve_tool_name(tool_name: str | None) -> str | None:
	if not tool_name:
		return None
	if tool_name.lower() in {"null", "none"}:
		return None
	return TOOL_ALIASES.get(tool_name, tool_name)


def normalize_args(tool_name: str, raw_args: Any) -> dict[str, Any]:
	expected = EXPECTED_ARG_ORDER.get(tool_name)
	if not expected:
		return raw_args if isinstance(raw_args, dict) else {}
	if isinstance(raw_args, dict) and all(key in raw_args for key in expected):
		return raw_args

	if not isinstance(raw_args, dict):
		raw_args = {}

	values = list(raw_args.values())
	normalized: dict[str, Any] = {}
	for position, key in enumerate(expected):
		if key in raw_args:
			normalized[key] = raw_args[key]
		elif position < len(values):
			normalized[key] = values[position]

	if tool_name == "search_web" and "query" not in normalized:
		normalized["query"] = raw_args.get("topic") or raw_args.get("source") or raw_args.get("description") or ""
	if tool_name == "get_current_weather" and "city" not in normalized:
		normalized["city"] = raw_args.get("location") or raw_args.get("query") or ""
	if tool_name == "calculator" and "expression" not in normalized:
		normalized["expression"] = raw_args.get("query") or raw_args.get("description") or ""

	return normalized


async def planner_node(state: GraphState) -> dict[str, Any]:
	# Phase 1: generate the full ordered plan from the goal.
	raw_plan = safe_llm_text(
		[SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=state["goal"])],
		purpose="plan",
		goal=state["goal"],
	)
	step_details = extract_json_array(raw_plan)
	plan = [step["description"] for step in step_details]
	print(f"Plan ({len(step_details)} steps):")
	for step in step_details:
		print(f"  Step {step['step']}: {step['description']} | tool={step.get('tool')}")

	return {
		"plan": plan,
		"step_details": step_details,
		"current_step": 0,
		"results": [],
	}


async def executor_node(state: GraphState) -> dict[str, Any]:
	plan = state.get("step_details", [])
	current_step = state.get("current_step", 0)
	results = list(state.get("results", []))

	# Stop guard: nothing to execute if all steps are already completed.
	if current_step >= len(plan):
		return {"results": results, "current_step": current_step}

	# Phase 2: execute exactly one step per node invocation.
	step = plan[current_step]
	print(f"Executing step {step.get('step', current_step + 1)}: {step.get('description', '')}")

	tool_map = await get_mcp_tools()
	tool_name = resolve_tool_name(step.get("tool"))

	# Tool path: call the selected tool with normalized arguments.
	if tool_name and tool_name in tool_map:
		args = normalize_args(tool_name, step.get("args") or {})
		if tool_name == "search_web" and not args.get("query"):
			args["query"] = step.get("description", "")
		if tool_name == "get_current_weather" and not args.get("city"):
			args["city"] = step.get("description", "")
		result = await tool_map[tool_name].ainvoke(args)
	else:
		# Synthesis path: no tool required, use LLM reasoning with prior context.
		context = "\n".join(results)
		step_description = step.get("description", "")
		prompt = step_description
		if context:
			prompt = f"{prompt}\n\nContext:\n{context}"
		result = safe_llm_text([HumanMessage(content=prompt)], purpose="synthesis", goal=step_description, context=context)

	result_text = stringify_tool_result(result)
	print(f"  Result: {result_text[:200]}")

	results.append(result_text)
	# Advance by one so the graph loop runs the next step on the next pass.
	return {
		"results": results,
		"current_step": current_step + 1,
	}


def route_after_execution(state: GraphState) -> str:
	# Continue looping until every planned step is executed.
	return "end" if state.get("current_step", 0) >= len(state.get("plan", [])) else "continue"


def build_graph() -> Any:
	# Graph flow: START -> planner -> executor -> (executor loop) -> END.
	workflow = StateGraph(GraphState)
	workflow.add_node("planner_node", planner_node)
	workflow.add_node("executor_node", executor_node)

	workflow.add_edge(START, "planner_node")
	workflow.add_edge("planner_node", "executor_node")
	workflow.add_conditional_edges(
		"executor_node",
		route_after_execution,
		{
			"continue": "executor_node",
			"end": END,
		},
	)
	return workflow.compile()


async def run_goal(goal: str) -> GraphState:
	app = build_graph()
	initial_state: GraphState = {
		"goal": goal,
		"plan": [],
		"step_details": [],
		"current_step": 0,
		"results": [],
	}
	return await app.ainvoke(initial_state)


def run_goal_sync(goal: str) -> GraphState:
	return asyncio.run(run_goal(goal))