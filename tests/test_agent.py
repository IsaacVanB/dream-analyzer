from __future__ import annotations

import json
import unittest
from datetime import date

from dream_analysis.agent import (
    AgentSearchRequiredError,
    AgentToolLimitError,
    DreamRagAgent,
)
from dream_analysis.models import SearchResult
from dream_analysis.ollama_client import OllamaGateway
from dream_analysis.tools import DreamSearchTool


class SequencedOllamaClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.chat_calls: list[dict] = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return self.responses.pop(0)


class FakeIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, date | None, date | None]] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, limit, start_date, end_date))
        return [
            SearchResult(
                dream_id="dream-1",
                document="A hidden room appeared behind the pantry.",
                metadata={"date": "1/2/2024"},
                distance=0.125,
            )
        ]


def tool_response(name: str, arguments: dict) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": name, "arguments": arguments}}
            ],
        }
    }


def final_response(content: str = "Grounded answer (dream-1, 1/2/2024).") -> dict:
    return {"message": {"role": "assistant", "content": content}}


class DreamRagAgentTests(unittest.TestCase):
    def make_agent(
        self,
        responses: list[dict],
    ) -> tuple[DreamRagAgent, SequencedOllamaClient, FakeIndex]:
        client = SequencedOllamaClient(responses)
        gateway = OllamaGateway(client=client)
        index = FakeIndex()
        agent = DreamRagAgent(
            ollama_gateway=gateway,
            search_tool=DreamSearchTool(index, result_limit=4),
        )
        return agent, client, index

    def test_agent_executes_search_and_returns_the_final_answer(self) -> None:
        agent, client, index = self.make_agent(
            [
                tool_response("search_dreams", {"query": "hidden room pantry"}),
                final_response(),
            ]
        )

        response = agent.answer(
            "What patterns appear in hidden-room dreams?",
            chat_model="tool-model",
            num_ctx=8192,
            num_predict=500,
            temperature=0.2,
        )

        self.assertEqual(response.answer, "Grounded answer (dream-1, 1/2/2024).")
        self.assertEqual(index.calls, [("hidden room pantry", 4, None, None)])
        self.assertEqual(len(response.tool_executions), 1)
        self.assertTrue(response.tool_executions[0].result["ok"])
        self.assertEqual(client.chat_calls[0]["model"], "tool-model")
        self.assertEqual(
            client.chat_calls[0]["options"],
            {"temperature": 0.2, "num_ctx": 8192, "num_predict": 500},
        )
        self.assertEqual(
            client.chat_calls[0]["tools"][0]["function"]["name"],
            "search_dreams",
        )

        follow_up_messages = client.chat_calls[1]["messages"]
        self.assertEqual(
            [message["role"] for message in follow_up_messages],
            ["system", "user", "assistant", "tool"],
        )
        tool_result = json.loads(follow_up_messages[-1]["content"])
        self.assertEqual(tool_result["dreams"][0]["dream_id"], "dream-1")
        self.assertEqual(follow_up_messages[-1]["tool_name"], "search_dreams")
        self.assertIn("untrusted data", follow_up_messages[0]["content"])

    def test_agent_passes_date_bounds_selected_by_the_model(self) -> None:
        agent, client, index = self.make_agent(
            [
                tool_response(
                    "search_dreams",
                    {
                        "query": "school classrooms teachers",
                        "start_date": "2026-07-01",
                        "end_date": "2026-07-31",
                    },
                ),
                final_response(),
            ]
        )

        agent.answer("What themes recur in school dreams from last month?")

        self.assertEqual(
            index.calls,
            [
                (
                    "school classrooms teachers",
                    4,
                    date(2026, 7, 1),
                    date(2026, 7, 31),
                )
            ],
        )
        tool_result = json.loads(client.chat_calls[1]["messages"][-1]["content"])
        self.assertEqual(tool_result["start_date"], "2026-07-01")
        self.assertEqual(tool_result["end_date"], "2026-07-31")

    def test_invalid_arguments_are_returned_to_the_model_without_searching(self) -> None:
        agent, client, index = self.make_agent(
            [tool_response("search_dreams", {}), final_response("Insufficient data.")]
        )

        response = agent.answer("What happened?")

        self.assertEqual(index.calls, [])
        self.assertFalse(response.tool_executions[0].result["ok"])
        error = json.loads(client.chat_calls[1]["messages"][-1]["content"])
        self.assertIn("query", error["error"])

    def test_unknown_tools_are_not_dispatched(self) -> None:
        agent, _, index = self.make_agent(
            [tool_response("delete_dreams", {"query": "all"}), final_response()]
        )

        response = agent.answer("Delete everything")

        self.assertEqual(index.calls, [])
        self.assertEqual(
            response.tool_executions[0].result["error"],
            "Unknown tool: delete_dreams",
        )

    def test_total_tool_calls_are_bounded(self) -> None:
        first_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_dreams",
                            "arguments": {"query": "house"},
                        }
                    },
                    {
                        "function": {
                            "name": "search_dreams",
                            "arguments": {"query": "school"},
                        }
                    },
                ],
            }
        }
        agent, _, index = self.make_agent([first_response])

        with self.assertRaisesRegex(AgentToolLimitError, "limit of 1"):
            agent.answer("Compare houses and schools", max_tool_calls=1)

        self.assertEqual(index.calls, [])

    def test_agent_reminds_model_that_search_is_required(self) -> None:
        agent, client, index = self.make_agent(
            [
                final_response("I can answer directly."),
                tool_response("search_dreams", {"query": "hidden room"}),
                final_response(),
            ]
        )

        response = agent.answer("What hidden rooms recur?")

        self.assertEqual(response.answer, "Grounded answer (dream-1, 1/2/2024).")
        self.assertEqual(index.calls, [("hidden room", 4, None, None)])
        reminder = client.chat_calls[1]["messages"][-1]["content"]
        self.assertIn("Call search_dreams", reminder)

    def test_agent_rejects_two_answers_without_search(self) -> None:
        agent, _, index = self.make_agent(
            [final_response("First"), final_response("Second")]
        )

        with self.assertRaisesRegex(AgentSearchRequiredError, "answered twice"):
            agent.answer("What happened?")

        self.assertEqual(index.calls, [])

    def test_system_prompt_defines_relative_date_behavior(self) -> None:
        prompt = DreamRagAgent._system_prompt()

        self.assertIn(date.today().isoformat(), prompt)
        self.assertIn("previous calendar month", prompt)
        self.assertIn("start_date and end_date", prompt)


if __name__ == "__main__":
    unittest.main()
