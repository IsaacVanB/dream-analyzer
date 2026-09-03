from __future__ import annotations

import json
import unittest
from datetime import date

from dream_analysis.agent import (
    AgentEmptyResponseError,
    AgentSearchRequiredError,
    DreamRagAgent,
    ToolExecution,
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
                final_response("Discarded draft answer."),
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
        self.assertEqual(len(response.assistant_messages), 3)
        self.assertEqual(
            response.assistant_messages[0]["tool_calls"][0]["function"]["name"],
            "search_dreams",
        )
        self.assertTrue(response.tool_executions[0].result["ok"])
        self.assertIsNotNone(response.tool_executions[0].report_result)
        report_result = response.tool_executions[0].report_result
        assert report_result is not None
        self.assertEqual(
            report_result["dreams"][0]["text"],
            "A hidden room appeared behind the pantry.",
        )
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
        final_messages = client.chat_calls[2]["messages"]
        self.assertEqual(
            [message["role"] for message in final_messages],
            ["system", "user"],
        )
        self.assertIn("RRF_SCORE", final_messages[-1]["content"])

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
                final_response("Discarded draft answer."),
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
            [
                tool_response("search_dreams", {}),
                final_response("Discarded draft answer."),
                final_response("Insufficient data."),
            ]
        )

        response = agent.answer("What happened?")

        self.assertEqual(index.calls, [])
        self.assertFalse(response.tool_executions[0].result["ok"])
        error = json.loads(client.chat_calls[1]["messages"][-1]["content"])
        self.assertIn("query", error["error"])

    def test_unknown_tools_are_not_dispatched(self) -> None:
        agent, _, index = self.make_agent(
            [
                tool_response("delete_dreams", {"query": "all"}),
                final_response("Discarded draft answer."),
                final_response(),
            ]
        )

        response = agent.answer("Delete everything")

        self.assertEqual(index.calls, [])
        self.assertEqual(
            response.tool_executions[0].result["error"],
            "Unknown tool: delete_dreams",
        )

    def test_tool_budget_forces_answer_and_retains_overflow_calls(self) -> None:
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
        agent, client, index = self.make_agent(
            [first_response, final_response("Forced grounded answer.")]
        )

        response = agent.answer("Compare houses and schools", max_tool_calls=1)

        self.assertEqual(response.answer, "Forced grounded answer.")
        self.assertTrue(response.forced_synthesis)
        self.assertIn("budget", response.forced_synthesis_reason)
        self.assertEqual(index.calls, [("house", 4, None, None)])
        self.assertEqual(len(response.tool_executions), 1)
        self.assertEqual(
            [call.arguments["query"] for call in response.unexecuted_tool_calls],
            ["school"],
        )
        self.assertIsNone(client.chat_calls[-1]["tools"])
        forced_messages = client.chat_calls[-1]["messages"]
        self.assertEqual([item["role"] for item in forced_messages], ["system", "user"])
        self.assertIn(
            "COMPLETED SEARCH EVIDENCE",
            forced_messages[-1]["content"],
        )
        self.assertIn("DREAM_ID: dream-1", forced_messages[-1]["content"])
        self.assertIn(
            "A hidden room appeared behind the pantry.",
            forced_messages[-1]["content"],
        )
        self.assertTrue(response.turn_traces[-1].forced_synthesis)
        self.assertIn(
            "COMPLETED SEARCH EVIDENCE",
            response.turn_traces[-1].request_prompt,
        )

    def test_budget_executes_remaining_capacity_before_forced_answer(self) -> None:
        second_response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_dreams",
                            "arguments": {"query": "school"},
                        }
                    },
                    {
                        "function": {
                            "name": "search_dreams",
                            "arguments": {"query": "exam"},
                        }
                    },
                ],
            }
        }
        agent, _, index = self.make_agent(
            [
                tool_response("search_dreams", {"query": "house"}),
                second_response,
                final_response("Forced comparison."),
            ]
        )

        response = agent.answer("Compare houses and schools", max_tool_calls=2)

        self.assertEqual(
            index.calls,
            [("house", 4, None, None), ("school", 4, None, None)],
        )
        self.assertEqual(
            [item.arguments["query"] for item in response.tool_executions],
            ["house", "school"],
        )
        self.assertEqual(
            [item.arguments["query"] for item in response.unexecuted_tool_calls],
            ["exam"],
        )

    def test_duplicate_search_reuses_cached_result_then_forces_answer(self) -> None:
        agent, _, index = self.make_agent(
            [
                tool_response("search_dreams", {"query": "hidden room"}),
                tool_response("search_dreams", {"query": "hidden room"}),
                final_response("The retrieved dream contains a hidden room."),
            ]
        )

        response = agent.answer("What hidden rooms recur?", max_tool_calls=2)

        self.assertEqual(index.calls, [("hidden room", 4, None, None)])
        self.assertFalse(response.tool_executions[0].cached)
        self.assertTrue(response.tool_executions[1].cached)
        self.assertTrue(response.tool_executions[1].result["cached"])
        self.assertTrue(response.forced_synthesis)
        self.assertIn(
            "SEARCH 2 [cached duplicate]",
            response.turn_traces[-1].request_prompt,
        )

    def test_empty_answer_gets_one_forced_synthesis_retry(self) -> None:
        agent, client, index = self.make_agent(
            [
                tool_response("search_dreams", {"query": "hidden room"}),
                final_response(""),
                final_response("A forced answer with evidence."),
            ]
        )

        response = agent.answer("What hidden rooms recur?", max_tool_calls=3)

        self.assertEqual(index.calls, [("hidden room", 4, None, None)])
        self.assertEqual(response.answer, "A forced answer with evidence.")
        self.assertTrue(response.forced_synthesis)
        self.assertIn("finished requesting searches", response.forced_synthesis_reason)
        self.assertIsNone(client.chat_calls[-1]["tools"])

    def test_empty_forced_answer_raises_with_response_diagnostics(self) -> None:
        empty_forced_response = {
            "done_reason": "length",
            "eval_count": 700,
            "message": {"role": "assistant", "content": ""},
        }
        agent, _, index = self.make_agent(
            [
                tool_response("search_dreams", {"query": "hidden room"}),
                final_response(""),
                empty_forced_response,
            ]
        )

        with self.assertRaisesRegex(
            AgentEmptyResponseError,
            "done_reason='length'",
        ) as raised:
            agent.answer("What hidden rooms recur?", max_tool_calls=3)

        self.assertEqual(index.calls, [("hidden room", 4, None, None)])
        self.assertEqual(len(raised.exception.turn_traces), 3)
        self.assertEqual(
            raised.exception.turn_traces[-1].diagnostics["eval_count"],
            700,
        )
        self.assertIn(
            "DREAM_ID: dream-1",
            raised.exception.turn_traces[-1].request_prompt,
        )

    def test_forced_synthesis_evidence_is_deduplicated_and_bounded(self) -> None:
        execution = ToolExecution(
            name="search_dreams",
            arguments={"query": "hidden room"},
            result={
                "ok": True,
                "dreams": [
                    {
                        "dream_id": "dream-1",
                        "date": "1/2/2024",
                        "distance": 0.1,
                        "text": "room " * 1000,
                    }
                ],
            },
        )
        duplicate = ToolExecution(
            name=execution.name,
            arguments=execution.arguments,
            result=execution.result,
            cached=True,
        )

        evidence = DreamRagAgent._format_synthesis_evidence(
            [execution, duplicate],
            max_chars=1200,
        )

        self.assertLessEqual(len(evidence), 1200)
        self.assertEqual(evidence.count("DREAM_ID: dream-1"), 1)
        self.assertIn("SEARCH 2 [cached duplicate]", evidence)
        self.assertIn("Distinct searches used for ranking: 1", evidence)
        self.assertIn("TRUNCATED", evidence)

    def test_synthesis_uses_rrf_and_limits_unique_dreams(self) -> None:
        def execution(query: str, dream_ids: list[str]) -> ToolExecution:
            dreams = [
                {
                    "dream_id": dream_id,
                    "date": "1/2/2024",
                    "distance": rank / 10,
                    "text": f"Full text for {dream_id}.",
                }
                for rank, dream_id in enumerate(dream_ids, start=1)
            ]
            result = {"ok": True, "dreams": dreams}
            return ToolExecution(
                name="search_dreams",
                arguments={"query": query},
                result=result,
                report_result=result,
            )

        evidence = DreamRagAgent._format_synthesis_evidence(
            [
                execution("rooms", ["dream-a", "dream-shared"]),
                execution("doors", ["dream-b", "dream-shared"]),
            ],
            max_chars=4000,
            max_dreams=1,
        )

        self.assertIn("DREAM_ID: dream-shared", evidence)
        self.assertNotIn("DREAM_ID: dream-a", evidence)
        self.assertNotIn("DREAM_ID: dream-b", evidence)
        self.assertIn("Dreams omitted by synthesis limit: 2", evidence)

    def test_synthesis_preserves_105_words_or_drops_lower_ranked_dreams(self) -> None:
        dreams = [
            {
                "dream_id": f"dream-{index}",
                "date": "1/2/2024",
                "distance": index / 10,
                "text": " ".join(f"dream{index}word{word}" for word in range(200)),
            }
            for index in range(1, 3)
        ]
        result = {"ok": True, "dreams": dreams}
        execution = ToolExecution(
            name="search_dreams",
            arguments={"query": "rooms"},
            result=result,
            report_result=result,
        )

        evidence = DreamRagAgent._format_synthesis_evidence(
            [execution],
            max_chars=2200,
            max_dreams=2,
        )

        self.assertIn("dream1word104", evidence)
        self.assertNotIn("DREAM_ID: dream-2", evidence)
        self.assertIn("Dreams omitted for context limit: 1", evidence)
        self.assertLessEqual(len(evidence), 2200)

    def test_agent_reminds_model_that_search_is_required(self) -> None:
        agent, client, index = self.make_agent(
            [
                final_response("I can answer directly."),
                tool_response("search_dreams", {"query": "hidden room"}),
                final_response("Discarded draft answer."),
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
        self.assertIn("SEARCH_COMPLETE", prompt)


if __name__ == "__main__":
    unittest.main()
