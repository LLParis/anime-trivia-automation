from __future__ import annotations

import asyncio
import base64
import json
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from anime_trivia_automation.config import (
    AppConfig,
    CaptureConfig,
    GeminiConfig,
    validate_config,
)
from anime_trivia_automation.gemini import (
    GeminiProvider,
    GeminiRequest,
)


class _Interaction:
    def __init__(self, payload: dict | str) -> None:
        self.output_text = payload if isinstance(payload, str) else json.dumps(payload)


class _FakeInteractions:
    def __init__(self, responses: list[dict | str | Exception], delay: float = 0.0) -> None:
        self.responses = list(responses)
        self.delay = delay
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Interaction(response)


class _FakeAio:
    def __init__(self, interactions: _FakeInteractions) -> None:
        self.interactions = interactions
        self.models = self
        self.closed = False

    async def get(self, *, model: str):
        response = self.interactions.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(name=f"models/{model}")

    async def aclose(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, interactions: _FakeInteractions) -> None:
        self.aio = _FakeAio(interactions)


def _answer_payload(
    answer: str | None = "Attack on Titan",
    *,
    answer_type: str = "anime_title",
    confidence: float = 0.98,
    confidence_label: str = "high",
    abstain: bool = False,
) -> dict:
    return {
        "answer": answer,
        "answer_type": answer_type,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "abstain": abstain,
        "evidence": "The quote is uniquely associated with this series.",
    }


class GeminiProviderTests(unittest.TestCase):
    def test_pinned_sdk_effective_retry_count_is_zero(self) -> None:
        client = GeminiProvider._default_client_factory("not-a-real-key", 1000)
        self.assertEqual(
            client.aio.interactions.sdk_configuration.retry_config.max_retries,
            0,
        )
        asyncio.run(client.aio.aclose())

    def make_provider(
        self,
        responses: list[dict | str | Exception],
        *,
        config: GeminiConfig | None = None,
        delay: float = 0.0,
        api_key: str = "unit-test-secret",
    ) -> tuple[GeminiProvider, _FakeInteractions, list[tuple[str, int]]]:
        interactions = _FakeInteractions(responses, delay=delay)
        client = _FakeClient(interactions)
        factory_calls: list[tuple[str, int]] = []

        def factory(key: str, timeout_ms: int):
            factory_calls.append((key, timeout_ms))
            return client

        provider = GeminiProvider(
            config or GeminiConfig(),
            environ={"GEMINI_API_KEY": api_key},
            client_factory=factory,
        )
        return provider, interactions, factory_calls

    def test_config_enforces_accuracy_first_models(self) -> None:
        config = AppConfig(
            capture=CaptureConfig(region=(0, 0, 1920, 1080), calibrated=True),
            gemini=GeminiConfig(primary_model="gemini-3.5-flash-lite"),
        )
        with self.assertRaisesRegex(ValueError, "primary_model"):
            validate_config(config)

    def test_missing_key_is_a_cached_fail_closed_preflight(self) -> None:
        factory_calls: list[tuple[str, int]] = []
        provider = GeminiProvider(
            GeminiConfig(),
            environ={},
            client_factory=lambda key, timeout: factory_calls.append((key, timeout)),
        )

        async def scenario():
            first = await provider.preflight()
            second = await provider.preflight()
            return first, second

        with mock.patch("winreg.QueryValueEx", side_effect=FileNotFoundError):
            first, second = asyncio.run(scenario())
        self.assertEqual(first.phase, "missing_key")
        self.assertFalse(first.available)
        self.assertEqual(second, first)
        self.assertEqual(factory_calls, [])

    @unittest.skipUnless(__import__("os").name == "nt", "Windows registry fallback")
    def test_user_environment_registry_supplies_a_key_to_an_old_process(self) -> None:
        provider, _interactions, factory_calls = self.make_provider(
            [{"ready": True}], api_key="unused-environment-key"
        )
        provider._environ = {}

        with mock.patch("winreg.QueryValueEx", return_value=("registry-secret", 1)):
            availability = asyncio.run(provider.preflight())

        self.assertTrue(availability.available)
        self.assertEqual(factory_calls[0][0], "registry-secret")

    def test_primary_preflight_and_image_resolution_are_strict_and_ungrounded(self) -> None:
        provider, interactions, factory_calls = self.make_provider(
            [{"ready": True}, _answer_payload()]
        )

        async def scenario():
            availability = await provider.preflight()
            result = await provider.resolve(
                GeminiRequest(
                    clue='"Dedicate your hearts!"',
                    expected_answer_type="anime_title",
                    image_bytes=b"cropped-png",
                    image_mime_type="image/png",
                )
            )
            return availability, result

        availability, result = asyncio.run(scenario())
        self.assertTrue(availability.available)
        self.assertEqual(availability.model, "gemini-3.7-flash")
        self.assertTrue(result.accepted)
        self.assertEqual(result.answer, "Attack on Titan")
        self.assertEqual(result.model, "gemini-3.7-flash")
        self.assertEqual(factory_calls[0][0], "unit-test-secret")
        self.assertGreaterEqual(factory_calls[0][1], 4000)

        request = interactions.calls[0]
        self.assertEqual(request["model"], "gemini-3.7-flash")
        self.assertEqual(request["generation_config"], {"thinking_level": "low"})
        self.assertNotIn("tools", request)
        self.assertEqual(request["response_format"]["mime_type"], "application/json")
        self.assertEqual(request["input"][0]["type"], "text")
        self.assertIn("ungrounded", request["input"][0]["text"])
        self.assertEqual(request["input"][1]["type"], "image")
        self.assertEqual(
            request["input"][1]["data"],
            base64.b64encode(b"cropped-png").decode("ascii"),
        )

    def test_type_mismatch_fails_closed_but_medium_confidence_answers(self) -> None:
        provider, interactions, _factory = self.make_provider(
            [
                {"ready": True},
                _answer_payload(answer_type="character"),
                {
                    **_answer_payload(confidence=0.62, confidence_label="medium"),
                    "alternatives": ["Attack on Titan", "Kabaneri", "Kabaneri", "@here"],
                },
                _answer_payload(confidence=0.20, confidence_label="low"),
            ]
        )

        async def scenario():
            await provider.preflight()
            wrong_type = await provider.resolve(
                GeminiRequest("Dedicate your hearts", "anime_title")
            )
            medium = await provider.resolve(
                GeminiRequest('"Dedicate your hearts!" 🏍️', "anime_title")
            )
            low = await provider.resolve(
                GeminiRequest("Dedicate your hearts", "anime_title")
            )
            return wrong_type, medium, low

        wrong_type, medium, low = asyncio.run(scenario())
        self.assertEqual(wrong_type.status, "invalid_response")
        self.assertIsNone(wrong_type.answer)
        # Wrong guesses are free: a medium-confidence answer is submitted, and
        # its distinct, sanitized alternatives ride along as follow-up guesses.
        self.assertEqual(medium.status, "answered")
        self.assertEqual(medium.answer, "Attack on Titan")
        self.assertEqual(medium.alternatives, ("Kabaneri",))
        # Below the configured floor the provider still abstains.
        self.assertEqual(low.status, "abstained")
        self.assertIsNone(low.answer)
        prompt_text = interactions.calls[1]["input"]
        self.assertIn("alternatives", prompt_text)
        self.assertIn("motorcycle", prompt_text)

    def test_model_abstention_is_preserved(self) -> None:
        provider, _interactions, _factory = self.make_provider(
            [
                {"ready": True},
                _answer_payload(
                    answer=None,
                    answer_type="character",
                    confidence=0.2,
                    confidence_label="low",
                    abstain=True,
                ),
            ]
        )

        async def scenario():
            await provider.preflight()
            return await provider.resolve(GeminiRequest("ambiguous clue", "character"))

        result = asyncio.run(scenario())
        self.assertEqual(result.status, "abstained")
        self.assertIsNone(result.answer)

    def test_answer_sanitizer_rejects_discord_mentions(self) -> None:
        provider, _interactions, _factory = self.make_provider(
            [{"ready": True}, _answer_payload("@everyone Attack on Titan")]
        )

        async def scenario():
            await provider.preflight()
            return await provider.resolve(
                GeminiRequest("Dedicate your hearts", "anime_title")
            )

        result = asyncio.run(scenario())
        self.assertEqual(result.status, "invalid_response")
        self.assertIsNone(result.answer)

    def test_absolute_deadline_cancels_a_late_cloud_result(self) -> None:
        provider, interactions, _factory = self.make_provider(
            [{"ready": True}, _answer_payload()]
        )

        async def scenario():
            await provider.preflight()
            interactions.delay = 0.10
            started = time.perf_counter()
            result = await provider.resolve(
                GeminiRequest(
                    "Dedicate your hearts",
                    "anime_title",
                    deadline=time.perf_counter() + 0.02,
                )
            )
            return result, time.perf_counter() - started

        result, elapsed = asyncio.run(scenario())
        self.assertEqual(result.status, "timeout")
        self.assertLess(elapsed, 0.08)

    def test_scout_is_disabled_and_never_selected_implicitly(self) -> None:
        provider, interactions, _factory = self.make_provider([{"ready": True}])

        async def scenario():
            await provider.preflight()
            return await provider.resolve(
                GeminiRequest("new clue", "anime_title", lane="scout")
            )

        result = asyncio.run(scenario())
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.model, "gemini-3.5-flash-lite")
        self.assertEqual(len(interactions.calls), 0)

    def test_explicitly_enabled_scout_uses_minimal_thinking(self) -> None:
        provider, interactions, _factory = self.make_provider(
            [{"ready": True}, _answer_payload()],
            config=GeminiConfig(scout_enabled=True),
        )

        async def scenario():
            await provider.preflight()
            return await provider.resolve(
                GeminiRequest("Dedicate your hearts", "anime_title", lane="scout")
            )

        result = asyncio.run(scenario())
        self.assertTrue(result.accepted)
        self.assertEqual(result.model, "gemini-3.5-flash-lite")
        self.assertEqual(
            interactions.calls[0]["generation_config"],
            {"thinking_level": "minimal"},
        )

    def test_exception_text_cannot_leak_the_api_key_to_logs_or_status(self) -> None:
        secret = "never-log-this-secret"
        provider, _interactions, _factory = self.make_provider(
            [RuntimeError(secret)], api_key=secret
        )

        with self.assertLogs("anime_trivia_automation.gemini", level="WARNING") as logs:
            availability = asyncio.run(provider.preflight())

        combined = "\n".join([*logs.output, availability.detail])
        self.assertEqual(availability.phase, "unavailable")
        self.assertNotIn(secret, combined)


if __name__ == "__main__":
    unittest.main()
