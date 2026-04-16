"""Test suite for vLLM deployment: basic inference and tool calling.

Uses the OpenAI SDK for direct API tests and the OpenAI Agents SDK
for agentic tool-calling evaluation. Model-agnostic -- discovers
whatever model is currently served.
"""

import asyncio
import json
import unicodedata
import pytest
from openai import OpenAI, AsyncOpenAI
from agents import (
    Agent,
    Runner,
    ModelSettings,
    OpenAIChatCompletionsModel,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)


BASE_URL = "http://localhost:8000/v1"
API_KEY = "none"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sync_client():
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


@pytest.fixture(scope="session")
def async_client():
    return AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)


@pytest.fixture(scope="class")
def agents_async_client():
    """Separate client for Agents SDK tests to avoid connection pool issues."""
    return AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)


@pytest.fixture(scope="session")
def model_id(sync_client):
    """Discover the first model the server is serving."""
    models = sync_client.models.list()
    assert models.data, "No models available on the server"
    return models.data[0].id


@pytest.fixture(scope="session", autouse=True)
def _configure_agents_sdk(async_client):
    """Point the Agents SDK at the local vLLM server."""
    set_default_openai_client(async_client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)


# ---------------------------------------------------------------------------
# Helpers -- Agents SDK tools
# ---------------------------------------------------------------------------

@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "new york": "Partly cloudy, 18C",
        "london": "Rainy, 12C",
        "tokyo": "Clear, 24C",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the result."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    try:
        result = eval(expression)  # noqa: S307 -- limited charset
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# ===================================================================
# 1. OpenAI SDK -- Direct Inference
# ===================================================================

class TestDirectInference:
    """Verify basic chat completions via the OpenAI SDK."""

    def test_model_available(self, model_id):
        assert model_id is not None

    def test_simple_completion(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Reply with only: hello"}],
            max_tokens=16,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        assert text is not None and len(text) > 0

    def test_system_prompt_respected(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Always respond in exactly three words."},
                {"role": "user", "content": "How are you?"},
            ],
            max_tokens=32,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        word_count = len(text.split())
        assert 1 <= word_count <= 8, f"Expected ~3 words, got {word_count}: {text!r}"

    def test_token_usage_reported(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_max_tokens_limit(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Write a long essay about space."}],
            max_tokens=10,
            temperature=0.0,
        )
        assert resp.choices[0].finish_reason in ("length", "stop")
        assert resp.usage.completion_tokens <= 15

    def test_temperature_zero_determinism(self, sync_client, model_id):
        kwargs = dict(
            model=model_id,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=16,
            temperature=0.0,
        )
        r1 = sync_client.chat.completions.create(**kwargs)
        r2 = sync_client.chat.completions.create(**kwargs)
        assert r1.choices[0].message.content == r2.choices[0].message.content

    def test_concurrent_requests(self, async_client, model_id):
        """Fire many requests in parallel and verify all complete correctly."""
        prompts = [
            "What is the capital of France? Answer in one word.",
            "What is the capital of Japan? Answer in one word.",
            "What is the capital of Brazil? Answer in one word.",
            "What is the capital of Egypt? Answer in one word.",
            "What is the capital of Australia? Answer in one word.",
            "What is the capital of Canada? Answer in one word.",
            "What is the capital of Italy? Answer in one word.",
            "What is the capital of Germany? Answer in one word.",
        ]
        expected = [
            "paris", "tokyo", "brasilia", "cairo",
            "canberra", "ottawa", "rome", "berlin",
        ]

        async def _run():
            tasks = [
                async_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": p}],
                    max_tokens=64,
                    temperature=0.0,
                )
                for p in prompts
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        assert len(results) == len(prompts)
        for resp, answer in zip(results, expected):
            text = resp.choices[0].message.content
            assert text is not None and len(text) > 0
            assert resp.usage.completion_tokens > 0
            normalized = unicodedata.normalize("NFD", text.lower())
            stripped = "".join(
                c for c in normalized if unicodedata.category(c) != "Mn"
            )
            assert answer in stripped, (
                f"Expected '{answer}' in response: {text!r}"
            )


# ===================================================================
# 2. OpenAI SDK -- Tool Calling
# ===================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


class TestToolCalling:
    """Verify that the model produces well-formed tool calls."""

    def test_single_tool_call(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is not None and len(msg.tool_calls) > 0
        tc = msg.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "city" in args

    def test_correct_tool_selection(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "What is 123 * 456?"},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is not None and len(msg.tool_calls) > 0
        tc = msg.tool_calls[0]
        assert tc.function.name == "calculate"

    def test_tool_call_finish_reason(self, sync_client, model_id):
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "Check the weather in London please."},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
        assert resp.choices[0].finish_reason in ("tool_calls", "stop")

    def test_tool_roundtrip(self, sync_client, model_id):
        """Full tool-use loop: request -> tool call -> tool result -> final answer."""
        messages = [
            {"role": "user", "content": "What is the weather in New York?"},
        ]
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is not None

        tc = msg.tool_calls[0]
        weather_result = "Partly cloudy, 18C"

        messages.append(msg.model_dump())
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": weather_result,
        })

        resp2 = sync_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=TOOLS,
            max_tokens=256,
            temperature=0.0,
        )
        final = resp2.choices[0].message.content
        assert final is not None and len(final) > 0


# ===================================================================
# 3. OpenAI Agents SDK -- Agentic Tool Use
# ===================================================================

class TestAgentsSDK:
    """Evaluate the Agents SDK running against the local vLLM server."""

    def _make_agent(self, model_id, agents_async_client, **kwargs):
        defaults = dict(
            name="test-agent",
            instructions="You are a helpful assistant. Use tools when needed.",
            model=OpenAIChatCompletionsModel(
                model=model_id,
                openai_client=agents_async_client,
            ),
            model_settings=ModelSettings(temperature=0.0, max_tokens=512),
        )
        defaults.update(kwargs)
        return Agent(**defaults)

    def test_basic_agent_response(self, model_id, agents_async_client):
        agent = self._make_agent(
            model_id,
            agents_async_client,
            instructions="Reply with only the word 'pong'.",
        )
        result = Runner.run_sync(agent, "ping")
        assert result.final_output is not None
        assert len(result.final_output) > 0

    def test_agent_tool_invocation(self, model_id, agents_async_client):
        agent = self._make_agent(
            model_id, agents_async_client, tools=[get_weather],
        )
        result = Runner.run_sync(agent, "What is the weather in Tokyo?")
        output = result.final_output
        assert output is not None
        assert any(
            term in output.lower()
            for term in ("24", "clear", "tokyo")
        ), f"Expected weather data in response: {output!r}"

    def test_agent_calculator(self, model_id, agents_async_client):
        agent = self._make_agent(
            model_id, agents_async_client, tools=[calculate],
        )
        result = Runner.run_sync(agent, "What is 15 * 37?")
        output = result.final_output
        assert output is not None
        assert "555" in output, f"Expected 555 in response: {output!r}"

    def test_agent_multi_tool(self, model_id, agents_async_client):
        agent = self._make_agent(
            model_id, agents_async_client, tools=[get_weather, calculate],
        )
        result = Runner.run_sync(
            agent,
            "What is the weather in London?",
        )
        output = result.final_output
        assert output is not None
        assert any(
            term in output.lower()
            for term in ("12", "rainy", "london")
        ), f"Expected London weather data in response: {output!r}"

    def test_agent_no_tool_when_unnecessary(self, model_id, agents_async_client):
        agent = self._make_agent(
            model_id, agents_async_client, tools=[get_weather, calculate],
        )
        result = Runner.run_sync(agent, "Say hello.")
        assert result.final_output is not None
        assert len(result.final_output) > 0


# ===================================================================
# 4. Deep Reasoning
# ===================================================================

FLUX_INTEGRAL_PROMPT = (
    "Let F(x, y, z) = <x^2 * z, x * y^2, z^3>, and let S be the closed "
    "surface bounding the solid region enclosed by the paraboloid "
    "z = 4 - x^2 - y^2 and the plane z = 0, with outward-pointing normal. "
    "Evaluate the flux integral: double integral over S of F dot dS. "
    "Show your full work and give the final numerical answer."
)


class TestDeepReasoning:
    """Verify the model can solve a multi-step calculus problem."""

    def test_flux_integral(self, sync_client, model_id):
        """Divergence theorem problem whose answer is 64*pi."""
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert mathematician. Show all work "
                        "step by step. State the final answer clearly."
                    ),
                },
                {"role": "user", "content": FLUX_INTEGRAL_PROMPT},
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        assert text is not None and len(text) > 100, "Response too short"

        text_normalized = text.lower().replace(" ", "")
        accepted = ("64pi", "64*pi", "64\\pi", "64 pi", "64*\\pi")
        assert any(
            a.replace(" ", "") in text_normalized for a in accepted
        ), f"Expected 64*pi in answer, got:\n{text[-300:]}"


# ===================================================================
# 5. Reasoning: Multi-Turn, Streaming, and Trace Handling
# ===================================================================

THINKING_BODY = {
    "chat_template_kwargs": {"enable_thinking": True},
    "skip_special_tokens": False,
}


class TestReasoning:
    """Verify thinking mode, streaming, and multi-turn trace handling."""

    def test_reasoning_separation(self, sync_client, model_id):
        """Reasoning and content are returned in separate fields."""
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is 12 * 13?"}],
            max_tokens=1024,
            temperature=0.0,
            extra_body=THINKING_BODY,
        )
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning", None)
        content = msg.content
        assert reasoning is not None and len(reasoning) > 0, (
            "reasoning field should be populated"
        )
        assert content is not None and len(content) > 0, (
            "content field should be populated"
        )
        assert "156" in content, f"Expected 156 in content: {content!r}"

    def test_multi_turn_with_reasoning(self, sync_client, model_id):
        """Multi-turn conversation: reasoning is generated each turn
        and prior reasoning is NOT leaked into the prompt."""
        # Turn 1
        r1 = sync_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is 7 * 8?"}],
            max_tokens=1024,
            temperature=0.0,
            extra_body=THINKING_BODY,
        )
        t1_reasoning = getattr(r1.choices[0].message, "reasoning", None)
        t1_content = r1.choices[0].message.content
        t1_tokens = r1.usage.prompt_tokens
        assert t1_content is not None
        assert "56" in t1_content

        # Turn 2 -- send back only content (correct behavior)
        r2 = sync_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "What is 7 * 8?"},
                {"role": "assistant", "content": t1_content},
                {"role": "user", "content": "Now what is 9 * 6?"},
            ],
            max_tokens=1024,
            temperature=0.0,
            extra_body=THINKING_BODY,
        )
        t2_content = r2.choices[0].message.content
        t2_tokens = r2.usage.prompt_tokens
        assert t2_content is not None
        assert "54" in t2_content

        # Turn 2 again -- send back content AND reasoning (bad practice)
        r2_with_reasoning = sync_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "What is 7 * 8?"},
                {
                    "role": "assistant",
                    "content": t1_content,
                    "reasoning": t1_reasoning or "dummy reasoning",
                },
                {"role": "user", "content": "Now what is 9 * 6?"},
            ],
            max_tokens=1024,
            temperature=0.0,
            extra_body=THINKING_BODY,
        )
        t2_wr_tokens = r2_with_reasoning.usage.prompt_tokens

        # vLLM should ignore the reasoning field -- token counts must match
        assert t2_tokens == t2_wr_tokens, (
            f"Reasoning leaked into prompt: {t2_tokens} vs {t2_wr_tokens} tokens"
        )

    def test_streaming_with_reasoning(self, async_client, model_id):
        """Streaming returns reasoning and content in separate deltas."""
        async def _run():
            reasoning_parts = []
            content_parts = []
            stream = await async_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "What is 11 * 12?"}],
                max_tokens=1024,
                temperature=0.0,
                stream=True,
                extra_body=THINKING_BODY,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                r = getattr(delta, "reasoning", None)
                if r:
                    reasoning_parts.append(r)
                c = getattr(delta, "content", None)
                if c:
                    content_parts.append(c)
            return "".join(reasoning_parts), "".join(content_parts)

        reasoning, content = asyncio.run(_run())
        assert len(reasoning) > 0, "No reasoning streamed"
        assert len(content) > 0, "No content streamed"
        assert "132" in content, f"Expected 132 in content: {content!r}"

    def test_streaming_multi_turn(self, async_client, model_id):
        """Multi-turn streaming: each turn produces reasoning + content."""
        async def _stream_turn(messages):
            reasoning_parts = []
            content_parts = []
            stream = await async_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
                stream=True,
                extra_body=THINKING_BODY,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                r = getattr(delta, "reasoning", None)
                if r:
                    reasoning_parts.append(r)
                c = getattr(delta, "content", None)
                if c:
                    content_parts.append(c)
            return "".join(reasoning_parts), "".join(content_parts)

        async def _run():
            # Turn 1
            r1, c1 = await _stream_turn([
                {"role": "user", "content": "What is the capital of France?"},
            ])
            assert len(c1) > 0
            assert "paris" in c1.lower()

            # Turn 2 -- only content from turn 1, no reasoning
            r2, c2 = await _stream_turn([
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": c1},
                {"role": "user", "content": "And what about Germany?"},
            ])
            assert len(c2) > 0
            assert "berlin" in c2.lower()
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert len(r1) > 0, "Turn 1 should have reasoning"
        assert len(r2) > 0, "Turn 2 should have reasoning"

    def test_agents_sdk_multi_turn_tool_use(self, model_id, agents_async_client):
        """Agents SDK multi-turn: tool call, result, final answer.

        The SDK should NOT replay reasoning into subsequent turns
        (default_should_replay_reasoning_content returns False for
        non-DeepSeek models).
        """
        agent = Agent(
            name="math-agent",
            instructions="Use the calculate tool for arithmetic. Be brief.",
            model=OpenAIChatCompletionsModel(
                model=model_id,
                openai_client=agents_async_client,
            ),
            model_settings=ModelSettings(
                temperature=0.0,
                max_tokens=512,
                extra_body=THINKING_BODY,
            ),
            tools=[calculate],
        )
        result = Runner.run_sync(
            agent, "What is 45 * 67? Then tell me the result plus 100.",
        )
        output = result.final_output
        assert output is not None
        flat = output.replace(",", "")
        assert "3015" in flat or "3115" in flat, (
            f"Expected 3015 or 3115 in output: {output!r}"
        )


# ===================================================================
# 6. Progressive Context Length
# ===================================================================

def _build_conversation(num_turns, words_per_turn=200):
    """Build a multi-turn conversation that grows the context window.

    Each user turn contains a numbered factual statement padded with filler,
    and each assistant turn acknowledges it. The final user message asks the
    model to recall fact #1 from the very beginning.
    """
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
    )
    # Repeat filler to hit approximate word target per turn
    padding = (filler * ((words_per_turn // len(filler.split())) + 1))
    padding_words = " ".join(padding.split()[:words_per_turn])

    messages = [
        {"role": "system", "content": "You are a helpful assistant with perfect memory."},
    ]
    for i in range(1, num_turns + 1):
        fact = f"FACT-{i}: The secret code for round {i} is ALPHA-{i * 111}."
        messages.append({
            "role": "user",
            "content": f"{fact} {padding_words}",
        })
        messages.append({
            "role": "assistant",
            "content": f"Understood, I have noted FACT-{i}.",
        })

    messages.append({
        "role": "user",
        "content": (
            "Recall FACT-1 exactly. What was the secret code for round 1? "
            "Reply with only the code, nothing else."
        ),
    })
    return messages


class TestContextLength:
    """Progressive context stress tests at increasing conversation depths."""

    @pytest.mark.parametrize(
        "num_turns,label",
        [
            (10, "~3K tokens"),
            (50, "~15K tokens"),
            (100, "~30K tokens"),
            (200, "~60K tokens"),
            (350, "~100K tokens"),
            (420, "~125K tokens"),
        ],
    )
    def test_recall_at_depth(self, sync_client, model_id, num_turns, label):
        """Verify the model can recall a fact from turn 1 after many turns."""
        messages = _build_conversation(num_turns)
        resp = sync_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=64,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        assert text is not None
        assert "111" in text, (
            f"Failed recall at {label} ({num_turns} turns): {text!r}"
        )
        prompt_tokens = resp.usage.prompt_tokens
        print(f"  {label}: {prompt_tokens} prompt tokens, recall OK")
