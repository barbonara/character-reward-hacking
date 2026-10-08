"""Roundtrip tests for tool-call parsing and prompt reconstruction."""

import pytest
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.model._call_tools import get_tools_info
from inspect_ai.tool import ToolCallError, ToolFunction, bash, python, web_browser
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.tinker_local.tinker_sampling import (

    _tinker_content_to_inspect,
    _tinker_tool_calls_to_inspect,
    convert_inspect_messages,
    convert_tools_to_toolspec,
    get_renderer,
    get_renderer_name_for_model,
    select_tools_for_choice,
)


pytestmark = pytest.mark.network  # every test here downloads a HF tokenizer


def _agentic_tools_info():
    return get_tools_info([bash(180), python(180)] + web_browser())


def test_convert_inspect_messages_includes_tool_errors() -> None:
    convo = convert_inspect_messages(
        [
            ChatMessageUser(content="answer the question"),
            ChatMessageAssistant(content="I'll submit now."),
            ChatMessageTool(
                content="",
                tool_call_id="functions.submit:0",
                function="submit",
                error=ToolCallError(type="parsing", message="2 is not of type 'string'"),
            ),
        ]
    )
    tool_message = convo[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["name"] == "submit"
    assert tool_message["tool_call_id"] == "functions.submit:0"
    assert isinstance(tool_message["content"], str)
    assert "Tool error" in tool_message["content"]
    assert "parsing" in tool_message["content"]
    assert "2 is not of type 'string'" in tool_message["content"]


def test_convert_inspect_messages_preserves_tool_output_with_error() -> None:
    convo = convert_inspect_messages(
        [
            ChatMessageUser(content="answer the question"),
            ChatMessageAssistant(content="I'll run python."),
            ChatMessageTool(
                content="Traceback (most recent call last): ...",
                tool_call_id="functions.python:0",
                function="python",
                error=ToolCallError(type="output_limit", message="Output exceeded limit"),
            ),
        ]
    )
    tool_message = convo[-1]
    assert isinstance(tool_message["content"], str)
    assert "Traceback (most recent call last): ..." in tool_message["content"]
    assert "Tool error (output_limit): Output exceeded limit" in tool_message["content"]


class TestQwenToolCallingRoundtrip:
    """Validate parse -> inspect conversion -> prompt render for Qwen tool calls."""

    MODEL_NAME = "Qwen/Qwen3-8B"

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer(self.MODEL_NAME)
        return get_renderer("qwen3", tokenizer)

    def _parse_assistant(self, raw_text: str, renderer) -> ChatMessageAssistant:
        response_tokens = renderer.tokenizer.encode(
            raw_text + "<|im_end|>",
            add_special_tokens=False,
        )
        parsed_message, success = renderer.parse_response(response_tokens)
        assert success
        return ChatMessageAssistant(
            content=_tinker_content_to_inspect(parsed_message["content"]),
            tool_calls=_tinker_tool_calls_to_inspect(parsed_message.get("tool_calls")),
        )

    def _render_prompt(self, messages: list, renderer) -> str:
        convo = convert_inspect_messages(messages)
        prompt = renderer.build_generation_prompt(convo)
        return renderer.tokenizer.decode(prompt.to_ints())

    def test_parse_and_roundtrip_single_tool_call(self, renderer):
        raw = (
            "<think>Need weather lookup</think>Let me check.\n"
            "<tool_call>\n"
            '{"id": "call_weather", "name": "get_weather", "arguments": {"city": "London", "units": "c"}}\n'
            "</tool_call>"
        )
        assistant = self._parse_assistant(raw, renderer)

        assert assistant.tool_calls is not None
        assert len(assistant.tool_calls) == 1
        tool_call = assistant.tool_calls[0]
        assert tool_call.id == "call_weather"
        assert tool_call.function == "get_weather"
        assert tool_call.arguments == {"city": "London", "units": "c"}

        prompt = self._render_prompt(
            [
                ChatMessageUser(content="What's the weather?"),
                assistant,
                ChatMessageTool(
                    content='{"temp": 14, "condition": "rain"}',
                    tool_call_id="call_weather",
                    function="get_weather",
                ),
            ],
            renderer,
        )

        assert "<tool_call>" in prompt
        assert '"name": "get_weather"' in prompt
        assert '"arguments": {"city": "London", "units": "c"}' in prompt
        assert "<tool_response>" in prompt
        assert '{"temp": 14, "condition": "rain"}' in prompt

    def test_parse_and_roundtrip_multiple_tool_calls_preserves_order(self, renderer):
        raw = (
            "<think>Need two checks</think>Running both.\n"
            "<tool_call>\n"
            '{"id": "id_1", "name": "check_a", "arguments": {"value": 1}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"id": "id_2", "name": "check_b", "arguments": {"value": 2}}\n'
            "</tool_call>"
        )
        assistant = self._parse_assistant(raw, renderer)

        assert assistant.tool_calls is not None
        assert [tc.id for tc in assistant.tool_calls] == ["id_1", "id_2"]
        assert [tc.function for tc in assistant.tool_calls] == ["check_a", "check_b"]

        prompt = self._render_prompt(
            [
                ChatMessageUser(content="run checks"),
                assistant,
                ChatMessageTool(content="ok-a", tool_call_id="id_1", function="check_a"),
                ChatMessageTool(content="ok-b", tool_call_id="id_2", function="check_b"),
            ],
            renderer,
        )

        assert prompt.count("<tool_call>") == 2
        assert prompt.count("<tool_response>") == 2
        assert prompt.index('"name": "check_a"') < prompt.index('"name": "check_b"')
        assert prompt.index("ok-a") < prompt.index("ok-b")

    def test_invalid_tool_call_payload_becomes_no_inspect_tool_call(self, renderer):
        raw = (
            "Trying a malformed call.\n"
            "<tool_call>\n"
            '{"name": "broken", "arguments": "not-a-dict"}\n'
            "</tool_call>"
        )
        assistant = self._parse_assistant(raw, renderer)
        assert assistant.tool_calls is None


class TestNemotronSuperToolCompatibility:
    """Validate tool schema handling with the Nemotron-3 Super renderer used by the RL runs.
    (Tool-call parse / roundtrip is covered by TestQwenToolCallingRoundtrip.)"""

    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer(self.MODEL_NAME)
        return get_renderer("nemotron3", tokenizer)

    def test_renderer_name_for_nemotron_super(self) -> None:
        assert get_renderer_name_for_model(self.MODEL_NAME) == "nemotron3"

    def test_toolspec_cleanup_and_tool_declaration(self, renderer) -> None:
        tool_specs = convert_tools_to_toolspec(_agentic_tools_info())

        bash_spec = next(spec for spec in tool_specs if spec["name"] == "bash")
        cmd_schema = bash_spec["parameters"]["properties"]["cmd"]
        assert "anyOf" not in cmd_schema
        assert "default" not in cmd_schema
        assert "enum" not in cmd_schema

        prefix_messages = renderer.create_conversation_prefix_with_tools(tool_specs)
        assert prefix_messages[0]["role"] == "system"
        assert prefix_messages[0]["content"].startswith("# Tools")
        assert "<name>bash</name>" in prefix_messages[0]["content"]

    def test_select_tools_for_choice(self) -> None:
        tools = _agentic_tools_info()
        assert select_tools_for_choice(tools, "none") == []
        selected = select_tools_for_choice(tools, ToolFunction(name="python"))
        assert [tool.name for tool in selected] == ["python"]
        with pytest.raises(ValueError):
            select_tools_for_choice(tools, ToolFunction(name="does_not_exist"))


class TestGptOssRenderer:
    """Validate GPT-OSS Harmony rendering preserves thinking and parses correctly."""

    MODEL_NAME = "openai/gpt-oss-120b"

    @pytest.fixture
    def renderer(self):
        tokenizer = get_tokenizer(self.MODEL_NAME)
        return get_renderer("gpt_oss_no_sysprompt", tokenizer)

    def test_renderer_name_for_gpt_oss(self):
        # Upstream's recommended renderer is gpt_oss_no_sysprompt; we keep that
        # so we don't append a conflicting "You are ChatGPT..." system prompt.
        assert get_renderer_name_for_model(self.MODEL_NAME) == "gpt_oss_no_sysprompt"

    def test_preserves_historical_thinking(self, renderer):
        messages = [
            ChatMessageUser(content="What is 2+2?"),
            ChatMessageAssistant(content=[
                ContentReasoning(reasoning="Adding."),
                ContentText(text="4"),
            ]),
            ChatMessageUser(content="And 3+3?"),
        ]
        convo = convert_inspect_messages(messages)
        rendered = renderer.tokenizer.decode(renderer.build_generation_prompt(convo).to_ints())
        # Historical assistant thinking is rendered into the analysis channel.
        assert "<|channel|>analysis<|message|>Adding.<|end|>" in rendered
        assert "<|channel|>final<|message|>4<|end|>" in rendered

    def test_parse_response_extracts_thinking_and_response(self, renderer):
        raw = (
            "<|channel|>analysis<|message|>Let me think.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer<|return|>"
        )
        tokens = renderer.tokenizer.encode(raw, add_special_tokens=False)
        message, success = renderer.parse_response(tokens)
        assert success
        assert message["content"] == [
            {"type": "thinking", "thinking": "Let me think."},
            {"type": "text", "text": "Answer"},
        ]

    def test_parse_action_synthesizes_stop_token_for_truncated_response(self, renderer):
        from src.utils.parsing import parse_action_to_reasoning_and_response

        # Truncated mid-final response (max_tokens hit, no <|return|> emitted).
        raw = (
            "<|channel|>analysis<|message|>thinking done<|end|>"
            "<|start|>assistant<|channel|>final<|message|>partial response"
        )
        tokens = renderer.tokenizer.encode(raw, add_special_tokens=False)
        reasoning, visible = parse_action_to_reasoning_and_response(tokens, renderer)
        assert reasoning == "thinking done"
        assert visible == "partial response"
