"""Regression tests for gateway transcript persistence."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


def _make_runner_with_db() -> Any:
    from gateway.run import GatewayRunner

    runner: Any = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner._session_db = object()
    return runner


def test_gateway_persists_current_platform_user_message_even_when_agent_persisted():
    """The gateway must own the inbound platform user row in state.db.

    Agent-side persistence can skip the live Discord/Telegram message when
    history_offset points past the repaired current turn.  The gateway still
    knows the exact platform message_id, so it must write that user row to DB
    even while skipping DB writes for assistant/tool rows the agent already
    persisted.
    """

    runner = _make_runner_with_db()
    event = SimpleNamespace(message_id="discord-msg-123")
    new_messages = [
        {"role": "user", "content": "Test discord context."},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "content": "{}", "tool_name": "noop"},
    ]

    runner._persist_gateway_turn_messages(
        session_id="sess-discord-dm",
        new_messages=new_messages,
        message_text="Test discord context.",
        event=event,
        response="ok",
        timestamp="2026-06-04T02:58:00",
        agent_persisted=True,
    )

    calls = runner.session_store.append_to_transcript.call_args_list
    assert len(calls) == 3

    user_call = calls[0]
    assert user_call.args[0] == "sess-discord-dm"
    assert user_call.args[1]["role"] == "user"
    assert user_call.args[1]["content"] == "Test discord context."
    assert user_call.args[1]["message_id"] == "discord-msg-123"
    assert user_call.kwargs.get("skip_db") in (None, False)

    assistant_call = calls[1]
    tool_call = calls[2]
    assert assistant_call.kwargs["skip_db"] is True
    assert tool_call.kwargs["skip_db"] is True


def test_gateway_synthesizes_platform_user_message_when_agent_result_omits_it():
    """Non-empty agent deltas can still omit the live platform user row."""

    runner = _make_runner_with_db()
    event = SimpleNamespace(message_id="discord-msg-456")

    runner._persist_gateway_turn_messages(
        session_id="sess-discord-dm",
        new_messages=[{"role": "assistant", "content": "reply only"}],
        message_text="fresh discord dm",
        event=event,
        response="reply only",
        timestamp="2026-06-04T03:48:00",
        agent_persisted=True,
    )

    calls = runner.session_store.append_to_transcript.call_args_list
    assert len(calls) == 2

    user_call = calls[0]
    assert user_call.args[1]["role"] == "user"
    assert user_call.args[1]["content"] == "fresh discord dm"
    assert user_call.args[1]["message_id"] == "discord-msg-456"
    assert user_call.kwargs.get("skip_db") in (None, False)

    assistant_call = calls[1]
    assert assistant_call.args[1]["role"] == "assistant"
    assert assistant_call.kwargs["skip_db"] is True
