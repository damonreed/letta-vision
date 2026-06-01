"""User input is checkpointed before the LLM call so failed steps retain the user turn."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from letta.agents.letta_agent_v3 import LettaAgentV3
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message


def _user_message(text: str = "hello") -> Message:
    return Message(
        role=MessageRole.user,
        content=[TextContent(text=text)],
        agent_id="agent-test",
    )


@pytest.mark.asyncio
async def test_persist_input_messages_before_step_checkpoints_and_clears():
    agent = LettaAgentV3.__new__(LettaAgentV3)
    agent._checkpoint_messages = AsyncMock()

    user_msg = _user_message()
    messages = [_user_message("context"), user_msg]

    cleared = await LettaAgentV3._persist_input_messages_before_step(
        agent,
        run_id="run-1",
        step_id="step-1",
        messages=messages,
        input_messages_to_persist=[user_msg],
        dry_run=False,
    )

    assert cleared == []
    agent._checkpoint_messages.assert_awaited_once_with(
        run_id="run-1",
        step_id="step-1",
        new_messages=[user_msg],
        in_context_messages=messages,
    )


@pytest.mark.asyncio
async def test_persist_input_messages_before_step_skips_dry_run():
    agent = LettaAgentV3.__new__(LettaAgentV3)
    agent._checkpoint_messages = AsyncMock()

    user_msg = _user_message()
    pending = [user_msg]

    result = await LettaAgentV3._persist_input_messages_before_step(
        agent,
        run_id="run-1",
        step_id="step-1",
        messages=[user_msg],
        input_messages_to_persist=pending,
        dry_run=True,
    )

    assert result is pending
    agent._checkpoint_messages.assert_not_called()
