"""Accessors for AgentState fields that remain in use but are marked deprecated in schemas."""

from __future__ import annotations

import warnings

from letta.schemas.agent import AgentState
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import Memory


def get_llm_config(agent_state: AgentState) -> LLMConfig:
    """Return the agent's LLM config without DeprecationWarning noise in server logs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return agent_state.llm_config


def get_agent_memory(agent_state: AgentState) -> Memory:
    """Return the agent's in-process memory object without DeprecationWarning noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        return agent_state.memory
