"""Gemini rejects tool schemas whose array union branch has no items type."""

from letta.llm_api.gemini_schema import sanitize_json_schema_for_gemini, sanitize_openai_tools_for_gemini
from letta.llm_api.openai_client import OpenAIClient
from letta.schemas.enums import AgentType, MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message

SCENECRAFT_VALUE = {
    "type": ["string", "number", "integer", "boolean", "object", "array", "null"],
    "description": "New value for that field. A string, number, boolean, object, array, or null.",
}


def _assert_array_branches_have_item_types(schema: dict) -> None:
    any_of = schema["anyOf"]
    array_branches = [branch for branch in any_of if branch.get("type") == "array"]
    assert array_branches
    for branch in array_branches:
        assert isinstance(branch["items"].get("type"), str)
        assert "anyOf" not in branch["items"]


def test_type_union_array_branch_gets_items_type():
    """The live scenecraft_set_attribute.value schema.

    Google expands the type list into any_of. Index 5 is ``array`` and used to
    have no items schema, which Gemini reports as ``any_of[5].items: missing field``.
    """
    schema = {
        "type": "object",
        "properties": {"value": dict(SCENECRAFT_VALUE)},
        "required": ["value"],
        "additionalProperties": False,
    }

    sanitized = sanitize_json_schema_for_gemini(schema)
    value = sanitized["properties"]["value"]

    assert "additionalProperties" not in sanitized
    assert value["description"].startswith("New value")
    assert "type" not in value
    assert [branch["type"] for branch in value["anyOf"][:5]] == ["string", "number", "integer", "boolean", "object"]
    assert value["anyOf"][-1] == {"type": "null"}
    _assert_array_branches_have_item_types(value)
    # Original schema is left alone for providers that accept type arrays.
    assert SCENECRAFT_VALUE["type"][5] == "array"


def test_anyof_items_without_type_are_hoisted():
    """Untyped JSON values expand to an array branch whose items are only anyOf."""
    schema = {
        "type": "array",
        "description": "Any JSON array",
        "items": {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "object", "additionalProperties": True},
            ]
        },
    }

    sanitized = sanitize_json_schema_for_gemini(schema)

    assert sanitized["description"] == "Any JSON array"
    _assert_array_branches_have_item_types(sanitized)
    assert {"type": "array", "items": {"type": "object"}} in sanitized["anyOf"]
    assert "additionalProperties" not in str(sanitized)


def test_typed_array_items_stay_in_place():
    schema = {"type": "array", "items": {"type": "string", "description": "A tag"}}

    assert sanitize_json_schema_for_gemini(schema) == schema


def test_nullable_primitive_union_is_unchanged():
    schema = {"type": ["string", "null"], "description": "Optional name"}

    assert sanitize_json_schema_for_gemini(schema) == schema


def test_default_is_stripped():
    schema = {"type": "string", "enum": ["character", "auto"], "default": "auto"}

    assert sanitize_json_schema_for_gemini(schema) == {"type": "string", "enum": ["character", "auto"]}


def test_gemini_request_rewrites_tools_and_other_models_do_not():
    tool = {
        "name": "scenecraft_set_attribute",
        "description": "Set one field",
        "parameters": {
            "type": "object",
            "properties": {
                "value": dict(SCENECRAFT_VALUE),
                "asset_type": {"type": "string", "enum": ["auto"], "default": "auto"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    }
    messages = [Message(role=MessageRole.user, content=[TextContent(text="hello")], agent_id="agent-abc")]
    client = OpenAIClient()

    gemini = LLMConfig(
        model="google/gemini-3.8-flash",
        model_endpoint_type="openrouter",
        model_endpoint="https://openrouter.ai/api/v1",
        handle="openrouter/google/gemini-3.8-flash",
        provider_name="openrouter",
        context_window=1_048_576,
    )
    gemini_request = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=gemini,
        tools=[tool],
    )
    value = gemini_request["tools"][0]["function"]["parameters"]["properties"]["value"]
    _assert_array_branches_have_item_types(value)
    assert "default" not in gemini_request["tools"][0]["function"]["parameters"]["properties"]["asset_type"]

    other = gemini.model_copy(update={"model": "anthropic/claude-sonnet-4", "handle": "openrouter/anthropic/claude-sonnet-4"})
    other_request = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=other,
        tools=[tool],
    )
    kept = other_request["tools"][0]["function"]["parameters"]["properties"]["value"]
    assert kept["type"] == SCENECRAFT_VALUE["type"]

    wrapped = sanitize_openai_tools_for_gemini([{"type": "function", "function": tool}])
    assert wrapped[0]["function"]["parameters"]["properties"]["value"]["anyOf"]
