from pathlib import Path

from letta.constants import FILES_TOOLS
from letta.functions.functions import load_function_set
from letta.utils import sanitize_filename
import letta.functions.function_sets.files as files_module


def _normalize_text_file_name(file_name: str) -> str:
    name = sanitize_filename(file_name.strip())
    if not Path(name).suffix:
        name = f"{name}.txt"
    return name


def test_add_text_file_in_files_tool_set():
    assert "add_text_file" in FILES_TOOLS


def test_add_text_file_function_schema():
    schemas = load_function_set(files_module)
    assert "add_text_file" in schemas
    params = schemas["add_text_file"]["json_schema"]["parameters"]["properties"]
    assert {"folder_id", "file_name", "content"}.issubset(params.keys())


def test_normalize_text_file_name_adds_txt_suffix():
    assert _normalize_text_file_name("notes") == "notes.txt"
    assert _normalize_text_file_name("readme.md") == "readme.md"


def test_agent_has_folder_uses_sources_not_folder_ids():
    from types import SimpleNamespace

    from letta.services.tool_executor.three_tier_file_tools import ThreeTierFileTools

    tools = ThreeTierFileTools.__new__(ThreeTierFileTools)
    agent = SimpleNamespace(sources=[SimpleNamespace(id="source-abc")])
    assert tools._agent_has_folder(agent, "source-abc") is True
    assert tools._agent_has_folder(agent, "source-other") is False
