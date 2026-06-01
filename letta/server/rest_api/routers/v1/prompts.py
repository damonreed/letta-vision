from fastapi import APIRouter, HTTPException, status

from letta.prompts.gpt_system import get_system_text
from letta.prompts.system_prompts import SYSTEM_PROMPTS

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.get("/system", operation_id="list_system_prompt_keys")
def list_system_prompt_keys() -> list[str]:
    """Return keys for built-in system prompt templates shipped with the server."""
    return sorted(SYSTEM_PROMPTS.keys())


@router.get("/system/{key}", operation_id="get_system_prompt_template")
def get_system_prompt_template(key: str) -> dict[str, str]:
    """Return the raw template text for a built-in system prompt (e.g. letta_v1)."""
    try:
        text = get_system_text(key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    return {"key": key, "text": text}
