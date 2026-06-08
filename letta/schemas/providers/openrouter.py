import os
from typing import Literal

from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError
from pydantic import Field

from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
from letta.errors import ErrorCode, LLMAuthenticationError, LLMError, LLMPermissionDeniedError
from letta.log import get_logger
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.openai import DEFAULT_EMBEDDING_BATCH_SIZE, OpenAIProvider
from letta.settings import model_settings

logger = get_logger(__name__)

# Default context window for models not in the API response
DEFAULT_CONTEXT_WINDOW = 128000

# OpenRouter /v1/models does not expose native embedding width; map known model IDs.
# See https://openrouter.ai/api/v1/models?output_modalities=embeddings
_OPENROUTER_EMBEDDING_DIMS_BY_ID: dict[str, int] = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
    "google/gemini-embedding-001": 3072,
    "google/gemini-embedding-2": 3072,
    "google/gemini-embedding-2-preview": 3072,
    "baai/bge-m3": 1024,
    "baai/bge-large-en-v1.5": 1024,
    "baai/bge-base-en-v1.5": 768,
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/multilingual-e5-large": 1024,
    "thenlper/gte-base": 768,
    "thenlper/gte-large": 1024,
    "mistralai/mistral-embed-2312": 1024,
    "mistralai/codestral-embed-2505": 1024,
    "qwen/qwen3-embedding-8b": 4096,
    "qwen/qwen3-embedding-4b": 2560,
    "sentence-transformers/all-minilm-l6-v2": 384,
    "sentence-transformers/all-minilm-l12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-minilm-l6-v2": 384,
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
}


class OpenRouterProvider(OpenAIProvider):
    """
    OpenRouter provider - https://openrouter.ai/

    OpenRouter is an OpenAI-compatible API gateway that provides access to
    multiple LLM providers (Anthropic, Meta, Mistral, etc.) through a unified API.
    """

    provider_type: Literal[ProviderType.openrouter] = Field(ProviderType.openrouter, description="The type of the provider.")
    provider_category: ProviderCategory = Field(ProviderCategory.base, description="The category of the provider (base or byok)")
    api_key: str | None = Field(None, description="API key for the OpenRouter API.", deprecated=True)
    base_url: str = Field("https://openrouter.ai/api/v1", description="Base URL for the OpenRouter API.")

    async def check_api_key(self):
        """Check if the API key is valid by making a test request to the OpenRouter API."""
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None
        if not api_key:
            raise ValueError("No API key provided")

        try:
            # Use async OpenAI client pointed at OpenRouter's endpoint
            client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)
            # Just list models to verify API key works
            await client.models.list()
        except AuthenticationError as e:
            raise LLMAuthenticationError(message=f"Failed to authenticate with OpenRouter: {e}", code=ErrorCode.UNAUTHENTICATED)
        except PermissionDeniedError as e:
            raise LLMPermissionDeniedError(message=f"Permission denied by OpenRouter: {e}", code=ErrorCode.PERMISSION_DENIED)
        except AttributeError as e:
            if "_set_private_attributes" in str(e):
                raise LLMError(
                    message=f"OpenRouter endpoint at {self.base_url} returned an unexpected non-JSON response. Verify the base URL and API key.",
                    code=ErrorCode.INTERNAL_SERVER_ERROR,
                )
            raise LLMError(message=f"{e}", code=ErrorCode.INTERNAL_SERVER_ERROR)
        except Exception as e:
            raise LLMError(message=f"{e}", code=ErrorCode.INTERNAL_SERVER_ERROR)

    def get_model_context_window_size(self, model_name: str) -> int | None:
        """Get the context window size for an OpenRouter model.

        OpenRouter models provide context_length in the API response,
        so this is mainly a fallback.
        """
        return DEFAULT_CONTEXT_WINDOW

    async def list_llm_models_async(self) -> list[LLMConfig]:
        """
        Return available OpenRouter models that support tool calling.

        OpenRouter provides a models endpoint that supports filtering by supported_parameters.
        We filter for models that support 'tools' to ensure Letta compatibility.
        """
        from letta.llm_api.openai import openai_get_model_list_async

        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None

        # OpenRouter supports filtering models by supported parameters
        # See: https://openrouter.ai/docs/requests
        extra_params = {"supported_parameters": "tools"}

        response = await openai_get_model_list_async(
            self.base_url,
            api_key=api_key,
            extra_params=extra_params,
        )

        data = response.get("data", response)

        configs = []
        for model in data:
            if "id" not in model:
                logger.warning(f"OpenRouter model missing 'id' field: {model}")
                continue

            model_name = model["id"]

            # OpenRouter returns context_length in the model listing
            if model.get("context_length"):
                context_window_size = model["context_length"]
            else:
                context_window_size = self.get_model_context_window_size(model_name)
                logger.debug(f"Model {model_name} missing context_length, using default: {context_window_size}")

            configs.append(
                LLMConfig(
                    model=model_name,
                    model_endpoint_type="openrouter",
                    model_endpoint=self.base_url,
                    context_window=context_window_size,
                    handle=self.get_handle(model_name),
                    max_tokens=self.get_default_max_output_tokens(model_name),
                    provider_name=self.name,
                    provider_category=self.provider_category,
                )
            )

        return configs

    @staticmethod
    def _model_has_embedding_output(model: dict) -> bool:
        architecture = model.get("architecture")
        if isinstance(architecture, dict):
            output_modalities = architecture.get("output_modalities") or []
            if any(str(m).lower() == "embeddings" for m in output_modalities):
                return True
            modality = str(architecture.get("modality") or "").lower()
            if "embeddings" in modality:
                return True
        return OpenAIProvider._looks_like_embedding_model(model.get("id", ""), model)

    @classmethod
    def _embedding_dim_for_openrouter_model(cls, model: dict, model_name: str) -> int:
        if model_name in _OPENROUTER_EMBEDDING_DIMS_BY_ID:
            return _OPENROUTER_EMBEDDING_DIMS_BY_ID[model_name]

        dim = cls._embedding_dim_from_model_record(model, model_name)
        if dim != 1536 or any(hint in model_name.lower() for hint in ("3-large", "ada-002", "3-small")):
            return dim

        name_lower = model_name.lower()
        if "minilm" in name_lower or "l6-v2" in name_lower:
            return 384
        if "bge-m3" in name_lower:
            return 1024
        if "bge-large" in name_lower or "e5-large" in name_lower or "gte-large" in name_lower:
            return 1024
        if "bge-base" in name_lower or "e5-base" in name_lower or "gte-base" in name_lower:
            return 768
        if "embed" in name_lower and "gemini" in name_lower:
            return 3072

        logger.warning(
            "Unknown OpenRouter embedding dimension for %s; defaulting to 1536. "
            "Override embedding_dim after sync if search quality is poor.",
            model_name,
        )
        return 1536

    async def _openrouter_api_key(self) -> str | None:
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None
        return api_key or model_settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

    async def list_embedding_models_async(self) -> list[EmbeddingConfig]:
        """Discover embedding models from OpenRouter's models API.

        Unlike chat models (filtered with supported_parameters=tools), embeddings are
        listed via output_modalities=embeddings. Do not fall back to the static OpenAI
        embedding catalog — those model IDs are not valid on OpenRouter.
        """
        from letta.llm_api.openai import openai_get_model_list_async

        api_key = await self._openrouter_api_key()

        try:
            response = await openai_get_model_list_async(
                self.base_url,
                api_key=api_key,
                extra_params={"output_modalities": "embeddings"},
            )
        except Exception as e:
            logger.info("Could not list OpenRouter embedding models from %s: %s", self.base_url, e)
            return []

        data = response.get("data", response)
        if not isinstance(data, list):
            return []

        configs: list[EmbeddingConfig] = []
        for model in data:
            if "id" not in model:
                continue
            model_name = model["id"]
            if not self._model_has_embedding_output(model):
                continue
            configs.append(
                EmbeddingConfig(
                    embedding_model=model_name,
                    embedding_endpoint_type="openrouter",
                    embedding_endpoint=self.base_url,
                    embedding_dim=self._embedding_dim_for_openrouter_model(model, model_name),
                    embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
                    handle=self.get_handle(model_name, is_embedding=True),
                    batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
                )
            )

        return configs
