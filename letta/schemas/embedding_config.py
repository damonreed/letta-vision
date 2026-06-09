import hashlib
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE, DEPLOYMENT_EMBEDDING_DIM


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model connection and processing parameters."""

    embedding_endpoint_type: Literal[
        "openai",
        "anthropic",
        "bedrock",
        "google_ai",
        "google_vertex",
        "azure",
        "groq",
        "ollama",
        "webui",
        "webui-legacy",
        "lmstudio",
        "lmstudio-legacy",
        "llamacpp",
        "koboldcpp",
        "vllm",
        "hugging-face",
        "mistral",
        "together",  # completions endpoint
        "pinecone",
        "openrouter",
    ] = Field(..., description="The endpoint type for the model.")
    embedding_endpoint: Optional[str] = Field(None, description="The endpoint for the model (`None` if local).")
    embedding_model: str = Field(..., description="The model for the embedding.")
    embedding_dim: int = Field(..., description="The dimension of the embedding.")
    embedding_chunk_size: Optional[int] = Field(300, description="The chunk size of the embedding.")
    handle: Optional[str] = Field(None, description="The handle for this config, in the format provider/model-name.")
    batch_size: int = Field(32, description="The maximum batch size for processing embeddings.")

    output_dimensionality: Optional[int] = Field(
        None, description="MRL target dim sent to the provider. Stored width equals this when set."
    )
    input_type: Optional[str] = Field("search_document", description="Doc-side input_type hint at ingest.")
    normalize: bool = Field(False, description="Client L2-normalizes returned vectors. Required when truncating below native.")
    embedding_space_id: Optional[str] = Field(
        None, description="Stable hash of the embedding space tuple. Computed, not user-set."
    )

    # azure only
    azure_endpoint: Optional[str] = Field(None, description="The Azure endpoint for the model.")
    azure_version: Optional[str] = Field(None, description="The Azure version for the model.")
    azure_deployment: Optional[str] = Field(None, description="The Azure deployment for the model.")

    def compute_space_id(self) -> str:
        """Stable sha256-hex prefix (16 chars) of the embedding space tuple."""
        parts = [
            self.embedding_endpoint_type,
            self.embedding_model,
            str(self.embedding_dim),
            str(bool(self.normalize)),
            self.input_type or "search_document",
        ]
        payload = "|".join(parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def ensure_space_id(self) -> "EmbeddingConfig":
        if not self.embedding_space_id:
            self.embedding_space_id = self.compute_space_id()
        return self

    @model_validator(mode="after")
    def _populate_space_id(self) -> "EmbeddingConfig":
        if not self.embedding_space_id:
            self.embedding_space_id = self.compute_space_id()
        return self

    @classmethod
    def default_config(cls, model_name: Optional[str] = None, provider: Optional[str] = None):
        if model_name in (
            "google/gemini-embedding-2-preview",
            "gemini-embedding-2",
            "google/gemini-embedding-2",
        ) or (provider == "openrouter" and model_name and "gemini-embedding-2" in model_name):
            return cls(
                embedding_model="google/gemini-embedding-2-preview",
                embedding_endpoint_type="openrouter",
                embedding_endpoint="https://openrouter.ai/api/v1",
                embedding_dim=DEPLOYMENT_EMBEDDING_DIM,
                output_dimensionality=DEPLOYMENT_EMBEDDING_DIM,
                input_type="search_document",
                normalize=True,
                embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
                handle="openrouter/google/gemini-embedding-2-preview",
            )
        if model_name == "text-embedding-ada-002" and provider == "openai":
            return cls(
                embedding_model="text-embedding-ada-002",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                embedding_dim=1536,
                embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
            )
        if (model_name == "text-embedding-3-small" and provider == "openai") or (not model_name and provider == "openai"):
            return cls(
                embedding_model="text-embedding-3-small",
                embedding_endpoint_type="openai",
                embedding_endpoint="https://api.openai.com/v1",
                # OpenAI default dimension for text-embedding-3-small.
                embedding_dim=1536,
                embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
            )
        elif model_name == "letta":
            return cls(
                embedding_endpoint="https://embeddings.letta.com/",
                embedding_model="letta-free",
                embedding_dim=1536,
                embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
                embedding_endpoint_type="openai",
            )
        elif provider == "pinecone":
            # default config for pinecone with empty endpoint
            return cls(
                embedding_endpoint=None,
                embedding_model="llama-text-embed-v2",
                embedding_dim=1536,  # assuming default openai dimension
                embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
                embedding_endpoint_type="pinecone",
            )
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def pretty_print(self) -> str:
        return (
            f"{self.embedding_model}"
            + (f" [type={self.embedding_endpoint_type}]" if self.embedding_endpoint_type else "")
            + (f" [ip={self.embedding_endpoint}]" if self.embedding_endpoint else "")
        )
