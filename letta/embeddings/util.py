"""Embedding vector utilities."""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np

from letta.constants import DEPLOYMENT_EMBEDDING_DIM
from letta.schemas.embedding_config import EmbeddingConfig


def l2_normalize(vec: Sequence[float]) -> List[float]:
    """L2-normalize a vector. Required after MRL truncation below native dim."""
    arr = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        return list(vec)
    return (arr / norm).tolist()


def prepare_vector_for_write(
    vec: Union[Sequence[float], np.ndarray],
    config: EmbeddingConfig,
) -> List[float]:
    """Prepare an embedding vector for storage at deployment native width (no 4096 padding)."""
    arr = np.asarray(vec, dtype=np.float64).flatten()
    target_dim = DEPLOYMENT_EMBEDDING_DIM
    config_dim = config.embedding_dim or target_dim
    if arr.shape[0] != config_dim:
        raise ValueError(
            f"Embedding dim {arr.shape[0]} does not match model config dim {config_dim}. "
            f"Passage storage requires {target_dim}-dim vectors on this deployment "
            f"(use openrouter/google/gemini-embedding-2-preview for folders)."
        )
    if config_dim != target_dim:
        raise ValueError(
            f"Embedding model '{config.handle or config.embedding_model}' uses {config_dim}-dim vectors; "
            f"this deployment stores {target_dim}-dim passage embeddings. "
            "Recreate the folder with openrouter/google/gemini-embedding-2-preview."
        )
    if config.normalize:
        return l2_normalize(arr)
    return arr.tolist()
