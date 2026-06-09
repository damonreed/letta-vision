"""Content-addressed object store (MinIO/S3 + GCS)."""

from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache
from typing import Optional
from urllib.parse import parse_qs, urlparse

import aioboto3
from botocore.config import Config

from letta.log import get_logger
from letta.settings import settings

logger = get_logger(__name__)


def _wire_byte_size(raw: bytes) -> int:
    """Base64-encoded size as it appears in provider JSON."""
    return len(base64.standard_b64encode(raw))


class ObjectStoreClient:
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self._session = aioboto3.Session()

    def _key(self, content_hash: str, suffix: str = "") -> str:
        base = f"sha256/{content_hash}{suffix}"
        return f"{self.prefix}/{base}" if self.prefix else base

    def _client_kwargs(self) -> dict:
        kwargs = {
            "service_name": "s3",
            "endpoint_url": self.endpoint_url,
            "aws_access_key_id": os.environ.get("MINIO_ROOT_USER") or os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("MINIO_ROOT_PASSWORD") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "config": Config(signature_version="s3v4"),
        }
        return {k: v for k, v in kwargs.items() if v is not None}

    async def put_bytes(self, content_hash: str, data: bytes, *, suffix: str = "") -> str:
        key = self._key(content_hash, suffix=suffix)
        async with self._session.client("s3", **self._client_kwargs()) as client:
            await client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return key

    async def get_bytes(self, key: str) -> bytes:
        async with self._session.client("s3", **self._client_kwargs()) as client:
            resp = await client.get_object(Bucket=self.bucket, Key=key)
            return await resp["Body"].read()

    async def presigned_get_url(self, key: str, expires_seconds: int = 3600) -> str:
        async with self._session.client("s3", **self._client_kwargs()) as client:
            return await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )

    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def wire_byte_size(data: bytes) -> int:
        return _wire_byte_size(data)


def _parse_object_store_uri(uri: str) -> tuple[str, str, Optional[str]]:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    endpoint = qs.get("endpoint", [None])[0]
    return bucket, prefix, endpoint


@lru_cache
def get_object_store_client() -> ObjectStoreClient:
    uri = settings.object_store_uri
    if not uri or not uri.startswith("s3://"):
        raise ValueError("LETTA_OBJECT_STORE_URI must be set to an s3:// URI for image storage")
    bucket, prefix, endpoint = _parse_object_store_uri(uri)
    return ObjectStoreClient(bucket=bucket, prefix=prefix, endpoint_url=endpoint)
