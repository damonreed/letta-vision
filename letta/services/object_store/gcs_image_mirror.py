"""GCS mirror for image bytes.

Canonical MinIO keys stay content-addressed. This mirror stores one object per
image id at ``gs://{bucket}/{image_id}`` so Zapimage can fetch a short-lived
HTTPS signed URL instead of a multi-megabyte data URI over MCP.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from letta.log import get_logger
from letta.settings import settings

logger = get_logger(__name__)


class GcsImageMirror:
    """Upload and sign image objects named by image id."""

    def __init__(
        self,
        bucket: str,
        *,
        project: str | None = None,
        signer_service_account: str | None = None,
        signed_url_ttl_seconds: int = 3600,
    ):
        self.bucket_name = bucket
        self.project = project
        self.signer_service_account = signer_service_account
        self.signed_url_ttl_seconds = signed_url_ttl_seconds
        self._client = None

    def gs_uri(self, image_id: str) -> str:
        return f"gs://{self.bucket_name}/{image_id}"

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self.project or None)
        return self._client

    def _blob(self, image_id: str):
        return self._get_client().bucket(self.bucket_name).blob(image_id)

    def _put_sync(self, image_id: str, data: bytes, content_type: str) -> str:
        blob = self._blob(image_id)
        blob.upload_from_string(data, content_type=content_type or "application/octet-stream")
        return self.gs_uri(image_id)

    def _exists_sync(self, image_id: str) -> bool:
        return self._blob(image_id).exists()

    def _signed_url_sync(self, image_id: str) -> str:
        from google.auth import default
        from google.auth.transport.requests import Request

        blob = self._blob(image_id)
        credentials, _project = default()
        kwargs: dict = {
            "version": "v4",
            "expiration": timedelta(seconds=self.signed_url_ttl_seconds),
            "method": "GET",
            "credentials": credentials,
        }
        # Service-account JSON keys can sign locally with the private key.
        # Keyless ADC has no signer — refresh a token and use IAM signBlob.
        if getattr(credentials, "signer", None) is None:
            if not credentials.valid:
                credentials.refresh(Request())
            if self.signer_service_account:
                kwargs["service_account_email"] = self.signer_service_account
                kwargs["access_token"] = credentials.token
        return blob.generate_signed_url(**kwargs)

    async def put_image(self, image_id: str, data: bytes, content_type: str) -> str:
        uri = await asyncio.to_thread(self._put_sync, image_id, data, content_type)
        logger.info("Mirrored %s to %s (%d bytes)", image_id, uri, len(data))
        return uri

    async def exists(self, image_id: str) -> bool:
        return await asyncio.to_thread(self._exists_sync, image_id)

    async def signed_get_url(self, image_id: str) -> str:
        return await asyncio.to_thread(self._signed_url_sync, image_id)


@lru_cache
def get_gcs_image_mirror() -> Optional[GcsImageMirror]:
    """Return the configured mirror, or None when GCS mirroring is disabled."""
    bucket = (settings.image_gcs_bucket or "").strip()
    if not bucket:
        return None
    return GcsImageMirror(
        bucket=bucket,
        project=(settings.image_gcs_project or settings.object_store_project or None),
        signer_service_account=(settings.image_gcs_signer_service_account or None),
        signed_url_ttl_seconds=int(settings.image_gcs_signed_url_ttl_seconds or 3600),
    )


async def mirror_image_bytes(image_id: str, data: bytes, content_type: str) -> str | None:
    """Write image bytes to GCS when mirroring is enabled. Returns gs:// URI."""
    mirror = get_gcs_image_mirror()
    if mirror is None:
        return None
    return await mirror.put_image(image_id, data, content_type)


async def ensure_mirrored_and_sign(
    image_id: str,
    *,
    minio_key: str,
    content_type: str,
) -> str:
    """Return a signed HTTPS URL for ``image_id``, copying from MinIO if needed."""
    mirror = get_gcs_image_mirror()
    if mirror is None:
        raise RuntimeError("LETTA_IMAGE_GCS_BUCKET is not configured")

    if not await mirror.exists(image_id):
        from letta.services.object_store.client import get_object_store_client

        raw = await get_object_store_client().get_bytes(minio_key)
        await mirror.put_image(image_id, raw, content_type)

    return await mirror.signed_get_url(image_id)
