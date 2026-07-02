from typing import Any

from app.storage.base import Storage, StorageError


class R2Storage(Storage):
    """Cloudflare R2 (S3-compatible) storage via boto3.

    R2 has a free tier (10 GB) with $0 egress, so the app streams objects back
    through its own download endpoint instead of minting presigned URLs — simpler
    to set up and no CORS/redirect wiring on the frontend.
    """

    def __init__(
        self,
        *,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        endpoint: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self._endpoint = endpoint or f"https://{account_id}.r2.cloudflarestorage.com"
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._client = client  # injectable for tests

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - depends on optional dep
                raise StorageError(
                    "boto3 is required for R2 storage; run `uv sync` in backend/"
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name="auto",
            )
        return self._client

    def save(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        try:
            self._get_client().put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        except Exception as exc:
            raise StorageError(f"Failed to upload {key!r} to R2: {exc}") from exc

    def load(self, key: str) -> bytes:
        try:
            response = self._get_client().get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as exc:
            raise StorageError(f"Failed to load {key!r} from R2: {exc}") from exc

    def delete(self, key: str) -> None:
        try:
            self._get_client().delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise StorageError(f"Failed to delete {key!r} from R2: {exc}") from exc
