from functools import lru_cache

from app.config import get_settings
from app.storage.base import Storage, StorageError
from app.storage.local import LocalStorage
from app.storage.r2 import R2Storage

__all__ = ["Storage", "StorageError", "get_storage"]

_R2_REQUIRED = ("r2_account_id", "r2_access_key_id", "r2_secret_access_key", "r2_bucket")


@lru_cache
def get_storage() -> Storage:
    """Return the process-wide storage backend selected by ``STORAGE_BACKEND``.

    Defaults to ``local`` (filesystem, zero setup). Set ``STORAGE_BACKEND=r2`` plus
    the ``R2_*`` credentials to use Cloudflare R2.
    """
    settings = get_settings()
    backend = settings.storage_backend.lower()

    if backend == "local":
        return LocalStorage(settings.upload_dir)

    if backend == "r2":
        missing = [name.upper() for name in _R2_REQUIRED if not getattr(settings, name)]
        if missing:
            raise StorageError(f"STORAGE_BACKEND=r2 requires: {', '.join(missing)}")
        return R2Storage(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket=settings.r2_bucket,
            endpoint=settings.r2_endpoint,
        )

    raise StorageError(f"Unknown STORAGE_BACKEND: {settings.storage_backend!r} (use 'local' or 'r2')")
