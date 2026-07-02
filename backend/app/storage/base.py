from abc import ABC, abstractmethod


class StorageError(RuntimeError):
    """Raised when a storage backend fails to save, load, or delete an object."""


class Storage(ABC):
    """Backend-agnostic blob storage keyed by an opaque string key.

    Implementations must be safe to share as a process-wide singleton. The key
    is stored on ``Document.storage_path`` and is the only handle the rest of the
    app needs — it never sees a filesystem path or bucket detail.
    """

    @abstractmethod
    def save(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        """Store ``data`` under ``key`` (overwriting any existing object)."""

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Return the bytes stored under ``key``; raise ``StorageError`` if missing."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key`` if present; deleting a missing key is not an error."""
