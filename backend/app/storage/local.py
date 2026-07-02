from pathlib import Path

from app.storage.base import Storage, StorageError


class LocalStorage(Storage):
    """Filesystem storage under a root directory. Default for local dev (zero setup)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _resolve(self, key: str) -> Path:
        # Resolve the key under root and reject anything that escapes it (e.g. "../").
        root = self.root.resolve()
        path = (root / key).resolve()
        if root != path and root not in path.parents:
            raise StorageError(f"Invalid storage key: {key!r}")
        return path

    def save(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def load(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"Object not found: {key!r}") from exc

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)
