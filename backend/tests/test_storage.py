import pytest

from app.storage.base import StorageError
from app.storage.local import LocalStorage
from app.storage.r2 import R2Storage


def test_local_storage_roundtrip(tmp_path) -> None:
    storage = LocalStorage(tmp_path)
    key = "documents/abc/notes.pdf"

    storage.save(key, b"%PDF-1.4 hello")
    assert storage.load(key) == b"%PDF-1.4 hello"

    storage.delete(key)
    with pytest.raises(StorageError):
        storage.load(key)


def test_local_storage_delete_missing_is_noop(tmp_path) -> None:
    LocalStorage(tmp_path).delete("documents/missing/x.pdf")  # must not raise


def test_local_storage_rejects_path_traversal(tmp_path) -> None:
    storage = LocalStorage(tmp_path)
    with pytest.raises(StorageError):
        storage.save("../escape.pdf", b"nope")


class FakeS3Client:
    """Minimal in-memory stand-in for a boto3 S3 client."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 - boto3 kwarg names
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": _Body(self.objects[Key])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _r2(client: FakeS3Client) -> R2Storage:
    return R2Storage(
        account_id="acct",
        access_key_id="k",
        secret_access_key="s",
        bucket="pdfs",
        client=client,
    )


def test_r2_storage_roundtrip() -> None:
    client = FakeS3Client()
    storage = _r2(client)

    storage.save("documents/1/a.pdf", b"data", "application/pdf")
    assert client.objects["documents/1/a.pdf"] == b"data"
    assert storage.load("documents/1/a.pdf") == b"data"

    storage.delete("documents/1/a.pdf")
    assert "documents/1/a.pdf" not in client.objects


def test_r2_storage_load_missing_raises() -> None:
    with pytest.raises(StorageError):
        _r2(FakeS3Client()).load("documents/missing.pdf")


def test_r2_endpoint_defaults_to_account_host() -> None:
    storage = R2Storage(
        account_id="myacct",
        access_key_id="k",
        secret_access_key="s",
        bucket="pdfs",
    )
    assert storage._endpoint == "https://myacct.r2.cloudflarestorage.com"
