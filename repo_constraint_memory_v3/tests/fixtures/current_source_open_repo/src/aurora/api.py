from aurora.errors import AuroraPublicError, InternalStorageError


def _read_storage(key: str) -> bytes:
    if not key:
        raise InternalStorageError("missing key")
    return key.encode()


def fetch_document(key: str) -> bytes:
    try:
        return _read_storage(key)
    except InternalStorageError as error:
        raise AuroraPublicError(str(error)) from error


def fetch_metadata(key: str) -> int:
    try:
        return len(_read_storage(key))
    except InternalStorageError as error:
        raise AuroraPublicError(str(error)) from error


def internal_probe(key: str) -> bytes:
    return _read_storage(key)


def unrelated_conversion(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise TypeError("not an integer") from error
