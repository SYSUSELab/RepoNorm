def serialize_timestamp(value: object) -> str:
    serializer = getattr(value, "to_repository_timestamp", None)
    if serializer is None:
        raise TypeError("timestamp serializer is required")
    return serializer()


def serialize_deadline(value: object) -> str:
    serializer = getattr(value, "to_repository_timestamp", None)
    if serializer is None:
        raise TypeError("timestamp serializer is required")
    return serializer()
