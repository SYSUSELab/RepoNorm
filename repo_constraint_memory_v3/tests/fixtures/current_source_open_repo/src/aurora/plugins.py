from collections.abc import Callable


REGISTRY: list[type[object]] = []


def participant(value: type[object]) -> type[object]:
    REGISTRY.append(value)
    return value


@participant
class JsonParticipant:
    participant_metadata = {"format": "json"}


@participant
class TextParticipant:
    participant_metadata: dict[str, str]
    participant_metadata = dict(format="text")


@participant
class DynamicParticipant:
    pass


def attach_dynamic_metadata(
    target: type[object],
    supplier: Callable[[], dict[str, str]],
) -> None:
    target.participant_metadata = supplier()  # type: ignore[attr-defined]
