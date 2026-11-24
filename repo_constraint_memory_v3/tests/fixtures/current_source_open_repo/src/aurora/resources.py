from contextlib import asynccontextmanager
from typing import AsyncIterator


class Channel:
    async def close(self) -> None:
        return None


@asynccontextmanager
async def managed_channel(channel: Channel) -> AsyncIterator[Channel]:
    try:
        yield channel
    finally:
        await channel.close()


async def upload(channel: Channel, payload: bytes) -> int:
    async with managed_channel(channel) as owned:
        await owned.close()
        return len(payload)


async def download(channel: Channel) -> bytes:
    async with managed_channel(channel):
        return b"payload"


async def dynamic_cleanup(owner: object) -> None:
    cleanup = getattr(owner, "cleanup", None)
    if cleanup is not None:
        await cleanup()
