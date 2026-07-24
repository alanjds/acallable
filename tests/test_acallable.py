import asyncio
import pytest
from acallable import awaitable


def test_simple_function():
    @awaitable
    def greet(name: str) -> str:
        return f"Hello, {name}"

    assert greet("world") == "Hello, world"


@pytest.mark.asyncio
async def test_function_with_async_body():
    @awaitable
    def fetch(url: str) -> str:
        return f"sync: {url}"

    @fetch.acall
    async def fetch_async(url: str) -> str:
        return f"async: {url}"

    # 1. Direct call in async context returns coroutine
    result = fetch("example.com")
    assert not isinstance(result, str)
    assert await result == "async: example.com"

    # 2. Awaited direct call runs async body
    assert await fetch("example.com") == "async: example.com"


@pytest.mark.asyncio
async def test_default_async_wrapper():
    @awaitable
    def compute(x: int) -> int:
        return x * 2

    # Direct call in async context returns coroutine
    result = compute(3)
    assert await result == 6

    # Awaited call works
    assert await compute(3) == 6


def test_default_async_wrapper_sync_context():
    """In sync context, direct call returns sync result."""
    @awaitable
    def compute(x: int) -> int:
        return x * 2

    assert compute(3) == 6
    # In sync context no await needed


@pytest.mark.asyncio
async def test_proper_class_decoration():
    @awaitable
    class ProperFetcher:
        def __call__(self, url: str) -> str:
            return f"sync: {url}"

        async def __acall__(self, url: str) -> str:
            return f"async: {url}"

    fetcher = ProperFetcher()

    # Direct call in async context returns coroutine
    result = fetcher("test")
    assert await result == "async: test"

    # Awaited direct call works
    assert await fetcher("test") == "async: test"


@pytest.mark.asyncio
async def test_async_context_in_asyncio_gather():
    @awaitable
    def compute(i: int) -> int:
        return i * 2

    results = await asyncio.gather(*(compute(i) for i in range(3)))
    assert results == [0, 2, 4]


# --- User-required examples ---

async def foo():
    """Async context — direct call returns coroutine (not a str)."""
    @awaitable
    def fetch(url: str) -> str:
        return f"sync: {url}"

    @fetch.acall
    async def fetch_async(url: str) -> str:
        return f"async: {url}"

    x = fetch("example.com")
    assert not isinstance(x, str)
    y = await x
    assert y == "async: example.com"


async def foo2():
    """Async context — fetch then await."""
    @awaitable
    def fetch(url: str) -> str:
        return f"sync: {url}"

    @fetch.acall
    async def fetch_async(url: str) -> str:
        return f"async: {url}"

    x = fetch("example.com")
    y = await x
    assert not isinstance(x, str)
    assert y == "async: example.com"


def foo3():
    """Sync context — direct call returns sync result."""
    @awaitable
    def fetch(url: str) -> str:
        return f"sync: {url}"

    @fetch.acall
    async def fetch_async(url: str) -> str:
        return f"async: {url}"

    x = fetch("example.com")
    assert isinstance(x, str)
    assert x == "sync: example.com"


@pytest.mark.asyncio
async def test_foo_async_context():
    await foo()


@pytest.mark.asyncio
async def test_foo2_async_context():
    await foo2()


def test_foo3_sync_context():
    foo3()