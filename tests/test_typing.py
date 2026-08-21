# Static typing checks for acallable, verifiable by ty / basedpyright.
#
# ``typing.assert_type`` is a no-op at runtime (returns its first
# argument) but is verified statically by type checkers, so this file
# doubles as a normal pytest test and a type-checking fixture.

from collections.abc import Awaitable, Coroutine
from typing import assert_type

from acallable import acallable


@acallable
def fetch(url: str, timeout: float = 1.0) -> str:
    return f'sync: {url}'


@fetch.acall
async def fetch(url: str, timeout: float = 1.0) -> str:
    return f'async: {url}'


def test_call_returns_sync_or_async():
    # __call__ dispatches sync/async, so the static type is the union.
    assert_type(fetch('https://example.com'), str | Awaitable[str])


def test_call_accepts_keyword_args():
    # Keyword parameters are visible thanks to ParamSpec.
    assert_type(fetch('https://example.com', timeout=5.0), str | Awaitable[str])


def test_sync_property_returns_plain_value():
    assert_type(fetch.sync('https://example.com'), str)


def test_acall_property_returns_awaitable():
    # __acall__ is the async callable; calling it yields Awaitable[str].
    coro = fetch.__acall__('https://example.com')
    assert_type(coro, Awaitable[str])
    if isinstance(coro, Coroutine):
        coro.close()
