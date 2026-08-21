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
    """Fetch some `url` with a `timeout`"""
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


# --- bound method .__acall__ typing ---
#
# At the class level, Store.save is an Acallable whose P ParamSpec
# includes `self`, so we pass the instance explicitly:
#   Store.save.__acall__(store, 'r')
#
# The *bound* form (store.save.__acall__('r')) auto-binds self at runtime
# via __get__ + functools.partial — see test_signature.py for runtime
# verification. Static typing of the bound form is a known limitation
# because ParamSpec cannot express "P without the leading self".


class Store:
    @acallable
    def save(self, record: str) -> None:
        self.last_sync_result = record

    @save.acall
    async def save(self, record: str) -> None:
        self.last_async_result = record


def test_bound_method_acall_typing():
    """Store.save.__acall__(store, 'r') -> Awaitable[None] (self explicit)."""
    store = Store()
    coro = Store.save.__acall__(store, 'r')
    assert_type(coro, Awaitable[None])
    if isinstance(coro, Coroutine):
        coro.close()


def test_bound_method_sync_property_typing():
    """Store.save.sync(store, 'r') -> None (self explicit)."""
    store = Store()
    result = Store.save.sync(store, 'r')
    assert_type(result, None)


def test_bound_method_call_typing():
    """Store.save(store, 'r') -> None | Awaitable[None] (self explicit)."""
    store = Store()
    result = Store.save(store, 'r')
    assert_type(result, None | Awaitable[None])
