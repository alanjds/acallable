# Tests for signature preservation of @acallable-decorated functions and methods.
#
# The decorated object must expose the original function's signature so that
# IDEs and static analyzers can offer autocomplete and so that
# inspect.signature works at runtime.

import asyncio
import inspect

import pytest

from acallable import acallable

# --- module-level function ---

@acallable
def fetch(url: str, timeout: float = 1.0) -> str:
    return f'sync: {url}'


@fetch.acall
async def fetch(url: str, timeout: float = 1.0) -> str:
    return f'async: {url}'


def test_module_level_function_signature():
    sig = inspect.signature(fetch)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['url', 'timeout']
    assert params[0].annotation is str
    assert params[1].annotation is float
    assert params[1].default == 1.0
    assert sig.return_annotation is str


def test_sync_property_signature():
    sig = inspect.signature(fetch.sync)
    assert list(sig.parameters) == ['url', 'timeout']


def test_acall_property_signature():
    sig = inspect.signature(fetch.__acall__)
    assert list(sig.parameters) == ['url', 'timeout']


# --- default async wrapper (no explicit @fn.acall) ---

@acallable
def compute(x: int, y: int = 2) -> int:
    return x * y


def test_default_acall_wrapper_preserves_signature():
    sig = inspect.signature(compute.__acall__)
    assert list(sig.parameters) == ['x', 'y']


# --- bound method ---

class DataStore:
    @acallable
    def save(self, record: dict, force: bool = False) -> None:
        self.last_sync_result = f'sync: {record}'

    @save.acall
    async def save(self, record: dict, force: bool = False) -> None:
        self.last_async_result = f'async: {record}'


def test_bound_method_signature():
    store = DataStore()
    sig = inspect.signature(store.save)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['record', 'force']
    assert params[0].annotation is dict
    assert params[1].annotation is bool
    assert params[1].default is False


def test_bound_method_sync_property_signature():
    store = DataStore()
    sig = inspect.signature(store.save.sync)
    assert list(sig.parameters) == ['record', 'force']


def test_bound_method_acall_property_signature():
    store = DataStore()
    sig = inspect.signature(store.save.__acall__)
    assert list(sig.parameters) == ['record', 'force']


# --- unbound (class-level) access ---

def test_unbound_method_signature():
    sig = inspect.signature(DataStore.save)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['self', 'record', 'force']
    assert params[0].annotation is inspect.Parameter.empty


# --- bound method: __acall__ runtime usage (the edge case) ---
#
# Verifies that calling .__acall__ on a bound method auto-binds `self`
# via the functools.partial in Acallable.__get__. Without this binding,
# store.method.__acall__('x') would fail with:
#   TypeError: missing 1 required positional argument: 'self'


class ConfigStore:
    """A method with a DEFAULT async wrapper (no explicit @method.acall).

    Used to verify that __acall__ binds `self` even for the auto-generated
    default async wrapper.
    """

    @acallable
    def load(self, key: str) -> str:
        self.last_sync_result = f'sync: {key}'
        return f'sync: {key}'


def test_bound_method_acall_direct_call_binds_self():
    """Calling store.save.__acall__('rec') must auto-bind self."""
    store = DataStore()
    coro = store.save.__acall__('rec', force=True)
    result = asyncio.run(coro)
    assert result is None
    assert store.last_async_result == 'async: rec'


def test_bound_method_acall_default_wrapper_binds_self():
    """Same edge case for the DEFAULT async wrapper (no explicit @fn.acall).

    The auto-generated default wrapper must also have self pre-bound
    when __acall__ is accessed via __get__ on an instance.
    """
    store = ConfigStore()
    coro = store.load.__acall__('mykey')
    result = asyncio.run(coro)
    assert result == 'sync: mykey'
    assert store.last_sync_result == 'sync: mykey'


def test_bound_method_acall_default_wrapper_signature_self_removed():
    """__acall__ signature on a bound method (default wrapper) has self stripped."""
    store = ConfigStore()
    sig = inspect.signature(store.load.__acall__)
    assert list(sig.parameters) == ['key']


async def _call_bound_load_in_async(store, key):
    """Helper: calls store.load from within an async def (forces __acall__ dispatch)."""
    await store.load(key)


@pytest.mark.asyncio
async def test_bound_method_call_async_context_dispatches_to_acall():
    """Calling store.load('key') from async context dispatches to __acall__
    with self already bound via the partial in __get__."""
    store = ConfigStore()
    await store.load('mykey')
    assert store.last_sync_result == 'sync: mykey'


def test_unbound_method_acall_signature_includes_self():
    """At class level (unbound), __acall__ still carries 'self' in its signature."""
    sig = inspect.signature(ConfigStore.load.__acall__)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ['self', 'key']
