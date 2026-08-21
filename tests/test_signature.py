# Tests for signature preservation of @acallable-decorated functions and methods.
#
# The decorated object must expose the original function's signature so that
# IDEs and static analyzers can offer autocomplete and so that
# inspect.signature works at runtime.

import inspect

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
