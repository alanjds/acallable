from __future__ import annotations

import asyncio

import pytest

from acallable import awaitable


@awaitable
class Fetcher:
    def __call__(self, url: str) -> str:
        return f"sync: {url}"

    async def __acall__(self, url: str) -> str:
        return f"async: {url}"


def test_full_class_decoration_sync_context():
    """Example from README: sync_main() calls fetcher(...) -> sync result"""
    fetcher = Fetcher()

    def sync_main():
        body = fetcher("https://example.com")  # calls __call__
        return body

    result = sync_main()
    assert result == "sync: https://example.com"


@pytest.mark.asyncio
async def test_full_class_decoration_async_context():
    """Example from README: async_main() uses await fetcher(...) -> calls __acall__"""
    fetcher = Fetcher()

    async def async_main():
        body = await fetcher("https://example.com")  # calls __acall__
        return body

    result = await async_main()
    assert result == "async: https://example.com"


# --- Test 7: Inheritance (as mentioned in README) ---


@awaitable
class BaseClass1:
    def __call__(self) -> str:
        return "base_sync"

    async def __acall__(self) -> str:
        return "base_async"


@awaitable
class BaseClass2:
    def __call__(self) -> str:
        return "base_sync"
    # No __acall__ defined - should default to wrapping


class DerivedClass1(BaseClass1):
    pass


class DerivedClass2(BaseClass2):
    pass


class DerivedClass12(BaseClass1):
    async def __acall__(self) -> str:
        return "derived_async"
    # super defined __call__. Use it


class DerivedClass22(BaseClass2):
    async def __acall__(self) -> str:
        return "derived_async"
    # super defined __call__. Use it


class DerivedClass13(BaseClass1):
    def __call__(self) -> str:
        return "derived_sync"
    # super defined __acall__. Use it


class DerivedClass23(BaseClass2):
    def __call__(self) -> str:
        return "derived_sync"
    # super have no __acall__. Wrap our __call__


class DerivedClass14(BaseClass1):
    def __call__(self) -> str:
        return "derived_sync"

    async def __acall__(self) -> str:
        return "derived_async"


class DerivedClass24(BaseClass2):
    def __call__(self) -> str:
        return "derived_sync"

    async def __acall__(self) -> str:
        return "derived_async"


def test_identity():
    base1 = BaseClass1()
    assert isinstance(base1, BaseClass1)

    base2 = BaseClass2()
    assert isinstance(base2, BaseClass2)


@pytest.mark.parametrize(('baseclass', 'derivedclass', 'sync_expected', 'async_expected'),
    [
    (BaseClass1, DerivedClass1, 'base_sync', 'base_async'),  # base defines both __call__ and __ acall__
    (BaseClass2, DerivedClass2, 'base_sync', 'base_sync'),  # base defines only __call__
    (BaseClass1, DerivedClass12, 'base_sync', 'derived_async'),  # derived defines only __acall__
    (BaseClass2, DerivedClass22, 'base_sync', 'derived_async'),  # derived defines only __acall__
    (BaseClass1, DerivedClass13, 'derived_sync', 'base_async'),  # derived w/ sync, base w/ both
    (BaseClass2, DerivedClass23, 'derived_sync', 'derived_sync'),  # derived w/ sync, base w/ sync,
    (BaseClass1, DerivedClass14, 'derived_sync', 'derived_async'),  # derived w/ both
    (BaseClass2, DerivedClass24, 'derived_sync', 'derived_async'),  # derived w/ both
])
def test_inheritance(baseclass: type, derivedclass: type, sync_expected: str, async_expected: str):
    """Test that inheritance works as documented."""

    derived = derivedclass()
    assert isinstance(derived, baseclass)

    # Inherited sync behavior
    assert derived() == sync_expected

    # Inherited async behavior
    async def test_async():
        result = derived()
        assert not isinstance(result, str)
        assert await result == async_expected

    asyncio.run(test_async())


# --- Test: __init_subclass__ preservation on @awaitable classes ---
@awaitable
class BaseWithInitSubclass:
    """Base that sets a class attribute via __init_subclass__."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.marked = True

    def __call__(self) -> str:
        return "base"


class DerivedWithMark(BaseWithInitSubclass):
    def __call__(self) -> str:
        return "derived"


def test_init_subclass_hook_preserved():
    """The user's own __init_subclass__ should still fire after @awaitable."""
    assert DerivedWithMark.marked is True


def test_init_subclass_hook_derived_dispatches():
    """Derived class with __call__ override should still get the dispatcher."""
    d = DerivedWithMark()
    assert d() == "derived"

    async def run():
        result = d()
        assert not isinstance(result, str)
        assert await result == "derived"

    asyncio.run(run())


@awaitable
class BaseWithInitSubclassKwarg:
    """Base that accepts a keyword arg via __init_subclass__."""

    def __init_subclass__(cls, tag=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.tag = tag

    def __call__(self) -> str:
        return "base"


class DerivedTagged(BaseWithInitSubclassKwarg, tag="hello"):
    def __call__(self) -> str:
        return "tagged"


def test_init_subclass_hook_kwargs():
    """Keyword arguments to __init_subclass__ should be forwarded."""
    assert DerivedTagged.tag == "hello"
    d = DerivedTagged()
    assert d() == "tagged"


# --- Test: __init__ preservation on @awaitable classes ---
@awaitable
class ClassWithInit:
    def __init__(self, value: str):
        self.value = value

    def __call__(self) -> str:
        return self.value


def test_init_preserved():
    """__init__ should work normally on @awaitable classes."""
    obj = ClassWithInit("test_value")
    assert obj.value == "test_value"
    assert obj() == "test_value"
