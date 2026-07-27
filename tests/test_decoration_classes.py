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
class BaseClass:
    def __call__(self, value: str) -> str:
        return f"base sync: {value}"

    async def __acall__(self, value: str) -> str:
        return f"base async: {value}"

def test_inheritance():
    """Test that inheritance works as documented."""
    base = BaseClass()

    class DerivedClass(BaseClass):
        pass

    derived = DerivedClass()

    # Inherited sync behavior
    assert derived("test") == "base sync: test"

    # Inherited async behavior
    async def test_async():
        result = derived("test")
        assert not isinstance(result, str)
        assert await result == "base async: test"

    asyncio.run(test_async())

def test_inheritance_acall():
    class DerivedClassAsync(BaseClass):
        async def __acall__(self, value: str) -> str:
            return f"derived async: {value}"

    derived = DerivedClassAsync()

    # Inherited sync behavior
    assert derived("test") == "base sync: test"

    # Inherited async behavior
    async def test_async():
        result = derived("test")
        assert not isinstance(result, str)
        assert await result == "derived async: test"

    asyncio.run(test_async())


def test_inheritance_dundercall():

    class DerivedClassSync(BaseClass):
        def __call__(self, value: str) -> str:
            return f"derived sync: {value}"

    derived = DerivedClassSync()

    # Inherited sync behavior
    assert derived("test") == "derived sync: test"

    # Inherited async behavior
    async def test_async():
        result = derived("test")
        assert not isinstance(result, str)
        assert await result == "base async: test"

    asyncio.run(test_async())



def test_inheritance_dundercall_acall():

    class DerivedClassAll(BaseClass):
        def __call__(self, value: str) -> str:
            return f"derived sync: {value}"

        async def __acall__(self, value: str) -> str:
            return f"derived async: {value}"

    derived = DerivedClassAll()

    # Inherited sync behavior
    assert derived("test") == "derived sync: test"

    # Inherited async behavior
    async def test_async():
        result = derived("test")
        assert not isinstance(result, str)
        assert await result == "derived async: test"

    asyncio.run(test_async())

# --- Test 8: Default async wrapper without explicit @acall ---

@awaitable
class SimpleFetcher:
    def __call__(self, msg: str) -> str:
        return f"sync: {msg}"
    # No __acall__ defined - should default to wrapping

def test_default_async_wrapper_for_class():
    """Test class without explicit __acall__ gets default wrapper."""
    fetcher = SimpleFetcher()

    # Sync context: direct call returns sync result
    assert fetcher("hello") == "sync: hello"

    # Async context: should return coroutine
    async def test_async():
        result = fetcher("hello")
        assert not isinstance(result, str)
        assert await result == "sync: hello"

    asyncio.run(test_async())

# --- Test 11: @awaitable applied to a subclass that only overrides __acall__ ---


@awaitable
class BaseClass2:
    def __call__(self, value: str) -> str:
        return f"base sync: {value}"

    async def __acall__(self, value: str) -> str:
        return f"base async: {value}"


@awaitable
class DerivedClass2(BaseClass2):
    async def __acall__(self, value: str) -> str:
        return f"derived async: {value}"


def test_decorated_subclass_acall_only_sync():
    """@awaitable on a subclass that only overrides __acall__ — sync path."""
    derived = DerivedClass2()
    assert derived("test") == "base sync: test"


@pytest.mark.asyncio
async def test_decorated_subclass_acall_only_async():
    """@awaitable on a subclass that only overrides __acall__ — async path."""
    derived = DerivedClass2()
    result = await derived("test")
    assert result == "derived async: test"


# --- Test 12: @awaitable applied to a subclass that overrides both ---


@awaitable
class DerivedClassAll2(BaseClass2):
    def __call__(self, value: str) -> str:
        return f"derived sync: {value}"

    async def __acall__(self, value: str) -> str:
        return f"derived async: {value}"


def test_decorated_subclass_both_sync():
    """@awaitable on a subclass that overrides both — sync path."""
    derived = DerivedClassAll2()
    assert derived("test") == "derived sync: test"


@pytest.mark.asyncio
async def test_decorated_subclass_both_async():
    """@awaitable on a subclass that overrides both — async path."""
    derived = DerivedClassAll2()
    result = await derived("test")
    assert result == "derived async: test"
