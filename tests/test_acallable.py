import asyncio
import pytest
from acallable import awaitable

# --- Existing tests remain ---

# --- Test 1: Basic sync/async function with explicit @fn.acall ---
@awaitable
def fetch(url: str) -> str:
    return f"sync: {url}"

@fetch.acall
async def fetch(url: str) -> str:
    return f"async: {url}"

def test_basic_sync_function_without_acall():
    """Test that @fetch('example.com') from sync context returns sync result."""
    assert fetch("example.com") == "sync: example.com"

async def test_basic_sync_function_from_async_context():
    """Test that fetch('example.com') from async context returns coroutine to be awaited."""
    result = fetch("example.com")
    assert not isinstance(result, str)
    assert await result == "async: example.com"

# --- Test 2: my_sync_main / my_async_main style examples ---

def my_sync_main():
    """Example from README: sync context direct call."""
    body = fetch("https://example.com")  # from sync context
    return body

async def my_async_main():
    """Example from README: async context indirect call (via await)."""
    body = await fetch("https://example.com")  # from async context
    return body

def test_my_sync_main_example():
    """Test the my_sync_main pattern from README."""
    result = my_sync_main()
    assert result == "sync: https://example.com"

@pytest.mark.asyncio
async def test_my_async_main_example():
    """Test the my_async_main pattern from README."""
    result = await my_async_main()
    assert result == "async: https://example.com"

# --- Test 3: Function without explicit async body (default wrapper behavior) ---

@awaitable
def compute(x: int) -> int:
    return x * 2

def test_default_async_wrapper_sync_context():
    """From README: def foo(): return compute(3) -> returns 6"""
    assert compute(3) == 6

@pytest.mark.asyncio
async def test_default_async_wrapper_async_context():
    """From README: async def foo(): return await compute(3) -> returns 6"""
    assert await compute(3) == 6

# --- Test 4: Class method decoration ---

class DataStore:
    def __init__(self):
        pass
    
    @awaitable
    def save(self, record: dict) -> None:
        self.last_sync_result = f"sync: {record}"
    
    @save.acall
    async def save(self, record: dict) -> None:
        self.last_async_result = f"async: {record}"

def test_class_method_decorator_sync_context():
    """Example from README: def foo(): store.save({"k": "v"}) -> sync"""
    store = DataStore()
    def foo():
        store.save({"k": "v"})
    foo()
    assert store.last_sync_result == "sync: {'k': 'v'}"

@pytest.mark.asyncio
async def test_class_method_decorator_async_context():
    """Example from README: async def foo(): await store.save({"k": "v"}) -> async"""
    store = DataStore()
    async def foo():
        await store.save({"k": "v"})
    await foo()
    assert store.last_async_result == "async: {'k': 'v'}"

# --- Test 5: Full class decoration ---

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

# --- Test 6: Method name preservation ---

def test_method_name_preservation():
    """Test that original names are preserved as documented."""
    # Function should keep its original name
    assert fetch.__name__ == "fetch"
    
    # For class methods, let's create a simpler test
    class MyClass:
        @awaitable
        def my_method(self, x: int) -> int:
            return x * 2
        
        @my_method.acall
        async def my_method(self, x: int) -> int:
            return x + 2
    
    # Method name should be preserved
    assert MyClass.my_method.__name__ == "my_method"
    
    # The decorated instance should have the methods
    instance = MyClass()
    assert instance.my_method(5) == 10  # sync result from __call__

# --- Test 7: Inheritance (as mentioned in README) ---

@awaitable
class BaseClass:
    def __call__(self, value: str) -> str:
        return f"base: {value}"
    
    async def __acall__(self, value: str) -> str:
        return f"base async: {value}"

class DerivedClass(BaseClass):
    pass

def test_inheritance():
    """Test that inheritance works as documented."""
    derived = DerivedClass()
    
    # Inherited sync behavior
    assert derived("test") == "base: test"
    
    # Inherited async behavior
    async def test_async():
        result = derived("test")
        assert not isinstance(result, str)
        assert await result == "base async: test"
    
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

# --- Test 9: Method name accessibility via .sync and .__acall__ properties ---

@awaitable
def greet(name: str) -> str:
    return f"Hello {name}"

@greet.acall
async def greet(name: str) -> str:
    return f"Hello async {name}"

def test_direct_access():
    """Test that .sync and .__acall__ property returns the specific implementation."""
    # Direct access to sync function
    sync_fn = greet.sync
    assert sync_fn("World") == "Hello World"
    assert callable(sync_fn)

    # Direct access inside an sync def
    def test_sync_fn_on_sync():
        sync = greet.sync("World")
        a_sync = greet.__acall__("World")
        natural = greet("World")

        assert callable(sync) is False and sync == "Hello World"
        assert callable(a_sync) is True and asyncio.run(a_sync) == "Hello async World"
        assert callable(natural) is False and natural == "Hello World"

    # Direct access inside an async def
    async def test_sync_fn_on_async():
        sync = greet.sync("World")
        a_sync = greet.__acall__("World")
        natural = greet("World")

        assert callable(sync) is False and sync == "Hello World"
        assert callable(a_sync) is True and (await a_sync) == "Hello async World"
        assert callable(natural) is True and (await natural) == "Hello async World"

    assert callable

def test_acall_access():
    """Test that .__acall__ property returns the async implementation."""
    async_fn = greet.__acall__
    
    # It should be a coroutine function
    async def test_async_fn():
        result = async_fn("World")
        assert not isinstance(result, str) or await result == "Hello async World"
        return await result
    
    result = asyncio.run(test_async_fn())
    assert result == "Hello async World"

# --- Test 10: Multiple decorated functions in same module ---

@awaitable
def func_a(x: int) -> int:
    return x + 1

@awaitable
def func_b(x: int) -> int:
    return x * 2

@await_b.acall
async def func_b(x: int) -> int:
    return x * 3

async def test_multiple_decorated_functions():
    """Test that multiple decorated functions work independently."""
    # In async context
    assert await func_a(5) == 6  # sync version in async context -> async via default wrapper
    assert await func_b(5) == 15  # explicit async version

print("All tests ready to run")
