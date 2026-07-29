"""
Stress tests for the async context detection via frame introspection.

When @acallable is called from a sync def, it must return the
plain sync value.  When called from an async def, it must return
a coroutine (the __acall__ result).
"""

import asyncio

import pytest

from acallable import acallable

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@acallable
def _identity(x: str) -> str:
    """A simple acallable-wrapped function that just echoes."""
    return f'sync:{x}'


@_identity.acall
async def _identity(x: str) -> str:
    return f'async:{x}'


def _is_coro(obj) -> bool:
    """Return True if *obj* is a coroutine object (not a coroutine function)."""
    return asyncio.iscoroutine(obj)


# ===================================================================
# 1 -- basic sync / async alternations
# ===================================================================

def test_decorated_from_sync():
    assert _identity('a') == 'sync:a'


@pytest.mark.asyncio
async def test_decorated_from_async():
    assert await _identity('a') == 'async:a'


# ===================================================================
# 2 -- sync -> sync -> decorated
# ===================================================================

def test_sync_calls_sync_decorated():
    def wrapper():
        return _identity('b')
    assert wrapper() == 'sync:b'


def test_sync_calls_sync_calls_decorated():
    def inner():
        return _identity('c')

    def outer():
        return inner()

    assert outer() == 'sync:c'


# ===================================================================
# 3 -- async -> sync -> decorated
# ===================================================================

@pytest.mark.asyncio
async def test_async_calls_sync_wrapper():
    """Async def calls a sync def that calls the decorated function."""
    def sync_wrapper():
        return _identity('d')

    result = sync_wrapper()
    assert result == 'sync:d'


# ===================================================================
# 4 -- sync calls async that calls decorated
# ===================================================================

def test_sync_calls_async_def():
    """Sync context calls an async function that calls the decorated one."""
    async def async_inner():
        return _identity('e')

    coro = async_inner()
    # async_inner returns the coroutine from _identity (because
    # _identity detects async context from async_inner's frame).
    result = asyncio.run(coro)
    assert _is_coro(result)
    # Can't use await here (sync function), so use asyncio.run
    assert asyncio.run(result) == 'async:e'


# ===================================================================
# 5 -- nested sync -> async -> sync -> decorated
# ===================================================================

def test_sync_async_sync_decorated():
    """sync -> (calls async which calls sync) -> decorated"""

    async def async_mid():
        def sync_wrapper():
            return _identity('f')
        return sync_wrapper()

    coro = async_mid()
    result = asyncio.run(coro)
    assert result == 'sync:f'


# ===================================================================
# 6 -- nested async -> sync -> async -> sync -> decorated
# ===================================================================

@pytest.mark.asyncio
async def test_async_sync_async_sync_decorated():
    """async -> (sync -> (async -> (sync -> decorated)))"""
    def sync_mid():
        async def async_inner():
            def sync_bottom():
                return _identity('g')
            return sync_bottom()
        return async_inner()

    coro = sync_mid()
    # async_inner() runs, calls sync_bottom() which returns "sync:g"
    result = await coro
    assert result == 'sync:g'


# ===================================================================
# 7 -- async generator context detection
# ===================================================================

@pytest.mark.asyncio
async def test_async_gen_detected():
    """Calling the decorated function inside an async generator."""

    async def async_gen():
        yield _identity('h')
        yield _identity('i')

    gen = async_gen()
    results = [await item async for item in gen]
    assert results == ['async:h', 'async:i']


# ===================================================================
# 8 -- lambda and comprehension frames
# ===================================================================

def test_lambda_in_sync():
    fn = lambda: _identity('j')
    assert fn() == 'sync:j'


@pytest.mark.asyncio
async def test_lambda_in_async():
    """Lambda is a sync callable — the immediate caller is sync → sync result."""
    fn = lambda: _identity('k')
    result = fn()
    assert result == 'sync:k'


def test_list_comp_in_sync():
    assert [_identity(x) for x in ('l',)] == ['sync:l']


@pytest.mark.asyncio
async def test_list_comp_in_async():
    results = [_identity(x) for x in ('m',)]
    assert len(results) == 1
    assert _is_coro(results[0])
    assert await results[0] == 'async:m'


# ===================================================================
# 9 -- sync callback invoked from async wrapper
# ===================================================================

@pytest.mark.asyncio
async def test_callback_in_async():
    """A sync callback (accepts a callable) invoked inside async."""
    def run_callback(cb):
        return cb()

    result = run_callback(lambda: _identity('n'))
    assert result == 'sync:n'


# ===================================================================
# 10 -- deeply nested (5+ levels) alternating context
# ===================================================================

def test_deep_sync_async_sync_async_sync():
    """
    sync -> async -> sync -> async -> sync -> decorated.
    5 levels of alternating contexts.

    The immediate caller is level5 (sync) → plain sync value.
    """

    def level1():
        async def level2():
            def level3():
                async def level4():
                    def level5():
                        return _identity('o')
                    return level5()
                return level4()
            return level3()
        return level2()

    c1 = level1()  # level2() coroutine
    c2 = asyncio.run(c1)  # level3 returns level4 coroutine
    c3 = asyncio.run(c2)  # level4 runs level5 → returns "sync:o"
    assert c3 == 'sync:o'


@pytest.mark.asyncio
async def test_deep_async_sync_async_sync_async():
    """
    async -> sync -> async -> sync -> async -> decorated.
    5 levels of alternating contexts.
    """

    async def level1():
        def level2():
            async def level3():
                def level4():
                    async def level5():
                        return _identity('p')
                    return level5()
                return level4()
            return level3()
        return level2()

    c1 = await level1()  # level2 returns level3 coroutine
    c2 = await c1       # level3 returns level4's result
    c3 = await c2       # level4 returns level5 coroutine -> identity coro
    assert _is_coro(c3)
    assert await c3 == 'async:p'


# ===================================================================
# 11 -- sync generator yields decorated (detection is NOT async here)
# ===================================================================

def test_generator_yields_decorated():
    """Sync generator itself is NOT async -> detection says sync."""

    def gen():
        yield _identity('q')

    results = list(gen())
    assert results == ['sync:q']


@pytest.mark.asyncio
async def test_async_gen_with_sync_part():
    """Async generator calls a sync function that calls the decorated one."""

    async def agen():
        def sync_sub():
            return _identity('r')
        yield sync_sub()

    values = [item async for item in agen()]
    assert len(values) == 1
    assert values[0] == 'sync:r'


# ===================================================================
# 12 -- method injection: class with @acallable methods called cross-context
# ===================================================================

class Fixture:
    @acallable
    def method(self, val: str) -> str:
        return f'sync-method:{val}'

    @method.acall
    async def method(self, val: str) -> str:
        return f'async-method:{val}'


def test_class_method_sync():
    obj = Fixture()
    assert obj.method('s') == 'sync-method:s'


@pytest.mark.asyncio
async def test_class_method_async():
    obj = Fixture()
    assert await obj.method('a') == 'async-method:a'


def test_class_method_sync_wrapper_in_async():
    """Sync function inside async def calling a method."""
    async def async_fn():
        obj = Fixture()
        return obj.method('t')

    coro = async_fn()
    result = asyncio.run(coro)
    # method() sees the async context from async_fn
    assert asyncio.iscoroutine(result)
    assert asyncio.run(result) == 'async-method:t'


# ===================================================================
# 13 -- decorated class with cross-context calls
# ===================================================================

@acallable
class CallableClass:
    def __call__(self, val: str) -> str:
        return f'sync-class:{val}'

    async def __acall__(self, val: str) -> str:
        return f'async-class:{val}'


def test_callable_class_sync():
    obj = CallableClass()
    assert obj('u') == 'sync-class:u'


@pytest.mark.asyncio
async def test_callable_class_async():
    obj = CallableClass()
    assert await obj('v') == 'async-class:v'


def test_callable_class_async_via_sync_wrapper():
    """Sync wrapper inside async def → immediate caller is sync → sync result."""
    async def async_fn():
        obj = CallableClass()

        def sync_wrapper():
            return obj('w')

        return sync_wrapper()

    coro = async_fn()
    result = asyncio.run(coro)
    assert result == 'sync-class:w'


# ===================================================================
# 14 -- sync-only deep nesting (no async frame anywhere)
# ===================================================================

def test_sync_deeply_nested_no_leak():
    """Many sync frames with no async frames -> must be sync."""

    def f1():
        def f2():
            def f3():
                def f4():
                    return _identity('x')
                return f4()
            return f3()
        return f2()

    assert f1() == 'sync:x'


# ===================================================================
# 15 -- functools.partial in async context
# ===================================================================

def test_partial_in_async():
    """Async context with functools.partial wrapping the decorated callable."""

    from functools import partial

    async def inner():
        p = partial(_identity, 'y')
        return p()

    coro = inner()
    result = asyncio.run(coro)
    assert _is_coro(result)
    assert asyncio.run(result) == 'async:y'


# ===================================================================
# 16 -- context manager (sync) inside async
# ===================================================================

@pytest.mark.asyncio
async def test_context_manager_in_async():
    """Sync context manager's __enter__ called from async context."""

    class SyncCM:
        def __enter__(self_):
            return _identity('z')

        def __exit__(self_, *a):
            return

    with SyncCM() as result:
        assert result == 'sync:z'


# ===================================================================
# 17 -- sync @property called from async
# ===================================================================

@pytest.mark.asyncio
async def test_property_in_async():
    """A sync @property that calls the decorated function inside async."""

    class Obj:
        @property
        def val(self_):
            return _identity('prop')

    obj = Obj()
    result = obj.val
    assert result == 'sync:prop'


# ===================================================================
# 18 -- deep recursion stress tests
# ===================================================================

def test_deep_recursion_sync():
    """Deep recursion (sync only) should always return sync."""

    def recurse(n):
        if n <= 0:
            return _identity('deep')
        return recurse(n - 1)

    assert recurse(50) == 'sync:deep'


@pytest.mark.asyncio
async def test_deep_recursion_async():
    """Deep recursion with async frame at the top."""

    async def recurse(n):
        if n <= 0:
            return await _identity('deep')
        return await recurse(n - 1)

    result = await recurse(50)
    assert result == 'async:deep'


# ===================================================================
# 19 -- Mixed sync/async call patterns in one test
# ===================================================================

def test_mixed_call_patterns():
    """Alternate between sync and async calls in a single test run."""
    # sync
    assert _identity('s1') == 'sync:s1'
    assert _identity('s2') == 'sync:s2'

    async def mixed():
        # async
        r1 = await _identity('a1')
        assert r1 == 'async:a1'

        # sync wrapper inside async
        def sync_wrap():
            return _identity('a2')

        r2 = sync_wrap()
        assert r2 == 'sync:a2'

        return 'ok'

    assert asyncio.run(mixed()) == 'ok'

    # back to sync
    assert _identity('s3') == 'sync:s3'


# ===================================================================
# 20 -- generator inside async def yields coroutines
# ===================================================================


@pytest.mark.asyncio
async def test_generator_in_async_yields_coroutine():
    """A sync generator defined and driven inside an async def.

    Generator frames (CO_GENERATOR) are transparent — they are
    suspension points whose body runs as part of whoever drives
    them.  Since the driver is an async def, the decorated
    function inside the generator returns a coroutine.
    """

    def gen():
        yield _identity('u1')
        yield _identity('u2')

    g = gen()
    r1 = next(g)
    r2 = next(g)

    assert _is_coro(r1), 'generator inside async def yields coroutines'
    assert _is_coro(r2)
    assert await r1 == 'async:u1'
    assert await r2 == 'async:u2'


# ===================================================================
# 21 -- asyncio.gather with generator pattern
# ===================================================================


@pytest.mark.asyncio
async def test_generator_asyncio_gather_pattern():
    """asyncio.gather(*gen()) — the driving use case for generator transparency.

    A generator inside an async def should yield coroutines that
    ``asyncio.gather`` can consume.
    """

    def gen():
        yield _identity('g1')
        yield _identity('g2')
        yield _identity('g3')

    results = await asyncio.gather(*gen())
    assert results == ['async:g1', 'async:g2', 'async:g3']


# ===================================================================
# 22 -- async with context manager
# ===================================================================


@pytest.mark.asyncio
async def test_async_with_aenter():
    """async with calls __aenter__ (async def) — the decorated func
    returns a coroutine, which __aenter__ must await."""

    class AsyncCM:
        async def __aenter__(self_):
            return await _identity('cm')

        async def __aexit__(self_, *a):
            pass

    async with AsyncCM() as result:
        assert result == 'async:cm'


# ===================================================================
# 23 -- generator driven from sync stays sync
# ===================================================================


def test_generator_driven_from_sync_stays_sync():
    """Same generator pattern called from sync context stays sync.

    The generator's f_back points to a sync frame — no async frames
    above → sync value.
    """

    def gen():
        yield _identity('v1')
        yield _identity('v2')

    results = list(gen())
    assert results == ['sync:v1', 'sync:v2']


# ===================================================================
# 24 -- nested generators (gen -> gen) inside async
# ===================================================================


@pytest.mark.asyncio
async def test_nested_generators_skipped():
    """Multiple levels of generator frames are all transparent."""

    def outer_gen():
        def inner_gen():
            yield _identity('nested')

        yield from inner_gen()

    g = outer_gen()
    r = next(g)
    assert _is_coro(r)
    assert await r == 'async:nested'


# ===================================================================
# 25 -- generator inside sync def stays sync (even if called from async)
# ===================================================================


@pytest.mark.asyncio
async def test_generator_inside_sync_def_stays_sync():
    """A generator defined inside a sync def, driven from async.

    The generator's f_back points to the sync def that drives it,
    not to the outer async def.
    """

    def sync_maker():
        def gen():
            yield _identity('sync-maker')

        return list(gen())

    # Even when called from async, sync_maker drives the generator
    # from its own sync frame.
    results = sync_maker()
    assert results == ['sync:sync-maker']
