from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Coroutine
from typing import Callable


def _is_async_context():
    """Detect if current call place is asynchronous."""
    frame = sys._getframe()
    flags = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
    while frame is not None:
        if frame.f_code.co_flags & flags:
            return True
        frame = frame.f_back
    return False


class _Awaitable_Function:
    def __init__(self, fn: Callable):
        # Default async as the wrapped sync
        async def __acall__(*args, **kwargs):
            return fn(*args, **kwargs)

        self._sync_func: Callable = fn
        self._async_func: Callable = __acall__

    @property
    def sync(self):
        return self._sync_func

    @property
    def __acall__(self):
        return self._async_func

    def __call__(self, *args, **kwargs) -> Callable | Coroutine:
        if _is_async_context():
            return self.__acall__(*args, **kwargs)
        else:
            return self.sync(*args, **kwargs)

    def acall(self, fn: Callable):
        """Used like @property.set"""
        self._async_func = fn
        return self

    def __getattr__(self, name):
        return getattr(self._sync_func, name)

    def __get__(self, instance, owner):
        """Descriptor protocol: bind to instance when accessed as a method."""
        if instance is None:
            return self
        # Return a lightweight bound callable that pre-fills `instance`
        # as the first argument of both sync and async implementations.

        bound = _Awaitable_Function.__new__(_Awaitable_Function)
        bound._sync_func = functools.partial(self._sync_func, instance)
        bound._async_func = functools.partial(self._async_func, instance)
        return bound


def _install_class_dispatcher(klass: type) -> None:
    """Install a context-aware __call__ dispatcher on a class that defines its own __call__.

    The sync path captures the subclass's own `__call__` from its `__dict__`
    & stores in `__awaitable_sync__` so a subclass can find the original via MRO.

    The async path looks up `__acall__` dynamically via MRO (`self.__acall__`),
    making subclasse overrides being respected automatically.
    """
    original_call: Callable = klass.__dict__["__call__"]
    klass.__awaitable_sync__ = original_call

    @functools.wraps(original_call)
    def dispatcher(self, *args, **kwargs):
        if _is_async_context():
            return self.__acall__(*args, **kwargs)
        return original_call(self, *args, **kwargs)

    klass.__call__ = dispatcher


def _make_init_subclass_hook():
    """Apply an `__init_subclass__` hook wrapping subclass `__call__` when overridden."""

    def init_subclass_hook(cls, **kwargs):
        if "__call__" in cls.__dict__:
            _install_class_dispatcher(cls)

    return init_subclass_hook


def _as_awaitable_type(klass: type) -> type:
    # If the class or parent has already decorated, its true
    # the original __call__ is stored in __awaitable_sync__
    original_call = getattr(klass, "__awaitable_sync__", klass.__call__)

    original_acall = getattr(klass, "__acall__", None)
    if original_acall is None:
        # No __acall__ -> wrap __call__ in a coro
        @functools.wraps(original_call)
        async def __acall__(self, *args, **kwargs):
            return original_call(self, *args, **kwargs)

        new_acall = __acall__
    else:
        new_acall = original_acall


    # Build the namespace for the new class: copy everything from klass
    # but inject our __call__ dispatcher and __init_subclass__ in the body
    # so that inheritance works correctly (in Python 3.12+ __init_subclass__
    # must be defined in the class body, not set dynamically).
    namespace = dict(klass.__dict__)
    # Remove meta-attributes that type.__new__ handles internally
    for key in (
        "__dict__",
        "__weakref__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__annotations__",
        "__init_subclass__",  # we'll provide our own
    ):
        namespace.pop(key, None)

    # Store the original sync call so subclasses decorated with @awaitable
    # can find it through MRO instead of capturing a parent dispatcher.
    namespace["__awaitable_sync__"] = original_call

    # Context-aware __call__ dispatcher for the class body
    # Sync path: use captured original_call.
    # Async path: dynamic MRO lookup so subclasses that only override
    # __acall__ (without touching __call__) still get their version used.
    def dispatcher(self, *args, **kwargs):
        if _is_async_context():
            return self.__acall__(*args, **kwargs)
        return original_call(self, *args, **kwargs)

    namespace["__call__"] = dispatcher
    namespace["__acall__"] = new_acall

    # Define __init_subclass__ in the body so Python 3.12+ properly
    # dispatches it with the new subclass as the argument.
    namespace["__init_subclass__"] = _make_init_subclass_hook()

    # Recreate the class: same name, same bases, but with our body-defined hooks.
    # This preserves isinstance checks and __class__ identity because the
    # @awaitable decorator returns this new class and assigns it to the same name.
    new_klass = type(klass.__name__, klass.__bases__, namespace)

    return new_klass


def awaitable(obj) -> Callable:
    """Decorated callable dispatches __call__ or __acall__

    When decorated is called on sync context, uses __call__
    When decorated is called on async context, uses __acall__

    Appliable on classes:
    ```
    @awaitable
    class A:
        def __call__(self, ...):
            return 'called from some `def`

        async def __acall__(self, ...):
            return 'called from some `async def`
    ```

    Appliable on functions and methods:
    ```
    @awaitable
    def func(...):
        return 'called from some `def`

    @func.acall
    async def func(...):
        return 'called from some `async def`
    ```
    """
    if isinstance(obj, type):
        return _as_awaitable_type(obj)
    else:
        return _Awaitable_Function(obj)
