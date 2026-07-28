from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable, Coroutine, Awaitable
from typing import Any, TypeVar, Union, overload, Generic

TClass = TypeVar("TClass", bound=type)
T = TypeVar("T")

_ARE_ASYNC_FLAGS = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_IS_GENERATOR = inspect.CO_GENERATOR


def _is_async_context(frame):
    """Detect if `frame` (or the first non-generator frame above it)
    belongs to an `async def`.

    Generator frames (`CO_GENERATOR`) are skipped as its body runs
    as part of who drives it. Does not define its own context.
    """
    while frame is not None:
        if frame.f_code.co_flags & _IS_GENERATOR:
            # `await` is not possible on this frame, but may be on an upper one
            frame = frame.f_back
            continue
        # 1st non-generator frame determines the context as sync or async
        return bool(frame.f_code.co_flags & _ARE_ASYNC_FLAGS)
    return False


class _Awaitable_Function(Generic[T]):
    def __init__(self, fn: Callable[..., T]):
        # Default async as the wrapped sync
        async def __acall__(*args, **kwargs) -> T:
            return fn(*args, **kwargs)

        self._sync_func: Callable[..., T] = fn
        self._async_func: Callable[..., Coroutine[Any, Any, T]] = __acall__

    @property
    def sync(self) -> Callable[..., T]:
        return self._sync_func

    @property
    def __acall__(self) -> Callable[..., Coroutine[Any, Any, T]]:
        return self._async_func

    def __call__(self, *args, **kwargs) -> T | Coroutine[Any, Any, T]:
        if _is_async_context(sys._getframe().f_back):
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
    & stores in `__acallable_sync__` so a subclass can find the original via MRO.

    The async path looks up `__acall__` dynamically via MRO (`self.__acall__`),
    making subclasse overrides being respected automatically.
    """
    original_call: Callable = klass.__dict__["__call__"]
    klass.__acallable_sync__ = original_call

    @functools.wraps(original_call)
    def dispatcher(self, *args, **kwargs):
        if _is_async_context(sys._getframe().f_back):
            return self.__acall__(*args, **kwargs)
        else:
            return original_call(self, *args, **kwargs)

    klass.__call__ = dispatcher


def _as_awaitable_type(klass: TClass) -> TClass:
    """Decorated class `__call__` dispatches to `__acall__` if called from async.

    The class' original __call__ is saved as `__acallable_sync__`,
    an `__acall__` is created if not existing, and `__call__` is replaced
    by something that detect being called from a `def` or `async def`.

    `__init_subclass__` is also created or augmanted to make the subclasses
    keep the behavior about `__call__` and `__acall__` transparently.
    """
    # If the class or a parent was already decorated, reuse its stored original
    # sync callable instead of capturing our own dispatcher as the "original".
    original_call = getattr(klass, "__acallable_sync__", klass.__call__)
    klass.__acallable_sync__ = original_call

    original_acall = getattr(klass, "__acall__", None)
    if original_acall is None:
        # No user-provided __acall__: auto-generate one that wraps __call__
        @functools.wraps(original_call)
        async def __acall__(self, *args, **kwargs):
            return self.__acallable_sync__(*args, **kwargs)

        klass.__acall__ = __acall__

    def dispatcher(self, *args, **kwargs):
        if _is_async_context(sys._getframe().f_back):
            return self.__acall__(*args, **kwargs)
        else:
            return self.__acallable_sync__(*args, **kwargs)

    klass.__call__ = dispatcher

    original_init_subclass = klass.__dict__.get("__init_subclass__", None)

    if isinstance(original_init_subclass, classmethod):
        original_init_subclass = original_init_subclass.__func__

    @functools.wraps(original_init_subclass)
    def combined(cls, **kwargs):
        if original_init_subclass:
            # Preexisting __init_subclass__:
            original_init_subclass(cls, **kwargs)
        if "__call__" in cls.__dict__:
            _install_class_dispatcher(cls)

    klass.__init_subclass__ = classmethod(combined)

    return klass


@overload
def awaitable(obj: TClass) -> TClass: ...
@overload
def awaitable(obj: Callable[..., T]) -> _Awaitable_Function[T]: ...

def awaitable(obj):
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
