from __future__ import annotations

__version__ = '0.1.0'

import functools
import inspect
import sys
from collections.abc import Awaitable, Callable
from typing import overload

_ARE_ASYNC_FLAGS = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_IS_GENERATOR = inspect.CO_GENERATOR


def _is_async_context(frame):
    """Detect if `frame` (or the first non-generator frame above it)
    belongs to an `async def`.

    Generator frames (`CO_GENERATOR`) are skipped as its body runs
    as part of who drives it. Considered as not defining its own context.
    """
    # It would be POSSIBLE to distinguish genexpr from def-yield generators
    # by inspecting `frame.f_code.co_name == '<genexpr>'` or .co_qualname
    # This would let me keep `x = (my_acallable(i) for i in range(10))` as transparent
    # and keep this other generator as sync context:
    # ```
    #     def mygen(): for i in range(10):
    #         for i in range(10):
    #             yield my_acallable(i)
    # ```
    # However what is more confusing?
    # - Two rules for generators: <genexpr> is transparent; `def..yield` is sync
    # - Two rules for sync detection: `def` is sync; EXCEPT `def..yield` is transparent
    # For now, I will reject the first and keep the second option.
    # A change here _WILL BE_ a breaking change,
    # but lets see what people think is reasonable and what is nonsense.
    while frame is not None:
        if frame.f_code.co_flags & _IS_GENERATOR:
            # `await` is not possible on this frame, but may be on an upper one
            frame = frame.f_back
            continue
        # 1st non-generator frame determines the context as sync or async
        return bool(frame.f_code.co_flags & _ARE_ASYNC_FLAGS)
    return False


class Acallable[T](Callable):
    """
    Functions and methods decorated with `@acallable` are `Acallable`s

    They dispatches async or sync based on where it was called:
    When called on some sync context, uses `self.sync`
    When called on some async context, uses `self.__acall__`
    """
    _sync_func: Callable[..., T] = None  # ty:ignore[invalid-assignment]
    _async_func: Callable[..., Awaitable[T]] = None  # ty:ignore[invalid-assignment]

    def __init__(self, fn: Callable[..., T]):
        # Default async as the wrapped sync
        async def __acall__(*args, **kwargs) -> T:
            return fn(*args, **kwargs)

        if fn is None:
            raise TypeError()

        self._sync_func = fn
        self._async_func = __acall__

    @property
    def sync(self) -> Callable[..., T]:
        """Always the synchronous (`def`) version of this Callable"""
        return self._sync_func

    @property
    def __acall__(self) -> Callable[..., Awaitable[T]]:
        """Always the asynchronous (`async def`) version of this Callable"""
        return self._async_func

    def __call__(self, *args, **kwargs) ->  T | Awaitable[T]:
        """Dispatches async or sync based on where it was called

        When called on some sync context, uses `self.sync`
        When called on some async context, uses `self.__acall__`
        """
        if _is_async_context(sys._getframe().f_back):
            return self.__acall__(*args, **kwargs)
        else:
            return self.sync(*args, **kwargs)

    def acall(self, fn: Callable[..., Awaitable[T]]) -> Acallable[T]:
        """Sets the __acall__ of this function. Used like `@property.set`::

            @acallable
            def func(...):
                return 'called from some `def`'

            @func.acall
            async def func(...):
                return 'called from some `async def`'
        """
        self._async_func = fn
        return self

    def __getattr__(self, name):
        return getattr(self._sync_func, name)

    def __setattr__(self, name, value):
        try:
            object.__getattribute__(self, name)
            instance_local = True
        except AttributeError:
            instance_local = False

        if instance_local:
            super().__setattr__(name, value)
        else:
            setattr(self._sync_func, name, value)

    def __get__(self, instance, owner):
        """Descriptor protocol: bind to instance when accessed as a method."""
        if instance is None:
            return self
        # Return a lightweight bound callable that pre-fills `instance`
        # as the first argument of both sync and async implementations.

        # Let typecheckers happy
        assert self._sync_func is not None
        assert self._async_func is not None

        bound = Acallable.__new__(Acallable)
        bound._sync_func = functools.partial(self._sync_func, instance)
        bound._async_func = functools.partial(self._async_func, instance)
        return bound


def _install_class_dispatcher(klass: Callable) -> None:
    """Install a context-aware __call__ dispatcher on a class that defines its own __call__.

    The sync path captures the subclass's own `__call__` from its `__dict__`
    & stores in `__acallable_sync__` so a subclass can find the original via MRO.

    The async path looks up `__acall__` dynamically via MRO (`self.__acall__`),
    making subclasse overrides being respected automatically.
    """
    original_call: Callable = klass.__dict__['__call__']
    klass.__acallable_sync__ = original_call  # ty:ignore[unresolved-attribute]

    @functools.wraps(original_call)
    def dispatcher(self, *args, **kwargs):
        if _is_async_context(sys._getframe().f_back):
            return self.__acall__(*args, **kwargs)
        else:
            return self.__acallable_sync__(*args, **kwargs)

    klass.__call__ = dispatcher  # ty:ignore[unresolved-attribute]


def _as_acallable_type[T: type](klass: T) -> T:
    """Decorated class `__call__` dispatches to `__acall__` if called from async.

    The class' original __call__ is saved as `__acallable_sync__`,
    an `__acall__` is created if not existing, and `__call__` is replaced
    by something that detect being called from a `def` or `async def`.

    `__init_subclass__` is also created or augmanted to make the subclasses
    keep the behavior about `__call__` and `__acall__` transparently.
    """
    # If the class or a parent was already decorated, reuse its stored original
    # sync callable instead of capturing our own dispatcher as the "original".
    original_call = getattr(klass, '__acallable_sync__', klass.__call__)
    klass.__acallable_sync__ = original_call  # ty:ignore[unresolved-attribute]

    original_acall = getattr(klass, '__acall__', None)
    if original_acall is None:
        # No user-provided __acall__: auto-generate one that wraps __call__
        @functools.wraps(original_call)
        async def __acall__(self, *args, **kwargs):
            return self.__acallable_sync__(*args, **kwargs)

        klass.__acall__ = __acall__  # ty:ignore[unresolved-attribute]

    def dispatcher(self, *args, **kwargs):
        if _is_async_context(sys._getframe().f_back):
            return self.__acall__(*args, **kwargs)
        else:
            return self.__acallable_sync__(*args, **kwargs)

    klass.__call__ = dispatcher  # ty:ignore[invalid-assignment]

    original_init_subclass = klass.__dict__.get('__init_subclass__', None)

    if isinstance(original_init_subclass, classmethod):
        original_init_subclass = original_init_subclass.__func__

    @functools.wraps(original_init_subclass)
    def combined(cls, **kwargs):
        if original_init_subclass:
            # Preexisting __init_subclass__:
            original_init_subclass(cls, **kwargs)
        if '__call__' in cls.__dict__:
            _install_class_dispatcher(cls)

    klass.__init_subclass__ = classmethod(combined)  # ty:ignore[invalid-assignment]

    return klass

@overload
def acallable[T: type](obj: T) -> T: ...
@overload
def acallable[T](obj: Callable[..., T]) -> Acallable[T]: ...

def acallable(obj):
    """Decorated callable dispatches `__call__` or `__acall__`

    When decorated is called on sync context, uses `__call__`
    When decorated is called on async context, uses `__acall__`

    Appliable on classes:
    ```
    @acallable
    class A:
        def __call__(self, ...):
            return 'called from some `def`'

        async def __acall__(self, ...):
            return 'called from some `async def`'
    ```

    Appliable on functions and methods:
    ```
    @acallable
    def func(...):
        return 'called from some `def`'

    @func.acall
    async def func(...):
        return 'called from some `async def`'
    ```
    """
    if isinstance(obj, type):
        return _as_acallable_type(obj)
    else:
        return Acallable(obj)
