from typing import Callable
import sys
import inspect
import functools


def _is_async_context():
    """Detect if current call stack is asynchronous."""
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

    def __call__(self, *args, **kwargs):
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
    """Install a context-aware __call__ dispatcher on a class that defines its own __call__."""
    # Grab the subclass's own __call__ from its __dict__ (not via MRO)
    original_call = klass.__dict__["__call__"]
    # Use MRO for __acall__ — picks up parent's if not overridden in this class
    original_acall = getattr(klass, "__acall__", None)

    def dispatcher(self, *args, **kwargs):
        if _is_async_context():
            if inspect.iscoroutinefunction(original_acall):
                return original_acall(self, *args, **kwargs)
            else:
                # Wrap sync __acall__ in a coroutine
                async def async_wrapper():
                    return original_acall(self, *args, **kwargs)

                return async_wrapper()
        return original_call(self, *args, **kwargs)

    klass.__call__ = dispatcher


def _make_init_subclass_hook(original_call, original_acall):
    """Create an __init_subclass__ hook that wraps subclass __call__ when overridden."""

    def init_subclass_hook(cls, **kwargs):
        # Only re-wrap when the subclass defines its own __call__
        if "__call__" in cls.__dict__:
            _install_class_dispatcher(cls)

    return init_subclass_hook


def _as_awaitable_type(klass: type) -> type:
    # Capture original __call__ and __acall__
    original_call = klass.__call__

    # Get or create a default __acall__
    original_acall = getattr(klass, "__acall__", None)
    if original_acall is None:

        async def _default_acall(self, *args, **kwargs):
            return original_call(self, *args, **kwargs)

        original_acall = _default_acall

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

    # Context-aware __call__ dispatcher for the class body
    def dispatcher(self, *args, **kwargs):
        if _is_async_context():
            return original_acall(self, *args, **kwargs)
        else:
            return original_call(self, *args, **kwargs)

    namespace["__call__"] = dispatcher
    namespace["__acall__"] = original_acall

    # Define __init_subclass__ in the body so Python 3.12+ properly
    # dispatches it with the new subclass as the argument.
    init_subclass_hook = _make_init_subclass_hook(original_call, original_acall)
    namespace["__init_subclass__"] = init_subclass_hook

    # Recreate the class: same name, same bases, but with our body-defined hooks.
    # This preserves isinstance checks and __class__ identity because the
    # @awaitable decorator returns this new class and assigns it to the same name.
    new_klass = type(klass.__name__, klass.__bases__, namespace)

    return new_klass


# Public decorator instance - need to make it callable to support @ syntax
def awaitable(obj):
    if isinstance(obj, type):
        return _as_awaitable_type(obj)
    else:
        return _Awaitable_Function(obj)
