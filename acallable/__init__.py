from typing import Callable
import sys
import inspect


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


def _as_awaitable_type(klass: type) -> type:
    raise NotImplementedError()

    def _create_dispatcher(self, cls):
        original_call = cls.__call__ if hasattr(cls, "__call__") else None
        async_impl = getattr(cls, "__acall__", None)

        def dispatcher(self, *args, **kwargs):
            if _is_async_context():
                if async_impl is not None:
                    return async_impl(self, *args, **kwargs)
                elif hasattr(self, "__acall__"):
                    return self.__acall__(self, *args, **kwargs)
                raise TypeError("No async implementation available")
            return original_call(self, *args, **kwargs)

        return dispatcher

    def _create_init_subclass_hook(self):
        original_init_subclass = getattr(
            self._original_class, "__init_subclass__", None
        )
        # Capture self to use in the hook
        awaitable_self = self

        def init_subclass_hook(subclass, **kwargs):
            if original_init_subclass is not None:
                original_init_subclass(subclass, **kwargs)
            subclass.__call__ = awaitable_self._create_dispatcher(subclass)

        return init_subclass_hook


# Public decorator instance - need to make it callable to support @ syntax
def awaitable(obj):
    if isinstance(obj, type):
        return _as_awaitable_type(obj)
    else:
        return _Awaitable_Function(obj)
