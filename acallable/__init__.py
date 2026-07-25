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


class _Awaitable:
    def __init__(self, obj):
        self._obj = obj
        if isinstance(obj, type):
            self._is_class = True
            self._original_class = obj
            self._original_acall = getattr(obj, "__acall__", None)
            self._original_class.__call__ = self._create_dispatcher(
                self._original_class
            )
            self._original_class.__init_subclass__ = self._create_init_subclass_hook()
        else:
            self._is_class = False
            self._func = obj
            self._acall_func = None

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

    def __call__(self, *args, **kwargs):
        if self._is_class:
            return self._original_class(*args, **kwargs)
        if _is_async_context():
            if self._acall_func is not None:
                return self._acall_func(*args, **kwargs)
            return self._func(*args, **kwargs)
        return self._func(*args, **kwargs)

    def register_acall(self, async_func):
        self._acall_func = async_func
        self._original_acall = async_func
        return async_func

    @property
    def acall(self):
        if not hasattr(self, "_acall_decorator"):

            def _acall_decorator(fn):
                self._acall_func = fn
                self._original_acall = fn
                return fn

            self._acall_decorator = _acall_decorator
        return self._acall_decorator

    @property
    def sync(self):
        if self._is_class:
            return self._original_class
        return self._func

    @property
    def __acall__(self):
        if self._is_class:
            return self._original_acall
        return self._acall_func

    def __getattr__(self, name):
        if self._is_class:
            return getattr(self._original_class, name)
        return getattr(self._func, name)


# Public decorator instance - need to make it callable to support @ syntax
def awaitable(obj):
    return _Awaitable(obj)