import sys
import inspect


def _is_async_context() -> bool:
    """
    Walk up the call stack to detect if caller is in an async context.
    Returns True if we encounter a CO_COROUTINE or CO_ASYNC_GENERATOR frame.
    """
    frame = sys._getframe()
    while frame is not None:
        # Check if this frame is a coroutine or async generator
        if frame.f_code.co_flags & (inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR):
            return True
        frame = frame.f_back
    return False


class _Awaitable:
    """
    Decorator implementation that:
    - Wraps a function or class
    - Provides sync/async dispatch based on caller context
    - Exposes .sync and .__acall__ properties for direct invocation
    - Uses .acall decorator to register async implementations
    """
    def __init__(self, func_or_class):
        # Store the original function or class
        self._func_or_class = func_or_class
        # Sync implementation - the original function or the class
        self._sync = func_or_class
        # Async implementation - starts as None, set via .acall
        self._acall = None
        
        # Check if decorating a class or function
        if isinstance(func_or_class, type):
            self._is_class = True
            # For classes, we need special handling
            self._original_class = func_or_class
            # Try to get the class's __acall__ method if it exists
            self._original_acall = getattr(func_or_class, '__acall__', None)
        else:
            self._is_class = False

    def __call__(self, *args, **kwargs):
        """
        Dispatch to sync or async implementation based on caller context.
        - If caller is in async context: return async result (coroutine)
        - If caller is in sync context: return sync result (direct value)
        """
        if self._is_class:
            # For classes, return an instance that has dual-dispatch behavior
            return self._create_dual_instance(*args, **kwargs)
        else:
            # For functions, dispatch directly
            if _is_async_context():
                # Caller is in async context - use async version
                if self._acall is not None:
                    return self._acall(*args, **kwargs)
                # No async implementation registered - return sync result
                # but wrapped so it can be awaited (preserves behavior)
                sync_result = self._sync(*args, **kwargs)
                async def _async_wrapper():
                    return sync_result
                return _async_wrapper()
            else:
                # Caller is in sync context - use sync version
                return self._sync(*args, **kwargs)

    def _create_dual_instance(self, *init_args, **init_kwargs):
        """
        Create a class instance that supports dual-dispatch __call__.
        """
        original_class = self._original_class
        original_acall = self._original_acall
        
        class _DualDispatchInstance(original_class):
            def __call__(self, *call_args, **call_kwargs):
                # Check if caller is in async context
                if _is_async_context():
                    # Async context - use async implementation
                    if original_acall is not None:
                        # Call async method with proper instance context (self as first arg)
                        return original_acall(self, *call_args, **call_kwargs)
                    raise TypeError("No async implementation available")
                else:
                    # Sync context - use sync implementation (inherited from original class)
                    return super().__call__(*call_args, **call_kwargs)
        
        # Create and return the instance
        return _DualDispatchInstance(*init_args, **init_kwargs)

    def register_acall(self, async_func):
        """
        Register an async implementation.
        Used when decorating a function with @fetch.acall pattern.
        """
        self._acall = async_func
        return async_func

    @property
    def acall(self):
        """
        Decorator for registering async implementations.
        Usage: @fetch.acall async def fetch_async(...)
        """
        if not hasattr(self, '_acall_decorator'):
            def _acall_decorator(async_func):
                self._acall = async_func
                return async_func
            self._acall_decorator = _acall_decorator
        return self._acall_decorator

    @property
    def sync(self):
        """
        Provide access to the synchronous implementation.
        For functions, returns the original function.
        For classes, returns a decorator that creates instances.
        """
        if self._is_class:
            # For classes, return a wrapper that creates instances
            def _sync_wrapper(*args, **kwargs):
                # This creates an instance that will use sync __call__ when called
                instance_class = self._create_dual_instance(*args, **kwargs)
                # We need to mark this instance so its __call__ uses sync path
                # One way is to set an attribute
                instance_class._ensure_sync = True
                return instance_class
            return _sync_wrapper
        else:
            # For functions, just return the sync function
            return self._func_or_class

    @property
    def __acall__(self):
        """
        Provide access to the asynchronous implementation.
        For functions, returns the async function.
        For classes, returns a wrapper that returns async-capable instances.
        """
        if self._is_class:
            # For classes, return a wrapper that creates instances 
            # whose __call__ uses async path when called in async context
            def _acall_wrapper(*args, **kwargs):
                # Create a class that always uses async __call__
                original_class = self._original_class
                original_acall = self._original_acall
                
                class _AlwaysAsyncInstance(original_class):
                    def __call__(self, *call_args, **call_kwargs):
                        # This instance always uses async implementation when called
                        if original_acall is not None:
                            return original_acall(self, *call_args, **call_kwargs)
                        raise TypeError("No async implementation available")
                
                return _AlwaysAsyncInstance(*args, **kwargs)
            return _acall_wrapper
        else:
            # For functions, return the async function
            return self._acall


# Public API
awaitable = _Awaitable