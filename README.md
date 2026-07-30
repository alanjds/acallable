# acallable

Write one named sync function with an alternative async version.
Call it from sync or async code. No name duplication.

```python
from acallable import acallable

@acallable
def fetch(url: str) -> str:
    print('synchronous version')
    import httpx
    r = httpx.get(url)
    return r.text

@fetch.acall
async def fetch(url: str) -> str:
    print('asynchronous version')
    import httpx
    async with httpx.AsyncClient() as client:
        r = await client.get(url)
    return r.text
```

Same name, both works:

```python
>>> def my_sync_main():
...    body = fetch("https://example.com")          # from sync context
...    ...

>>> my_sync_main()
synchonous version

>>> async def my_async_main():
...    body = await fetch("https://example.com")    # from async context
...    ...

>>> asyncio.run(my_async_main())
asynchronous version
```

When not provided, the async version is just the sync one wrapped in a coroutine. For free :)

Also works with classes (and methods btw):

```python
@acallable
class Fetcher:
    def __call__(self, url: str) -> str:          # sync body
        import httpx
        return httpx.get(url).text

    async def __acall__(self, url: str) -> str:   # async body
        import httpx
        async with httpx.AsyncClient() as client:
            return (await client.get(url)).text
```

Based on the [Jonny Saunders' idea][acall-proposal]
of having `__acall__` as the natural async version of `__call__`

## The problem it solves

Python's async coloring forces library authors to choose:
- Ship `get()` and `aget()`?
- Or ship `get()` and `get_async()`?
- Or create two classes `SyncVersion.get()` and `AsyncVersion.get()`?

Users then have to pick the right name depending on their own calling context, and
switch if that context changes. The two names diverge over time, doubling maintenance.

`acallable` lets you register one sync body and one async body under the same name.
The right one is **selected automatically based on the call site** being inside an
async context or not.

## Install

```
pip install acallable
```


## Where this idea comes from?

### The naming problem

The standard workaround for dual sync/async is two names: `get` & `aget`,
`send` & `send_async`, `fetch_sync` & `fetch`. Or two classes maybe extending from a single base.
Guido van Rossum described [one approach][guido] that adds a `.sync()` attribute alongside the async function.
That keeps one canonical implementation but still requires callers to choose
which name to use.

### The `__acall__` idea

Jonny Saunders [proposed][acall-proposal] an `__acall__` dunder method: an
async counterpart to `__call__` so that `await foo()` would dispatch to
`__acall__` if present, while `foo()` keeps dispatching to `__call__` as normal.
The pattern also maps cleanly like a property-setter decorator:

```python
class Runner:
    def process(self): ...

    @process.async
    async def process(self): ...
```

This example above does not work and uses the `async` reserved word,
but the idea is not bad. `acallable` implements this idea.

`@acallable` decorates the sync body; `@fn.acall` registers the async body.
No new syntax and no language changes.

### When `__acall__` vs. `__call__`? When "called from an awaitable context"!

The original `__acall__` proposed to dispatch based on the question
"is this call expression being awaited right now?":
- `foo()` -> `foo.__call__()` (as today)
- `await foo()` -> `foo.__acall__()` (new)

However it lets open [the question][acall-question]: what is `x` in this example?

```python
async def bar():
    x = foo()
    await x
```

As `await x` should be `x.__await__`, detecting `x = foo.__acall__()` is not possible,
breaking expectations.

To solve it, `acallable` shifts the `__acall__` choosing question to:

**"Is the call site inside a frame where `await` is legal?"**

With this definition, `x` surely is a coroutine returned by `foo.__acall__()`,
because `foo()` is called _inside an `async def`_.

### Implementation details

> **Is the call site inside a frame where `await` is legal?**

That new definition also solves the following example:

```python
async def compute_many(numbers):
    results = await asyncio.gather(*(compute(i) for i in numbers))
    ...
```

Should `compute(i)` be `compute.__call__(i)` or `compute.__acall__(i)`?

Here `compute(i)` is called from inside a generator expression. The immediate
frame is a `CO_GENERATOR`, where `await` is not legal. The _immediate_ context is sync, then
`asyncio.gather` would receive plain values instead of coroutines.
However it whould not be correct, as "the call site **IS** inside a frame where `await` is legal".

`acallable` walks up the call stack transparently past any `CO_GENERATOR` frames
(sync genexprs and plain generators) until it finds some sync or async enclosing context.
`CO_ASYNC_GENERATOR` frames (`async def ... yield`) are considered async contexts.
That let genexpr-inside-gather patterns working correctly.

The frame-inspection overhead using `sys._getframe` and `co_flags` was
[reported][frame-detection] in the Python discussion forums as costing less than 1.5µs;

#### Should _all_ `CO_GENERATOR` be transparent?

This is debatable because standard `def gen(): ... yield <something>` is _also_ `CO_GENERATOR`.

The question goes down to one of this confusing solutions:
1. Two rules for generators: `<genexpr>` is transparent; `def..yield` is sync
1. Two rules for sync detection: `def` is sync; EXCEPT `def..yield` that is transparent

_FOR NOW_ this lib considers ALL generators transparent (Option 2).
A future change on this _could_ occur,
but surely versioned as a breaking change and documented.

## Usage

### Decorating functions

```python
from acallable import acallable

@acallable
def greet(name: str) -> str:
    return f"Hello, {name}"          # sync body: runs from plain def

@greet.acall
async def greet(name: str) -> str:
    await asyncio.sleep(0)           # async body: runs when awaited
    return f"Hello, {name}"

# From sync caller, like `def _():`
print(greet("world"))                # Hello, world

# From async caller, like `async def _():`
print(await greet("world"))          # Hello, world
```

### Decorating a Class method

```python
from acallable import acallable

class DataStore:
    def __init__(db, ...):
        ...

    @acallable
    def save(self, record: dict) -> None:
        db.execute_sync(record)

    @save.acall
    async def save(self, record: dict) -> None:
        await db.execute_async(record)

store = DataStore()

def foo(): store.save({"k": "v"})               # sync
async def foo(): async store.save({"k": "v"})   # async
```

### Decorating a whole Class

Define `__call__` (sync) and `__acall__` (async) directly on the class body.
`@acallable` wires the dispatcher automatically:

```python
from acallable import acallable

@acallable
class Fetcher:
    def __call__(self, url: str) -> str:          # sync body
        import httpx
        return httpx.get(url).text

    async def __acall__(self, url: str) -> str:   # async body
        import httpx
        async with httpx.AsyncClient() as client:
            return (await client.get(url)).text

fetcher = Fetcher()

def sync_main():
    body = fetcher("https://example.com")         # calls __call__

async def async_main():
    body = await fetcher("https://example.com")   # calls __acall__
```

The decorated inheritance is not touched: `isinstance(fetcher, Fetcher)` works,
subclassing works, type checkers are kept _mostly_ happy.

### Default async body

If you omit `@fn.acall` or `Class.__acall__`,
the async path automatically wraps the sync body in a coroutine.
Simple functions that have no real async I/O _yet_ get both paths for free:

```python
@acallable
def compute(x: int) -> int:
    return x * 2

def foo(): return compute(3)              # returns 6
async def foo(): return await compute(3)  # also returns 6
```

This way you can create APIs today that will have real async I/O in the future.
No need to break the contracts later.

### Static typing caveats

Static typers are currently (2026) not following the decoration code "magic",
leading to detecting false errors like `__await__ may be missing`. I tried my best but
could not "fix" these by luring typecheckers into accepting that
a class `__call__` can return an `int | Awaitable[int]` for example.

My current advise is to lower the typechecker issue level to "warning" or to "ignore"
for now for the await-related kind of checks below. If this lib gets traction,
they someday may provide a better option. Until that:

```toml
# pyproject.toml
...

[tool.ty.rules]
invalid-await = "warn"  # or "ignore"

[tool.pyright]
reportGeneralTypeIssues = "warning"  # or "none"

[tool.mypy.overrides]
disable_error_code = ["operator"]  # Sorry, no way to force "warning" here.

[tool.ruff]
lint.extend-ignore = [
    'N807',  # Allow functions named __acall__ to start with `__`
]
```


### Implementation inspiration & acknowledgements

[zyncio][zyncio] (by Benjy Wiener) solves the same problem from the opposite direction: write
everything as a coroutine and branch inside the function if `zyncio.is_sync()`.

`acallable` keeps the two bodies separate (one sync fn, one async fn) and let the decorator
handle detection and dispatch. The call site needs no mixin inheritance or subclass split.

I am thankful for Benjy Wiener for the inspiration and implementation ideas.
`zyncio` is the best working approximation from `__acall__` that I could find.
I just found the DX worth some improvement that would not be possible
with a direct contribution without a huge API breaking change.

Thanks also Ricardo Robles for [exploring][frame-detection] the `sys._getframe` + `co_flags`
detection approach.

And Serhiy Storchaka for [the important corner question][what-is-x] about `x = foo(); await x`

Hope to had not missed many important contributions, besides GvH of course :)

## Links

- Python discussion: [How can async support dispatch between sync and async variants?][thread]
- `__acall__` proposal: [posts #20][acall-proposal] and [#22][acall-detail]
- Frame-inspection detection: [post #3 in the overload proposal thread][frame-detection]
- Related project: [zyncio][zyncio], the coroutine-first approach

[guido]: https://discuss.python.org/t/how-can-async-support-dispatch-between-sync-and-async-variants-of-the-same-code/15014/8
[acall-proposal]: https://discuss.python.org/t/how-can-async-support-dispatch-between-sync-and-async-variants-of-the-same-code/15014/20
[acall-detail]: https://discuss.python.org/t/how-can-async-support-dispatch-between-sync-and-async-variants-of-the-same-code/15014/22
[acall-question]: https://discuss.python.org/t/how-can-async-support-dispatch-between-sync-and-async-variants-of-the-same-code/15014/21
[zyncio]: https://pypi.org/project/zyncio/
[frame-detection]: https://discuss.python.org/t/proposal-for-a-new-way-to-overload-methods-by-arguments-and-whether-it-is-synchronous-or-asynchronous/104902/3
[what-is-x]: https://discuss.python.org/t/proposal-for-a-new-way-to-overload-methods-by-arguments-and-whether-it-is-synchronous-or-asynchronous/104902/6
[thread]: https://discuss.python.org/t/how-can-async-support-dispatch-between-sync-and-async-variants-of-the-same-code/15014


## License

This package is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
and can undestand more at http://choosealicense.com/licenses/apache/ on the
sidebar notes.

Apache Licence v2.0 is a MIT-like licence. This means, in plain English:
- It's truly open source
- You can use it as you wish, for money or not
- You can sublicence it (change the licence!!)
- This way, you can even use it on your closed-source project
As long as:
- You cannot use the authors name, logos, etc, to endorse a project
- You keep the authors copyright notices where this code got used, even on your closed-source project
(come on, even Microsoft kept BSD notices on Windows about its TCP/IP stack :P)
