"""Killable model isolation: a long-lived spawned worker per stateful adapter.

Architecture decision 3 of the adapters plan: no model construction, decode,
generation, or synthesis runs on the event loop. Each offline adapter owns one
:class:`ModelWorker` — a ``multiprocessing`` child created with the
cross-platform ``spawn`` context. The model is constructed **inside** the
child by an importable factory; only serializable requests/results cross the
bounded IPC queues.

Cancellation is cooperative first (a shared event the handler polls at safe
points, raising :class:`WorkerCancelledError`), but a worker that does not
acknowledge shutdown within the grace budget is terminated, joined, and killed
as a final fallback — cancelling an asyncio future is never considered
sufficient shutdown.

The child never handles terminal SIGINT itself (the parent owns cancellation
and teardown), so Ctrl-C in the CLI cannot orphan half-dead children.

Handler contract::

    def factory(**kwargs) -> Handler          # importable module attribute
    def handler(payload, cancel_event) -> Any # runs serially in the child

The handler should call :func:`raise_if_cancelled` at natural boundaries
(e.g. between decoded segments / generated blocks) so a barge-in interrupts
promptly without killing the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import multiprocessing
import queue as queue_mod
import signal
import threading
from typing import Any

__all__ = [
    "ModelWorker",
    "WorkerCancelledError",
    "WorkerError",
    "raise_if_cancelled",
]


class WorkerError(RuntimeError):
    """The worker failed: startup failure, crash, or closed mid-request."""


class WorkerCancelledError(Exception):
    """Raised inside the child handler when cooperative cancellation fires."""


def raise_if_cancelled(cancel_event: Any) -> None:
    """Raise :class:`WorkerCancelledError` when the shared cancel event is set."""
    if cancel_event.is_set():
        raise WorkerCancelledError()


def _resolve_factory(path: str) -> Any:
    """Resolve ``"pkg.module:attr"`` (or dotted) to the factory callable."""
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, _, attr = path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _child_main(
    factory_path: str,
    factory_kwargs: dict[str, Any],
    req_q: Any,
    res_q: Any,
    cancel_event: Any,
) -> None:  # pragma: no cover - runs in the spawned child (covered by spawn tests)
    # The parent owns cancellation/teardown; the child must not race it by
    # dying on the terminal's SIGINT before cooperative shutdown is attempted.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        factory = _resolve_factory(factory_path)
        handler = factory(**factory_kwargs)
    except BaseException as exc:  # report any startup failure
        res_q.put(("fatal", None, f"{type(exc).__name__}: {exc}"))
        return
    res_q.put(("ready", None, None))
    while True:
        message = req_q.get()
        if message is None:
            break
        req_id, payload = message
        # A cancel signal always targets the request that was in flight when it
        # was set (the parent cancels only in-flight work). Clearing it at
        # pickup keeps a late signal from insta-cancelling fresh work while
        # still letting an in-flight handler observe its own cancellation.
        cancel_event.clear()
        try:
            value = handler(payload, cancel_event)
            res_q.put(("ok", req_id, value))
        except WorkerCancelledError:
            res_q.put(("cancelled", req_id, None))
        except BaseException as exc:  # serialize, don't crash
            res_q.put(("error", req_id, f"{type(exc).__name__}: {exc}"))
    res_q.put(("closed", None, None))


class ModelWorker:
    """One spawned, long-lived model process behind an async request API.

    Args:
        factory: Importable path (``"pkg.module:attr"``) of the handler
            factory; resolved and called **in the child**, so the heavy model
            import/construction never touches the event loop.
        factory_kwargs: Serializable keyword arguments for the factory.
        name: Human-readable worker name (process title / errors).
        ready_timeout_s: Budget for child startup + model load.
        grace_s: Per-stage shutdown budget (cooperative join, terminate join,
            kill join, bridge-thread join).
        request_maxsize: Bound of the request IPC queue.
    """

    def __init__(
        self,
        factory: str,
        factory_kwargs: dict[str, Any] | None = None,
        *,
        name: str = "model-worker",
        ready_timeout_s: float = 120.0,
        grace_s: float = 2.0,
        request_maxsize: int = 4,
    ) -> None:
        self._factory = factory
        self._factory_kwargs = dict(factory_kwargs or {})
        self._name = name
        self._ready_timeout_s = ready_timeout_s
        self._grace_s = grace_s
        self._ctx = multiprocessing.get_context("spawn")
        self._cancel_event = self._ctx.Event()
        self._req_q = self._ctx.Queue(maxsize=request_maxsize)
        self._res_q = self._ctx.Queue()
        self._proc: Any = None
        self._reader: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready: asyncio.Future[None] | None = None
        self._pending: dict[int, asyncio.Future[tuple[str, Any]]] = {}
        self._counter = 0
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def pid(self) -> int | None:
        """The child PID while alive, else ``None``."""
        if self._proc is None:
            return None
        try:
            return int(self._proc.pid) if self._proc.is_alive() else None
        except ValueError:  # process object already closed by aclose()
            return None

    @property
    def started(self) -> bool:
        """``True`` once :meth:`start` completed successfully."""
        return self._proc is not None and self._ready is not None and self._ready.done()

    async def start(self) -> None:
        """Spawn the child and wait until its model reports readiness."""
        if self._closed:
            raise WorkerError(f"{self._name}: worker already closed")
        if self._proc is not None:
            if self._ready is not None:
                await self._ready
            return
        self._loop = asyncio.get_running_loop()
        self._ready = self._loop.create_future()
        self._proc = self._ctx.Process(
            target=_child_main,
            args=(
                self._factory,
                self._factory_kwargs,
                self._req_q,
                self._res_q,
                self._cancel_event,
            ),
            name=self._name,
            daemon=True,
        )
        self._proc.start()
        self._reader = threading.Thread(
            target=self._read_results, name=f"{self._name}-bridge", daemon=True
        )
        self._reader.start()
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), self._ready_timeout_s)
        except TimeoutError:
            await self.aclose()
            raise WorkerError(
                f"{self._name}: model worker did not become ready within "
                f"{self._ready_timeout_s:.1f}s"
            ) from None
        except WorkerError:
            await self.aclose()
            raise

    def _read_results(self) -> None:
        """Bridge thread: pump child results onto the event loop."""
        assert self._loop is not None
        while not self._reader_stop.is_set():
            try:
                message = self._res_q.get(timeout=0.05)
            except queue_mod.Empty:
                if self._proc is not None and not self._proc.is_alive():
                    # Child died without a closing message: fail everything.
                    self._post(("fatal", None, "worker process died unexpectedly"))
                    return
                continue
            if message[0] == "closed":
                return
            self._post(message)

    def _post(self, message: tuple[str, Any, Any]) -> None:
        assert self._loop is not None
        with contextlib.suppress(RuntimeError):  # loop already closed at teardown
            self._loop.call_soon_threadsafe(self._dispatch, message)

    def _dispatch(self, message: tuple[str, Any, Any]) -> None:
        kind, req_id, value = message
        if kind == "ready":
            if self._ready is not None and not self._ready.done():
                self._ready.set_result(None)
            return
        if kind == "fatal":
            error = WorkerError(f"{self._name}: {value}")
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(error)
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()
            return
        pending: asyncio.Future[tuple[str, Any]] | None = self._pending.pop(req_id, None)
        if pending is not None and not pending.done():
            pending.set_result((kind, value))

    async def request(self, payload: Any) -> tuple[str, Any]:
        """Send one serializable request; return ``(status, value)``.

        ``status`` is ``"ok"``, ``"cancelled"``, or ``"error"`` (with the
        child's error text as ``value``). Requests are serialized — the worker
        processes one at a time, which is also what keeps result ordering
        trivially correct for callers.
        """
        if self._closed:
            raise WorkerError(f"{self._name}: worker already closed")
        await self.start()
        async with self._lock:
            assert self._loop is not None
            self._counter += 1
            req_id = self._counter
            future: asyncio.Future[tuple[str, Any]] = self._loop.create_future()
            self._pending[req_id] = future
            try:
                self._req_q.put_nowait((req_id, payload))
            except queue_mod.Full:
                self._pending.pop(req_id, None)
                raise WorkerError(f"{self._name}: request queue full") from None
            try:
                return await future
            finally:
                self._pending.pop(req_id, None)

    def signal_cancel(self) -> None:
        """Set the cooperative cancellation event the handler polls."""
        self._cancel_event.set()

    def clear_cancel(self) -> None:
        """Re-arm the cancellation event before dispatching new work."""
        self._cancel_event.clear()

    async def aclose(self) -> None:
        """Cooperative close -> timed join -> terminate -> join -> kill/join.

        Also stops the bridge thread and closes the IPC queues within the same
        budget; idempotent.
        """
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.is_alive():
            self._cancel_event.set()
            with contextlib.suppress(queue_mod.Full, ValueError):
                self._req_q.put_nowait(None)  # cooperative shutdown sentinel
            await asyncio.to_thread(proc.join, self._grace_s)
            if proc.is_alive():
                proc.terminate()
                await asyncio.to_thread(proc.join, self._grace_s)
            if proc.is_alive():  # pragma: no cover - kill is a last resort
                proc.kill()
                await asyncio.to_thread(proc.join, self._grace_s)
        self._reader_stop.set()
        if self._reader is not None:
            await asyncio.to_thread(self._reader.join, self._grace_s)
            self._reader = None
        error = WorkerError(f"{self._name}: worker closed")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        if self._ready is not None and not self._ready.done():
            self._ready.set_exception(WorkerError(f"{self._name}: closed before ready"))
            # The exception is retrieved by whoever awaits start(); if nobody
            # does, don't let asyncio log "exception was never retrieved".
            with contextlib.suppress(Exception):
                self._ready.exception()
        for q in (self._req_q, self._res_q):
            with contextlib.suppress(Exception):
                q.close()
                q.cancel_join_thread()
        if proc is not None:
            with contextlib.suppress(Exception):
                proc.close()
