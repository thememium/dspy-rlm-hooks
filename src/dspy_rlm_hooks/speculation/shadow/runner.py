"""Parent-side ShadowRunner driver: subprocess lifecycle, dispatch of recorded
tool calls, peek plan handling, and chained-continuation bookkeeping."""

from __future__ import annotations

import pickle
import threading
from types import SimpleNamespace
from typing import Any

from dspy_rlm_hooks.speculation.shadow.analysis import STMT_WALL_BUDGET_S
from dspy_rlm_hooks.speculation.shadow.snapshot import classify_ns
from dspy_rlm_hooks.speculation.shadow.worker import _mp_context, _shadow_worker
from dspy_rlm_hooks.speculation.store import SpecStore
from dspy_rlm_hooks.speculation.streaming import ChainMeta, Segment
from dspy_rlm_hooks.speculation.tool import (
    canonical_hash,
    contains_nonspec,
    spec_key,
    split_batch_call,
)

# =============================================================================
# parent side
# =============================================================================


class ShadowRunner:
    """Subprocess-isolated speculative executor of generated code.

    The worker process executes each :class:`Segment` against a deepcopy-forked
    namespace with jailed builtins and recording hooks. Predicted ``(tool, args)``
    calls and peek plans stream back to the parent for claim-hook dispatch.
    """

    def __init__(
        self,
        real_locals: dict,
        shadow_hooks: dict,
        store: SpecStore,
        real_builtins: dict,
        launcher=None,
        registry=None,
        taint_skip: bool = True,
        stmt_budget: float = STMT_WALL_BUDGET_S,
        persistent: bool = False,
    ) -> None:
        self.store = store
        self.launcher = launcher  # needed only for peeks
        self.registry = registry
        self.taint_skip = taint_skip
        self.hooks = dict(shadow_hooks)
        self.persistent = persistent  # stay alive across turns (reset per turn)
        self.aborted: str | None = None
        self.executed = 0
        self.predicted: list[tuple[str, tuple]] = []  # (tool_name, args)
        self._last_peek_tally: dict = {}  # spec_key -> count from the last plan
        self._done = threading.Event()
        self._turn_ended = threading.Event()
        self._turn_open = False  # worker is between reset and end_turn
        # chained-continuation state (see plan_peeks_with_chains)
        self._chain_lock = threading.Lock()
        self._pending_chains: dict[int, ChainMeta] = {}
        self._cont_keys: dict[int, Any] = {}  # cont_id -> realized claim key
        self._segkey_to_real: dict[tuple, Any] = {}  # raw-material id -> claim key
        # batched-call decomposition: element peeks under per-element claim
        # keys, assembled back into the ordered list when all resolve
        self._batch_seq = 0
        self._batch_keys: dict[Any, list] = {}  # batch key -> [element keys]
        self._batch_values: dict[Any, dict[int, Any]] = {}  # batch key -> {idx: value}
        self._key_to_batch: dict[
            Any, tuple[Any, int]
        ] = {}  # element key -> (batch key, idx)
        self._chain_ready: dict[int, set[str]] = {}  # cont_id -> satisfied deps
        self._dep_values: dict[int, dict[str, Any]] = {}  # cont_id -> dep results
        self._send_lock = threading.Lock()  # the pipe is not thread-safe
        self._closed = False
        # classified seed — kept so a crashed worker can respawn without
        # re-serializing the host namespace
        self._seed = classify_ns(real_locals)
        self._real_builtins = real_builtins
        self._stmt_budget = stmt_budget
        # last per-turn seed pushed via begin_turn(ns) — skipping an identical
        # reseed avoids re-serializing the whole namespace every turn
        self._last_turn_seed = dict(real_locals)
        self._spawn()

    # -- process lifecycle ----------------------------------------------------
    def _spawn(self) -> None:
        """Start (or restart) the worker subprocess from the classified seed."""
        self._closed = False
        ctx = _mp_context()
        self._parent_conn, child_conn = ctx.Pipe()
        payload = {
            "ns_seed": self._seed,
            "spec_names": set(self.hooks),
            "taint_skip": self.taint_skip,
            "budget": self._stmt_budget,
        }
        self._proc = ctx.Process(
            target=_shadow_worker,
            args=(child_conn, self._parent_conn, payload),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._conn = self._parent_conn
        self._done = threading.Event()
        self._turn_ended = threading.Event()
        self._turn_open = True  # a freshly spawned worker must be drained too
        self._reader = threading.Thread(
            target=self._read, daemon=True, name="shadow-reader"
        )
        self._reader.start()
        if (
            self.launcher is not None
            and getattr(self.launcher, "bus", None) is not None
        ):
            self.launcher.bus.subscribe(self._on_bus_event)

    # -- chained continuations -------------------------------------------------
    def _on_bus_event(self, kind: str, data: dict) -> None:
        """Watch speculation resolutions and fire continuations whose
        dependencies are satisfied."""
        if self._closed or kind != "ready":
            return
        key = data.get("key")
        spec = data.get("spec")
        try:
            self._fire_chains_for_key(key, spec)
        except Exception:
            pass  # continuation fires must never break dispatch

    def _chain_dep_matches(self, meta: ChainMeta, key: Any) -> bool:
        """True when any dep of the chain refers to the given claim key."""
        for _name, (ref_kind, ref) in meta.deps.items():
            if ref_kind == "key" and ref == key:
                return True
            if ref_kind == "cont" and self._cont_keys.get(ref) == key:
                return True
        return False

    def _fire_chains_for_key(self, key: Any, spec: Any) -> None:
        """Record a producer resolution; fire every chain whose deps are now
        all satisfied. A FAILED producer drops its dependent chains (the real
        run will surface the same error through its own claim)."""
        if key in self._key_to_batch:
            self._record_batch_element(key, spec)
            return
        if spec is not None and spec.error is not None:
            with self._chain_lock:
                self._pending_chains = {
                    cid: m
                    for cid, m in self._pending_chains.items()
                    if not self._chain_dep_matches(m, key)
                }
            return
        value = spec._result if spec is not None else None
        to_fire: list[tuple[int, dict[str, Any]]] = []
        with self._chain_lock:
            for cont_id, meta in self._pending_chains.items():
                if not self._chain_dep_matches(meta, key):
                    continue
                values = self._dep_values.setdefault(cont_id, {})
                for name, (ref_kind, ref) in meta.deps.items():
                    if name in values:
                        continue
                    if (ref_kind == "key" and ref == key) or (
                        ref_kind == "cont" and self._cont_keys.get(ref) == key
                    ):
                        values[name] = value
                    elif ref_kind == "key":
                        # dep already ready from an earlier resolution
                        prior = self._result_for_key(ref)
                        if prior is not None or self._key_resolved(ref):
                            values[name] = prior
                if meta.deps and all(n in values for n in meta.deps):
                    to_fire.append((cont_id, dict(values)))
        for cont_id, values in to_fire:
            self._send_chain_fire(cont_id, values)

    def _key_resolved(self, key: Any) -> bool:
        """True when some speculation for the key resolved (value may be None)."""
        with self.store._lock:
            for spec in self.store._q.get(key, ()):
                if spec.state in ("ready", "claimed"):
                    return True
        return False

    def _result_for_key(self, key: Any) -> Any:
        """Best-effort resolved value for a key (for deps already ready when a
        chain registers later)."""
        with self.store._lock:
            for spec in self.store._q.get(key, ()):
                if spec.state in ("ready", "claimed") and spec.error is None:
                    return spec._result
        return None

    def _send_chain_fire(self, cont_id: int, values: dict[str, Any]) -> None:
        blobs: dict[str, bytes] = {}
        for name, value in values.items():
            try:
                blobs[name] = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                return  # unpicklable dep result: drop the chain
        try:
            with self._send_lock:
                self._conn.send(("fire_chain", cont_id, blobs))
        except Exception:
            pass

    @property
    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def ensure_alive(self) -> None:
        """Respawn the worker if it died (a persistent runner outlives crashes)."""
        if not self._proc.is_alive():
            self.aborted = None
            self._spawn()

    # -- producer side --------------------------------------------------------
    def feed(self, seg: Segment) -> None:
        with self._send_lock:
            self._conn.send(seg)

    def feed_peek(self, tail: str) -> None:
        """Queue a peek over the current unclosed tail. Runs in the worker AFTER
        all fed statements, so the namespace it evaluates against is exactly the
        state those statements produced."""
        self._conn.send(("peek", tail))

    def begin_turn(self, ns: dict | None = None) -> None:
        """Start a turn on a persistent runner: reset worker state to the seed
        and clear this turn's parent-side accumulators. Pipe ordering
        guarantees the reset lands before any segments fed after it.

        ``ns`` (when given) RESEEDS the worker: the original seed captures only
        ``input_args``, but cross-iteration state sync produces a fresh live
        snapshot each turn; without reseeding, a persistent worker would reset
        to stale values every turn. An identical namespace skips the reseed
        (no re-serialization)."""
        self.ensure_alive()
        self.predicted = []
        self.executed = 0
        self.aborted = None
        self._last_peek_tally = {}
        with self._chain_lock:
            self._pending_chains = {}
            self._cont_keys = {}
            self._segkey_to_real = {}
            self._batch_keys = {}
            self._batch_values = {}
            self._key_to_batch = {}
            self._chain_ready = {}
            self._dep_values = {}
        self._turn_ended.clear()
        reseed = None
        if ns is not None and ns != self._last_turn_seed:
            self._seed = classify_ns(ns)
            self._last_turn_seed = dict(ns)
            reseed = self._seed
        with self._send_lock:
            self._conn.send(("reset", reseed))
        self._turn_open = True

    def end_turn(self, timeout: float | None = None) -> bool:
        """Drain a persistent runner's turn: wait until the worker has
        processed every message fed so far. Returns True when acknowledged."""
        if not self._turn_open:
            return True
        self._turn_open = False
        if not self._proc.is_alive():
            return False
        try:
            with self._send_lock:
                self._conn.send(("end_turn",))
        except (OSError, BrokenPipeError):
            return False
        return self._turn_ended.wait(timeout)

    def shutdown(self) -> None:
        """Terminate a persistent runner gracefully (session close)."""
        self._closed = True
        self._turn_open = False
        self.finish()
        self.join(5)

    def finish(self) -> None:
        try:
            with self._send_lock:
                self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass  # already terminated (e.g. abort() raced ahead of finish())

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker to finish; terminate it on timeout. Returns True
        if it completed (a hang never blocks real execution)."""
        self._proc.join(timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(5)
            return False
        self._reader.join(timeout or 5)
        return True

    def abort(self, why: str = "external") -> None:
        self._closed = True
        self.aborted = why
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(2)  # deterministic death for respawn checks

    # -- reader ---------------------------------------------------------------
    def _read(self) -> None:
        try:
            while True:
                msg = self._conn.recv()
                if msg is None:
                    break
                kind = msg[0]
                if kind == "tool":
                    _, name, args, kwargs = msg[:4]
                    cont_id = msg[4] if len(msg) > 4 else None
                    self.predicted.append((name, args))
                    key = self._dispatch(name, args, kwargs)
                    if key is not None:
                        seg_id = (name, canonical_hash(name, args, kwargs))
                        with self._chain_lock:
                            self._segkey_to_real[seg_id] = key
                            if cont_id is not None:
                                self._cont_keys[cont_id] = key
                elif kind == "plans":
                    self._handle_plans(msg[1], msg[2] if len(msg) > 2 else ())
                elif kind == "executed":
                    self.executed += 1
                elif kind == "abort":
                    self.aborted = msg[1]
                elif kind == "turn_ended":
                    self._turn_ended.set()
                elif kind == "evict_tool":
                    self.store.evict_tool(msg[1], "shadow-rebind")
        except (EOFError, OSError):
            pass
        finally:
            self._done.set()

    def _dispatch(self, name: str, args: tuple, kwargs: dict) -> Any:
        """Dispatch one shadow-recorded call; returns its claim key (or None).

        The call comes from an EXECUTED segment, i.e. the model's statement
        closed: its speculation is ADOPTED so tail-shrink bet retraction can
        no longer evict it before the real run claims it.
        """
        if self.launcher is None:
            return None
        tool = self.registry.get(name) if self.registry else None
        if tool is None or not tool.speculatable:
            return None
        if tool.gate_fn and not tool.gate_fn(args, kwargs):
            return None
        if contains_nonspec(args) or contains_nonspec(kwargs):
            # the sandbox recorded this call with a TAINT MARKER in its args
            # (assigned and used within the same segment): never dispatch a
            # speculative call with garbage arguments
            return None
        # Batched tools (llm_query_batched & friends) are claimed PER ELEMENT
        # by the real run, so a whole-batch peek could never be claimed —
        # dispatch one peek per prompt element instead.
        decomposed = self._decompose_batched(name, args, kwargs, needed=1, adopt=True)
        if decomposed is not None:
            return decomposed[0]
        self.launcher.ensure_peeked(tool, args, kwargs, 1)
        key = spec_key(tool, args, kwargs)
        self.store.adopt(key)
        return key

    def _batch_key_for(self, name: str, args: tuple, kwargs: dict) -> Any:
        """Synthetic stable key for a batched call's ASSEMBLED result, or None
        when the call is not batched-shaped (or has no single-tool registry
        entry to claim elements against)."""
        registry = self.registry
        if registry is None or not name.endswith("_batched"):
            return None
        split = split_batch_call(args, kwargs)
        if split is None:
            return None
        prompts, rest, clean, _kwname = split
        single = registry.get(name[: -len("_batched")])
        if single is None:
            return None
        return ("__batch__", f"{name}|{spec_key(single, (prompts,) + rest, clean)[1]}")

    def _decompose_batched(
        self, name: str, args: tuple, kwargs: dict, needed: int, adopt: bool
    ) -> tuple[Any, list] | None:
        """Dispatch one peek PER ELEMENT of a batched call, under the same
        per-element claim keys the real run uses. Registers a batch group so
        the ordered list of element results can be assembled and published to
        dependent chains when every element resolves. Returns
        ``(batch_key, elem_keys)``, or None when the call is not batched-shaped
        (the caller falls back to a whole-batch dispatch)."""
        if self.launcher is None or self.registry is None:
            return None
        batch_key = self._batch_key_for(name, args, kwargs)
        if batch_key is None:
            return None
        split = split_batch_call(args, kwargs)
        if split is None:
            return None
        prompts, rest, clean, _kwname = split
        single = self.registry.get(name[: -len("_batched")])
        assert single is not None
        elem_keys: list = []
        for elem in prompts:
            elem_args = (elem,) + rest
            key = spec_key(single, elem_args, clean)
            elem_keys.append(key)
            self.launcher.ensure_peeked(single, elem_args, clean, needed)
            if adopt:
                self.store.adopt(key)
        with self._chain_lock:
            self._batch_keys[batch_key] = elem_keys
            self._batch_values.setdefault(batch_key, {})
            for i, k in enumerate(elem_keys):
                self._key_to_batch[k] = (batch_key, i)
        return batch_key, elem_keys

    def _record_batch_element(self, key: Any, spec: Any) -> None:
        """Record one batch element's resolved value; assemble and publish the
        ordered list to dependent chains when the last element lands."""
        batch_key, idx = self._key_to_batch[key]
        with self._chain_lock:
            values = self._batch_values.setdefault(batch_key, {})
            if spec is not None and spec.error is None:
                values[idx] = spec._result
            else:
                # a failed element means the batch result is garbage: never
                # assemble (dependent chains stay unfired, like failed producers)
                self._batch_keys.pop(batch_key, None)
                return
            keys = self._batch_keys.get(batch_key, [])
            complete = bool(keys) and all(i in values for i in range(len(keys)))
            assembled = [values[i] for i in range(len(keys))] if complete else None
        if complete:
            # publish the assembled list under the batch key (recursion depth 1:
            # the batch key is never itself a batch element)
            self._fire_chains_for_key(
                batch_key, SimpleNamespace(_result=assembled, error=None)
            )

    def _handle_plans(self, plans, chain_metas=()) -> None:
        if self.launcher is None:
            return
        new_tally: dict = {}
        # dedupe by HASHABLE identity (batched args contain lists)
        seen: dict[tuple, int] = {}
        ordered: list = []
        for p in plans:
            tool = self.registry.get(p.tool) if self.registry else None
            if tool is None or not tool.speculatable:
                continue
            if tool.gate_fn and not tool.gate_fn(p.args, p.kwargs):
                continue
            ident = (p.tool, repr(p.args), repr(sorted(p.kwargs.items())))
            if ident in seen:
                seen[ident] += 1
                continue
            seen[ident] = 1
            ordered.append((tool, p.args, p.kwargs))
        for tool, args, kwargs in ordered:
            needed = seen[(tool.name, repr(args), repr(sorted(kwargs.items())))]
            decomposed = self._decompose_batched(
                tool.name, args, kwargs, needed, adopt=False
            )
            if decomposed is not None:
                batch_key, elem_keys = decomposed
                # element-level tally: retraction must evict per-element peeks
                for elem_key in elem_keys:
                    new_tally[elem_key] = needed
                continue
            self.launcher.ensure_peeked(tool, args, kwargs, needed)
            new_tally[spec_key(tool, args, kwargs)] = needed
        # BET RETRACTION: a key the previous plan justified but this one doesn't
        # means new tokens invalidated the bet — evict the stale peeks now.
        for key, old_n in self._last_peek_tally.items():
            keep = new_tally.get(key, 0)
            if keep < old_n:
                self.store.evict_unadopted_peeks(key, keep, "peek-retracted")
        self._last_peek_tally = new_tally

        # CHAIN REGISTRATION: the newest generation replaces pending chains.
        # Worker-side Plan keys hash raw args; real claim keys hash the
        # signature-bound material — translate before watching.
        plan_key_to_real: dict = {}
        for p in plans:
            if p.key is None:
                continue
            tool = self.registry.get(p.tool) if self.registry else None
            if tool is not None:
                batch_key = self._batch_key_for(p.tool, p.args, p.kwargs)
                if batch_key is not None:
                    # the batch's ASSEMBLED list is what dependent chains see
                    plan_key_to_real[p.key] = batch_key
                else:
                    plan_key_to_real[p.key] = spec_key(tool, p.args, p.kwargs)
        translated: list[ChainMeta] = []
        for m in chain_metas:
            deps: dict[str, Any] = {}
            unwatchable = False
            for name, (ref_kind, ref) in m.deps.items():
                if ref_kind == "key":
                    deps[name] = (ref_kind, plan_key_to_real.get(ref, ref))
                elif ref_kind == "segkey":
                    real = self._segkey_to_real.get(ref)
                    if real is None:
                        # the producer dispatch has not been processed yet:
                        # drop the chain (a later peek re-plans it)
                        unwatchable = True
                        break
                    deps[name] = ("key", real)
                else:
                    deps[name] = (ref_kind, ref)
            if not unwatchable:
                translated.append(ChainMeta(cont_id=m.cont_id, tool=m.tool, deps=deps))
        with self._chain_lock:
            self._pending_chains = {m.cont_id: m for m in translated}
        for meta in translated:
            for name, (ref_kind, ref) in meta.deps.items():
                if ref_kind != "key":
                    continue
                if self._key_resolved(ref):
                    spec = self._spec_for_key(ref)
                    if spec is not None:
                        self._fire_chains_for_key(ref, spec)
                        break

    def _spec_for_key(self, key: Any) -> Any:
        with self.store._lock:
            for spec in self.store._q.get(key, ()):
                if spec.state in ("ready", "claimed"):
                    return spec
        return None
