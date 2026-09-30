









from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold as OriginalStratifiedKFold,
    StratifiedShuffleSplit as OriginalStratifiedShuffleSplit,
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def label_counts(y: Any) -> dict[str, int]:
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return {str(value.item() if hasattr(value, "item") else value): int(count)
            for value, count in zip(values, counts)}


class FallbackAudit:
    def __init__(self, path: Path, component: str):
        self.path = Path(path)
        self.component = component
        self.started_at = now()
        self.events: list[dict[str, Any]] = []

    def record(self, kind: str, **details: Any) -> None:
        event = {"event_index": len(self.events), "kind": kind, **details}
        self.events.append(event)
        print(f"[edge-case-fallback] {kind}: {details}", flush=True)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        counts = Counter(event["kind"] for event in self.events)
        payload = {
            "schema_version": 1,
            "component": self.component,
            "policy": {
                "outer_cv": (
                    "Use the requested stratified splitter; only on ValueError, "
                    "use deterministic KFold with the same n_splits, shuffle, and seed."
                ),
                "inner_validation": (
                    "Use the requested stratified shuffle split; only on ValueError, "
                    "use deterministic ShuffleSplit with the same validation size and seed."
                ),
                "temperature": (
                    "When split-half cross-fitting has an empty half, use neutral T=1 "
                    "for both experts on that held-out fold."
                ),
            },
            "started_at": self.started_at,
            "finished_at": now(),
            "n_events": len(self.events),
            "event_counts": dict(sorted(counts.items())),
            "events": self.events,
        }
        tmp = self.path.with_suffix(self.path.suffix + ".partial")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)


def make_safe_splitters(audit: FallbackAudit):
    class SafeStratifiedKFold:
        def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state
            self._original = OriginalStratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
            )

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def split(self, X, y, groups=None):
            try:
                splits = list(self._original.split(X, y, groups))
            except ValueError as exc:
                audit.record(
                    "outer_unstratified_kfold",
                    reason=str(exc),
                    n_samples=int(len(y)),
                    label_counts=label_counts(y),
                    n_splits=int(self.n_splits),
                    shuffle=bool(self.shuffle),
                    random_state=self.random_state,
                )
                random_state = self.random_state if self.shuffle else None
                fallback = KFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=random_state,
                )
                splits = list(fallback.split(X, y, groups))
            yield from splits

    class SafeStratifiedShuffleSplit:
        def __init__(
            self,
            n_splits=10,
            *,
            test_size=None,
            train_size=None,
            random_state=None,
        ):
            self.n_splits = n_splits
            self.test_size = test_size
            self.train_size = train_size
            self.random_state = random_state
            self._original = OriginalStratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                train_size=train_size,
                random_state=random_state,
            )

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

        def split(self, X, y, groups=None):
            try:
                splits = list(self._original.split(X, y, groups))
            except ValueError as exc:
                audit.record(
                    "inner_unstratified_shuffle",
                    reason=str(exc),
                    n_samples=int(len(y)),
                    label_counts=label_counts(y),
                    n_splits=int(self.n_splits),
                    test_size=self.test_size,
                    train_size=self.train_size,
                    random_state=self.random_state,
                )
                fallback = ShuffleSplit(
                    n_splits=self.n_splits,
                    test_size=self.test_size,
                    train_size=self.train_size,
                    random_state=self.random_state,
                )
                splits = list(fallback.split(X, y, groups))
            yield from splits

    return SafeStratifiedKFold, SafeStratifiedShuffleSplit
