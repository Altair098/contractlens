from __future__ import annotations

import contextlib
from typing import Any, Iterator

_TRACING = {"enabled": False}


def enable_tracing(enabled: bool = True) -> None:
    _TRACING["enabled"] = enabled
    if enabled:
        try:
            import mlflow

            mlflow.litellm.autolog()
        except Exception:
            pass


@contextlib.contextmanager
def span(name: str, span_type: str = "UNKNOWN", inputs: dict[str, Any] | None = None) -> Iterator[Any]:
    if not _TRACING["enabled"]:
        yield None
        return
    import mlflow

    with mlflow.start_span(name=name, span_type=span_type) as s:
        if inputs:
            s.set_inputs(inputs)
        yield s
