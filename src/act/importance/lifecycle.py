"""Append-only expert lifecycle fossils for topology research."""

from dataclasses import asdict, dataclass
import json

from .signals import ExpertSignal


@dataclass(frozen=True)
class LifecycleFossil:
    step: int
    operation: str
    expert_id: str
    layer_id: str
    architecture_hash_before: str
    architecture_hash_after: str
    signal: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LifecycleLedger:
    """In-memory append-only ledger that serializes deterministic topology fossils."""

    def __init__(self) -> None:
        self._records: list[LifecycleFossil] = []

    def record(
        self,
        *,
        step: int,
        operation: str,
        expert_id: str,
        layer_id: str,
        architecture_hash_before: str,
        architecture_hash_after: str,
        signal: ExpertSignal | None = None,
    ) -> LifecycleFossil:
        if step < 0:
            raise ValueError("step must be non-negative")
        if not operation or not expert_id or not layer_id:
            raise ValueError("operation, expert_id, and layer_id are required")
        fossil = LifecycleFossil(
            step,
            operation,
            expert_id,
            layer_id,
            architecture_hash_before,
            architecture_hash_after,
            signal.to_dict() if signal else None,
        )
        self._records.append(fossil)
        return fossil

    @property
    def records(self) -> tuple[LifecycleFossil, ...]:
        return tuple(self._records)

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n"
            for record in self._records
        )
