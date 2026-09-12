from pathlib import Path

import pytest

from act.training import SessionCheckpointPolicy, retain_checkpoints


def test_session_policy_gates_interval_and_size(tmp_path: Path) -> None:
    policy = SessionCheckpointPolicy(interval=4, max_checkpoints=2, max_bytes=4)
    assert not policy.should_save(0)
    assert policy.should_save(8)
    checkpoint = tmp_path / "checkpoint-8.pt"
    checkpoint.write_bytes(b"1234")
    policy.accept(checkpoint)
    checkpoint.write_bytes(b"12345")
    with pytest.raises(ValueError, match="exceeds"):
        policy.accept(checkpoint)


def test_retain_checkpoints_keeps_newest_files(tmp_path: Path) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"checkpoint-{index}.pt"
        path.write_bytes(str(index).encode())
        paths.append(path)
    retained = retain_checkpoints(tmp_path, max_checkpoints=2)
    assert len(retained) == 2
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 2
