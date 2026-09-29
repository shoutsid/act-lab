"""Kaggle TPU kernel entry for the null queue (generated, do not edit)."""
import glob
import runpy
import sys


def ensure_xla() -> str:
    """Torch-XLA, installing it if the image does not already carry it."""
    import subprocess

    try:
        import torch_xla  # noqa: F401
        return "already-present"
    except ImportError:
        pass
    for spec in ("torch_xla", "torch_xla[tpu]"):
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", spec],
            capture_output=True, text=True,
        )
        detail = (proc.stdout + proc.stderr)[-1500:]
        if proc.returncode == 0:
            try:
                import torch_xla  # noqa: F401
                return f"installed:{spec}"
            except ImportError:
                print(f"pip installed {spec} but torch_xla still will not import",
                      flush=True)
        print(f"pip {spec} failed ({proc.returncode}):\n{detail}", flush=True)
    return "unavailable"
def main() -> int:
    status = ensure_xla()
    print(f"torch_xla: {status}", flush=True)
    matches = sorted(
        glob.glob("/kaggle/input/**/scripts/tpu_null_queue.py", recursive=True)
    )
    if not matches:
        print("tpu_null_queue.py is not in the mounted code dataset", flush=True)
        return 2
    runner = matches[0]
    print(f"null queue runner: {runner}", flush=True)
    sys.argv = [
        runner, "--device", "xla", "--seeds", "15024-15035", "--steps", str(6000),
        "--workers", str(8), "--fidelity", "declared",
        "--dataset", "sid123123234/act-v0-5-2-null", "--output", "/kaggle/working/tpu-null-queue.json",
    ]
    runpy.run_path(runner, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
