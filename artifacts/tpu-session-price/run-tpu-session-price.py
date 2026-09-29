"""Kaggle TPU kernel entry for the session price probe (generated, do not edit)."""
import glob
import runpy
import sys


def ensure_xla() -> str:
    """Torch-XLA, installing it if the image does not already carry it."""
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
        glob.glob("/kaggle/input/**/scripts/tpu_session_price_probe.py", recursive=True)
    )
    if not matches:
        print("tpu_session_price_probe.py is not in the mounted code dataset", flush=True)
        return 2
    probe = matches[0]
    print(f"price probe: {probe}", flush=True)
    sys.argv = [
        probe, "--device", "xla", "--steps", str(240), "--seed", str(15012),
        "--output", "/kaggle/working/tpu-session-price-probe.json",
    ]
    runpy.run_path(probe, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
