"""Activation checkpointing, on the backend that will run it.

The model blocks recompute their activations in the backward pass instead of storing
them, which is what makes the declared 285M shape fit the hardware it was pre-registered
against (section 5.2's fit arithmetic, and the reason ``declared_session_config`` sets
``activation_checkpointing=True`` explicitly). The call that does that, however, is not
the same call on every backend.

**Measured 2026-09-29, on a TPU.** The declared-shape session path died 4.5 s into its
first training step with ``AttributeError: module 'torch' has no attribute 'xla'`` - in
all three phases that train. The site is ``torch/utils/checkpoint.py``:

    def _get_device_module(device="cuda"):
        if device == "meta":
            return torch.device("meta")
        device_module = getattr(torch, device)   # device="xla" -> AttributeError
        return device_module

reached from ``_checkpoint_without_reentrant_generator_impl`` - i.e. from the
``use_reentrant=False`` path this package calls - whenever a checkpointed input lives on
a non-CPU device, because the RNG state it preserves is fetched per device through that
module. XLA is not a device module on ``torch``: it is a separate package. So
``torch.utils.checkpoint`` cannot checkpoint anything on a TPU, at any dtype, and the
failure is not about the model: it is that the call has never been run on this backend.
Upstream says the same thing in both directions - PyTorch/XLA ships its own fork of the
file for exactly this, and its public entry point documents that only the reentrant
implementation is available on XLA (``raise ValueError("XLA currently does not support
use_reentrant==False")``).

Hence one dispatch, and the *device decides* - the same rule ``step_marker`` applies to
the lazy backend's step boundary, and for the same reason: a difference that only shows
up on one accelerator should be settled in one place rather than at four call sites.
The XLA fork also inserts the ``optimization_barrier`` XLA needs between the saved
forward and the recompute, so the two implementations are not interchangeable even
where both run.

Two consequences worth stating rather than leaving to be rediscovered:

* **The XLA call is the reentrant one.** Torch's non-reentrant implementation is what
  the CUDA campaign runs; on XLA the only implementation is the reentrant one, so the
  recomputation boundary is per-block either way but the mechanism differs. The blocks
  carry no dropout and consume no RNG, which is what makes the reentrant path's
  RNG-restoring recompute bit-identical here - the same property the resume contract
  rests on.
* **It is not a drop-in for every checkpoint in this repository.** The fork's ``backward``
  returns the gradients of its inputs, so a checkpoint whose function returns a tuple
  (the chunked scan) is fine, but a caller that re-enters ``torch.autograd.grad`` - which
  only the non-reentrant implementation supports - would not be. Every call site here
  trains with ``.backward()``.

    from act.model.checkpointing import activation_checkpoint

    hidden = activation_checkpoint(hidden.device, block, hidden, positions)
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def activation_checkpoint(
    device: torch.device, function: Callable[..., Any], *args: Any
) -> Any:
    """Run ``function``, recomputing its activations in the backward pass.

    ``device`` is the device of the tensors being checkpointed (``hidden.device`` at
    every call site here), not a requested string: what matters is where the tensors
    actually are, so a model moved after construction is checkpointed on the backend it
    ended up on.

    On the CPU and on CUDA this is ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``
    unchanged - the call the GPU campaign's records were produced with, and the one whose
    gradient semantics (no re-entrant autograd, ``autograd.grad`` supported) the training
    loop relies on. On XLA it is PyTorch/XLA's own implementation, because the upstream
    one raises before the first block runs (see the module docstring).
    """
    if device.type == "xla":
        return _xla_activation_checkpoint(function, *args)
    return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)


def _xla_activation_checkpoint(function: Callable[..., Any], *args: Any) -> Any:
    """PyTorch/XLA's fork of the checkpoint utility, imported where it is used.

    Imported here rather than at module scope so a CPU or CUDA session never pays for it,
    and refused by name rather than by falling through to the upstream call: an ``xla``
    tensor cannot exist without this package, so reaching the error below means the device
    and the environment disagree - and the message should say which of the two is missing
    rather than letting the reader find ``module 'torch' has no attribute 'xla'`` at the
    first training step of a queued run.

    No ``use_reentrant`` argument is passed: the fork accepts the keyword and rejects
    ``False``, so passing the upstream call's shape through would turn a working path into
    a ``ValueError``. The default (``True``) is the only implementation XLA has.
    """
    try:
        from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint
    except ImportError as error:  # pragma: no cover - an xla tensor implies the package
        raise RuntimeError(
            "checkpointed activations were requested for an XLA device, but torch_xla "
            "is not installed; torch's own checkpoint cannot run on XLA (it resolves "
            "the device module as torch.xla, which does not exist), so the XLA "
            "implementation is required"
        ) from error
    return xla_checkpoint(function, *args)
