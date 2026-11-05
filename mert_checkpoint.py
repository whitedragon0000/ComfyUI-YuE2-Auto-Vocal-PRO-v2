"""Safetensors -> MERT2 checkpoint loader, independent of the HF runtime.

`safe​tensors.torch.load_model` inspects model buffers for shared-storage
aliases *before* it calls ``load_state_dict``.  Torchaudio's spectrogram
window may be a view into a bigger backing storage, which makes that
preflight raise even for a valid MERT checkpoint.  Use the raw, safe tensor
reader and PyTorch's native, shape-checking state-dict loader instead.

Only model-synthesized *non-trainable DSP buffers* may be absent.  We require
all learned parameters and all other buffers to be present and reject any
unexpected tensors.  This is not a relaxed model compatibility check.
"""
from __future__ import annotations

import json
from pathlib import Path

# Torchaudio creates these buffers deterministically from the MERT config.
# Some published checkpoints contain them; others omit them.  These are never
# trainable MERT features, normalization stats, or layer weights.
REGENERABLE_DSP_BUFFERS = frozenset({
    'feature_extractor.spectrogram.window',
    'feature_extractor.mel_scale.fb',
})


def load_mert_checkpoint(model, weights_path, *, reporter=print, tensor_loader=None):
    """Load selected MERT weights and verify model coverage without shared-buffer scans.

    Returns a small diagnostics dictionary.  Does not download anything.
    ``tensor_loader`` is injectable for deterministic CPU regression tests.
    """
    if tensor_loader is None:
        from safetensors.torch import load_file
        tensor_loader = load_file

    path = Path(weights_path)
    if not path.is_file() or path.suffix.lower() != '.safetensors':
        raise RuntimeError(f'MERT model must be an existing .safetensors file: {path}')

    tensors = tensor_loader(str(path), device='cpu')
    if not isinstance(tensors, dict) or not tensors:
        raise RuntimeError(f'MERT safetensors contained no weight tensors: {path}')
    # Unrecognized keys are an error even if tensors were otherwise loadable.
    try:
        incompat = model.load_state_dict(tensors, strict=False)
    except (ValueError, RuntimeError, TypeError) as exc:
        raise RuntimeError(f'MERT model tensor names/shapes do not match {path}: {exc}') from exc

    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    optional = sorted(missing & REGENERABLE_DSP_BUFFERS)
    not_loaded = sorted(missing - REGENERABLE_DSP_BUFFERS)
    if not_loaded or unexpected:
        def sample(items):
            return ', '.join(items[:8]) + (f' ... (+{len(items)-8})' if len(items)>8 else '')
        raise RuntimeError(
            f'MERT checkpoint does not match MERT-v2-FullSong architecture: {path}. '
            f'Missing learned/required keys ({len(not_loaded)}): {sample(not_loaded) or "none"}. '
            f'Unexpected keys ({len(unexpected)}): {sample(sorted(unexpected)) or "none"}. '
            'Do not substitute another audio encoder as MERT.'
        )
    # The buffers are synthesized when the architecture is built.  They may
    # not be replaced by an arbitrary empty/uninitialized tensor.
    if optional:
        buffers = dict(model.named_buffers())
        for name in optional:
            value = buffers.get(name)
            if value is None or value.numel() == 0:
                raise RuntimeError(f'MERT regenerated DSP buffer is not initialized: {name}')
    diagnostics = {
        'stage': 'mert_checkpoint_load',
        'status': 'loaded',
        'method': 'safetensors.load_file+torch.load_state_dict',
        'model': str(path),
        'weight_tensors': len(tensors),
        'required_missing': 0,
        'unexpected': 0,
        'regenerated_dsp_buffers': optional,
    }
    reporter('[YuE2 Reference Voice][MERT CHECKPOINT] ' + json.dumps(diagnostics, ensure_ascii=False))
    return diagnostics
