"""Single-source, bounded execution progress for the Reference Voice graph.

Never emits a second native ComfyUI ProgressBar: native YuE2/ComfyUI KSampler
already publish their own per-subtask progress. The extra parent ProgressBar
was the cause of double-counted/misrouted UI progress in earlier versions.
Per-run websocket milestones display only actual completed work, not an ETA.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import threading
import time
import uuid

_LOCK = threading.RLock()
_STATES = {}
_MAX_RUNS = 128
_STAGES = {
    'graph_prepared': 1, 'models_validated': 3,
    'reference_lyrics_started': 5, 'reference_lyrics_completed': 12,
    'score_selection_started': 13, 'score_selection_completed': 22,
    'score_alignment_started': 23, 'score_alignment_completed': 35,
    'mert_tokenizer_started': 36, 'mert_tokenizer_completed': 48,
    'ar_started': 49, 'ar_completed': 76,
    'nar_sampling_started': 77, 'nar_sampling_completed': 94,
    'audio_decoded': 96, 'prefix_trimmed': 98, 'execution_completed': 100,
}


@dataclass
class _State:
    node_id: str
    run_id: str
    value: int = 0
    stage: str = ''
    updated: float = 0.0


def _emit(event: dict):
    try:
        from server import PromptServer
        instance = getattr(PromptServer, 'instance', None)
        if instance:
            instance.send_sync('yue2.reference_progress', event)
    except Exception as e:
        logging.debug('Reference progress event unavailable: %s', e)


def start_run(node_id):
    raw = str(node_id or '')
    if not raw:
        return ''
    run = uuid.uuid4().hex[:16]
    key = raw + '::' + run
    with _LOCK:
        now = time.monotonic()
        for previous in list(_STATES):
            if now-_STATES[previous].updated > 3600:
                del _STATES[previous]
        if len(_STATES) >= _MAX_RUNS:
            for previous in sorted(_STATES, key=lambda x: _STATES[x].updated)[:32]:
                del _STATES[previous]
        _STATES[key] = _State(raw, run, updated=now)
    return key


def report_progress(key, stage, value=None):
    if stage not in _STAGES and stage not in ('ar_tokens', 'nar_steps'):
        raise ValueError('Unknown YuE2 stage: ' + stage)
    target = _STAGES.get(stage) if value is None else value
    target = min(100, max(0, int(target)))
    if stage != 'execution_completed':
        target = min(99, target)
    key = str(key or '')
    if not key:
        return target
    with _LOCK:
        state = _STATES.get(key)
        if not state:
            # Preserve older direct-node workflows while avoiding native bar.
            base, _, runid = key.partition('::')
            state = _State(base, runid or 'legacy', updated=time.monotonic())
            _STATES[key] = state
        if target < state.value or (target == state.value and stage == state.stage):
            return state.value
        state.value = target
        state.stage = stage
        state.updated = time.monotonic()
        payload = {'node_id': state.node_id, 'run_id': state.run_id,
                   'value': state.value, 'total': 100, 'stage': stage,
                   'completed': stage == 'execution_completed'}
    _emit(payload)
    if stage not in ('ar_tokens', 'nar_steps'):
        print(f'[YuE2 Reference Voice][PROGRESS] {target}% | {stage} | node={payload["node_id"]} | run={payload["run_id"]}')
    return target


def reset_for_tests():
    with _LOCK:
        _STATES.clear()
