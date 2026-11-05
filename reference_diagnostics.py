"""Privacy-aware diagnostics for Reference Voice; never changes generation inputs.

Hashes establish which text reached the score composer and the CLIP encoder.
They do *not* imply that an autoregressive model sang that text. Segmented
ASR measures the decoded output, independently of the model's requested text.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import uuid
import zipfile

from .lyrics_audit import word_tokens

SCHEMA_VERSION = 'reference-voice-trace/1.0'
DIAGNOSTIC_MODES = ('off', 'basic', 'segmented')


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def _sung_tokens(value: str) -> list[str]:
    # Uses the same parser that enforces immutable user words in the planner.
    from .lyric_contract import _tokens
    return [str(w).casefold() for w in _tokens(value)[0]]


def text_proof(text: str) -> dict:
    words = _sung_tokens(text)
    sections = re.findall(r'(?m)^\s*\[([^\]]+)\]', str(text))[:80]
    allowed = {'verse','chorus','pre-chorus','bridge','intro','outro','interlude',
               'instrumental','end','hook','solo','break','refrain'}
    section_types = [x.strip().casefold() if x.strip().casefold() in allowed
                     else 'custom_section' for x in sections]
    return {
        'raw_sha256': sha256_text(text),
        'sung_word_sequence_sha256': sha256_text('\u241f'.join(words)),
        'sung_words': len(words),
        'characters': len(str(text)),
        'section_headers': section_types,
    }


def composition_proof(reference: str, target: str, composite: str) -> dict:
    ref = _sung_tokens(reference)
    tgt = _sung_tokens(target)
    got = _sung_tokens(composite)
    total = len(ref) + len(tgt)
    boundary_ok = got == ref + tgt
    return {
        'reference': text_proof(reference),
        'target': text_proof(target),
        'composite': text_proof(composite),
        'expected_boundary_word_index': len(ref),
        'expected_total_words': total,
        'reference_prefix_exact': got[:len(ref)] == ref,
        'target_suffix_exact': got[len(ref):] == tgt,
        'all_words_in_order': boundary_ok,
        'reference_and_target_same_words': ref == tgt,
        'note': 'Exact input-word sequence, not acoustic word fidelity or decoder isolation.',
    }


def audio_fingerprint(audio) -> dict:
    """Partial sample fingerprint, not an audio export; handles CPU/GPU tensors.

    Sampling limits GPU->CPU traffic regardless of audio duration. The hash is
    for cross-report role checks, not to cryptographically authenticate an
    entire waveform (and is labeled as such).
    """
    from collections.abc import Mapping
    if not isinstance(audio, Mapping) or 'waveform' not in audio or 'sample_rate' not in audio:
        return {'available': False}
    t = audio['waveform']
    if getattr(t, 'ndim', 0) != 3:
        return {'available': False, 'reason': 'expected_3d'}
    frames = int(t.shape[-1]); sr = int(audio['sample_rate'])
    if frames < 1 or sr < 1:
        return {'available': False, 'reason': 'empty'}
    digest = hashlib.sha256(f'{sr}:{tuple(int(s) for s in t.shape)}'.encode())
    slices = []
    # Up to 3x 2048 samples; no bulk download from GPU.
    take = min(2048, frames)
    for offset in (0, max(0, frames // 2 - take // 2), max(0, frames - take)):
        sample = t[..., offset:offset + take].detach().to(dtype=__import__('torch').float32, device='cpu').contiguous()
        digest.update(sample.numpy().tobytes())
        slices.append({'start_seconds': round(offset / sr, 3), 'sample_count': take})
    return {
        'available': True, 'sample_rate': sr, 'channels': int(t.shape[1]),
        'duration_seconds': round(frames / sr, 3),
        'partial_pcm_sha256': digest.hexdigest(), 'sample_windows': slices,
        'scope': 'Fingerprint of sampled PCM windows; no audio content saved.',
    }


def temporal_windows(duration_seconds: float, *, window_seconds=20.0, max_windows=6) -> list[tuple[float, float]]:
    duration = float(duration_seconds)
    if duration <= 0:
        return []
    # For long outputs, spread windows over the song so an outro isn't missed.
    if duration <= window_seconds * max_windows:
        out = []
        start = 0.0
        while start < duration - 1e-6:
            end = min(duration, start + window_seconds)
            out.append((round(start, 3), round(end, 3)))
            start = end
        return out
    spacing = (duration - window_seconds) / (max_windows - 1)
    return [(round(i * spacing, 3), round(i * spacing + window_seconds, 3)) for i in range(max_windows)]


def evidence_for_segment(transcript: str, target_text: str, reference_text: str,
                         *, include_text=False) -> dict:
    tokens = word_tokens(transcript)
    target = set(_sung_tokens(target_text))
    reference = set(_sung_tokens(reference_text))
    target_only = target - reference
    reference_only = reference - target
    overlap = target & reference
    counts = Counter(tokens)
    info = {
        'recognized_words': len(tokens),
        'target_exclusive_hits': sum(counts[x] for x in target_only),
        'reference_exclusive_hits': sum(counts[x] for x in reference_only),
        'ambiguous_shared_vocabulary_hits': sum(counts[x] for x in overlap),
        'out_of_both_vocab_hits': sum(n for w, n in counts.items() if w not in target | reference),
        'transcription_sha256': sha256_text(transcript),
        'transcription_word_sequence_sha256': sha256_text('\u241f'.join(tokens)),
    }
    if include_text:
        info['recognized_excerpt'] = ' '.join(tokens[:50])
        info['reference_exclusive_words_sample'] = [w for w in tokens if w in reference_only][:20]
        info['out_of_vocab_words_sample'] = [w for w in tokens if w not in target | reference][:20]
    return info


def compact_report(original: dict, *, include_text=False) -> dict:
    """Export useful evidence, not the huge syllable-by-syllable log or raw audio."""
    from pathlib import Path
    stages = original.get('stages', {})
    score = stages.get('score_alignment', {})
    auto = score.get('automatic_target_score', {})
    ar = stages.get('ar_prefix', {})
    ref_status = dict(stages.get('reference_lyrics', {}))
    if ref_status.get('model_path'):
        ref_status['model_path'] = Path(ref_status['model_path']).name.replace('\\','/').split('/')[-1]
    tokenizer = stages.get('mert_and_tokenizer', {})
    tokenizer_brief = {k:tokenizer.get(k) for k in (
        'stage','status','reference_frames','reference_seconds','mert_model','tokenizer_name','tokenizer','runtime')}
    segment_asr = original.get('diagnostic_segment_asr', {})
    segment_asr = dict(segment_asr)
    if segment_asr.get('model'):
        segment_asr['model'] = str(segment_asr['model']).replace('\\','/').split('/')[-1]
    return {
        'schema': SCHEMA_VERSION,
        'execution': original.get('execution'),
        'seed': original.get('seed'),
        'selected_models': original.get('selected_models', {}),
        'reference_lyrics_provenance': ref_status,
        'lexical_inputs': score.get('diagnostic_lexical_inputs', {}),
        'actual_encoder_payload': ar.get('diagnostic_encoder_payload', {}),
        'semantic_prefix': {
            'prefix_tokens': ar.get('prefix_tokens'),
            'reference_prefix_seconds': ar.get('reference_prefix_seconds'),
            'reference_tokenizer': tokenizer_brief,
            'ar_generation': {k: ar.get(k) for k in (
                'mode', 'target_duration_budget_seconds', 'combined_duration_budget_seconds',
                'target_score_nominal_seconds', 'actual_target_seconds', 'minimum_accepted_target_seconds',
                'semantic_truncated')},
        },
        'reference_strategy': original.get('reference_strategy', {}),
        'score': {
            'reference_original_bpm': score.get('reference_original_bpm'),
            'target_bpm': score.get('target_bpm'),
            'tempo_stretch_rate': score.get('tempo_stretch_rate'),
            'reference_codec_frames': score.get('reference_codec_frames'),
            'reference_prefix_seconds': score.get('reference_prefix_seconds'),
            'reference_target_lyrics_strictly_concatenated': score.get('reference_target_lyrics_strictly_concatenated'),
            'composite_target_verified': score.get('composite_target_verified'),
            'reference_words': score.get('reference_words'),
            'target_words': score.get('target_words'),
            'target_nominal_seconds': score.get('target_nominal_seconds'),
            'source_duration_seconds': (auto.get('vocal_melody_alignment') or {}).get('source_duration_s'),
            'score_changed': (auto.get('vocal_melody_alignment') or {}).get('score_changed'),
            'native_section_contract': (auto.get('lyrics') or {}).get('native_section_contract', {}),
            'duration_fit': {k:v for k,v in
                ((auto.get('vocal_melody_alignment') or {}).get('duration_fit') or {}).items()
                if k not in ('syllable_mapping',)},
            'alignment_mode': (auto.get('vocal_melody_alignment') or {}).get('mode'),
        },
        'audio_roles': original.get('diagnostic_audio_roles', {}),
        'audio_output': original.get('output_audio', {}),
        'full_output_asr': {k: v for k, v in original.get('lyrics_fidelity_audit', {}).items()
                            if k not in ('possible_reference_contamination_sample',
                                         'possible_source_contamination_sample',
                                         'potential_unrequested_words_sample')},
        'segment_asr': segment_asr,
        'diagnosis': original.get('diagnostic_findings', []),
        'limitations': 'ASR singing accuracy is not reliable enough to prove which words the model sang.',
    }


def diagnosis(trace: dict) -> list[dict]:
    findings = []
    lex = trace.get('lexical_inputs', {})
    enc = trace.get('actual_encoder_payload', {})
    if lex and not lex.get('all_words_in_order', True):
        findings.append({'severity':'error', 'code':'COMPOSITE_LEXICAL_BOUNDARY_MISMATCH',
                         'meaning':'Target/reference word order changed before CLIP encoding.'})
    if enc and enc.get('plan_composite_hash_matches_encoder') is False:
        findings.append({'severity':'error','code':'ENCODER_PAYLOAD_MISMATCH',
                         'meaning':'The text passed to clip.tokenize differs from the composed text.'})
    generation = trace.get('semantic_prefix', {}).get('ar_generation', {})
    if generation.get('semantic_truncated') is True:
        findings.append({'severity':'warning','code':'SEMANTIC_GENERATION_HIT_TOKEN_CEILING',
                         'meaning':'The native YuE2 autoregressive generator reached its token ceiling without MUSIC_END; do not treat this as a natural ending.'})
    # Compare source-clip length with *normalized* semantic prefix duration.
    # This is a timeline discrepancy, not proof that the gap is digital silence:
    # tempo and first-note alignment can account for some of the difference.
    selection=trace.get('reference_strategy', {}).get('clip', {})
    prefix_duration=trace.get('semantic_prefix', {}).get('reference_prefix_seconds')
    selected_duration=selection.get('selected_seconds')
    if isinstance(prefix_duration, (int, float)) and isinstance(selected_duration, (int, float)):
        gap=round(float(prefix_duration)-float(selected_duration),3)
        if gap>=2.0:
            findings.append({'severity':'warning','code':'REFERENCE_PREFIX_LONGER_THAN_SELECTED_AUDIO',
                             'meaning':f'Native semantic prefix ({prefix_duration:.2f}s) exceeds selected reference clip ({selected_duration:.2f}s) by {gap:.2f}s. Inspect score normalization and padding; this does not prove silence.'})
    segments = trace.get('segment_asr', {}).get('windows', [])
    ref_hits = sum(s.get('reference_exclusive_hits', 0) for s in segments)
    tgt_hits = sum(s.get('target_exclusive_hits', 0) for s in segments)
    if segments and ref_hits > max(3, 2 * tgt_hits):
        findings.append({'severity':'warning','code':'REFERENCE_LIKE_ASR_TOKENS',
                         'meaning':'Segmented ASR recognized more reference-exclusive than target-exclusive words; not proof of leakage.'})
    if trace.get('reference_strategy', {}).get('strong_reference_lexical_signal'):
        findings.append({'severity':'error', 'code':'REFERENCE_LYRICS_CONTAMINATION_SUSPECTED',
                         'meaning':'Segmented Whisper strongly favors reference-only words; lexical fidelity failed heuristic gate. Requires listening.'})
    if segments and tgt_hits == 0 and sum(s.get('recognized_words',0) for s in segments) >= 8:
        findings.append({'severity':'warning','code':'TARGET_EXCLUSIVE_WORDS_NOT_RECOGNIZED',
                         'meaning':'ASR found no target-only vocabulary in sampled audio; could be transcription failure or lexical drift.'})
    if segments and all(s.get('recognized_words', 0) <= 2 for s in segments):
        findings.append({'severity':'warning','code':'SEGMENTED_ASR_NEAR_EMPTY',
                         'meaning':'Whisper recognized almost nothing in all sampled windows: investigate instrumental output, model ASR and voice audibility.'})
    roles = trace.get('audio_roles', {})
    source = roles.get('source', {})
    reference = roles.get('reference', {})
    if source.get('partial_pcm_sha256') and source.get('partial_pcm_sha256') == reference.get('partial_pcm_sha256'):
        findings.append({'severity':'info','code':'SOURCE_AND_REFERENCE_AUDIO_MATCH',
                         'meaning':'Source and Reference Voice audio sample fingerprints match. Confirm this is intentional.'})
    if not segments:
        findings.append({'severity':'info','code':'TRY_SEGMENTED_ASR',
                         'meaning':'Set Reference Voice diagnostic_mode=segmented for local time-window ASR evidence.'})
    findings.append({'severity':'info','code':'PREFIX_LEAKAGE_NOT_DETERMINISTICALLY_EXCLUDABLE',
                     'meaning':'Forced semantic reference tokens remain in AR history. Input separation does not guarantee generated lyric separation.'})
    return findings


def readable_summary(trace: dict) -> str:
    lex = trace.get('lexical_inputs', {})
    enc = trace.get('actual_encoder_payload', {})
    audit = trace.get('full_output_asr', {})
    ar = trace.get('semantic_prefix', {})
    lines = [
        'YuE2 Reference Voice — Diagnostic Trace (no audio included)',
        'Schema: ' + trace.get('schema', 'unknown'),
        'Run seed: ' + str(trace.get('seed', 'unknown')),
        'Requested target words: ' + str(lex.get('target', {}).get('sung_words')),
        'Reference words: ' + str(lex.get('reference', {}).get('sung_words')),
        'Composite input boundary valid: ' + str(lex.get('all_words_in_order')),
        'Composed text hash matches actual CLIP encoder payload: ' + str(enc.get('plan_composite_hash_matches_encoder')),
        'AR mode: ' + str(ar.get('ar_generation', {}).get('mode')),
        'Reference prefix seconds: ' + str(ar.get('reference_prefix_seconds')),
        'Generated target seconds: ' + str(ar.get('ar_generation', {}).get('actual_target_seconds')),
        'Whisper output recognized words: ' + str(audit.get('recognized_words')),
        'Whisper output word error rate: ' + str(audit.get('word_error_rate')),
        'Temporal ASR windows: ' + str(len(trace.get('segment_asr', {}).get('windows', []))),
        '', 'FINDINGS:',
    ]
    for f in trace.get('diagnosis', []):
        lines.append('- [' + f['severity'] + '] ' + f['code'] + ': ' + f['meaning'])
    lines += ['', 'ASR is probabilistic, especially on sung mixed audio. No binary audio is in this package.']
    return '\n'.join(lines) + '\n'


def export_bundle(trace: dict, *, output_root=None) -> dict:
    """Atomic ZIP save. Always leave the AUDIO output functional on disk errors."""
    try:
        if output_root is None:
            import folder_paths
            output_root = folder_paths.get_output_directory()
        folder = Path(output_root) / 'yue2_reference_diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        name = 'yue2_reference_trace_' + stamp + '_' + uuid.uuid4().hex[:10]
        destination = folder / (name + '.zip')
        tmp = folder / ('.' + name + '.tmp')
        with zipfile.ZipFile(tmp, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('reference_trace.json', json.dumps(trace, ensure_ascii=False, indent=2, default=str))
            z.writestr('DIAGNOSI.txt', readable_summary(trace))
        os.replace(tmp, destination)
        return {'status':'saved', 'path':str(destination), 'bytes':destination.stat().st_size,
                'contains_audio':False, 'files':['reference_trace.json', 'DIAGNOSI.txt']}
    except Exception as exc:
        return {'status':'not_saved', 'reason':type(exc).__name__ + ': ' + str(exc), 'contains_audio':False}
