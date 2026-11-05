"""CPU-only, best-effort forensic flight recorder for the existing ComfyUI nodes.

This module wraps registered node FUNCTION methods without changing their return
values, sampling, graph edges, models or output audio.  ZIPs are checkpointed
before model execution and after intermediate steps, including on exceptions.
The optional REPLAY mode contains copyrighted music notation and private Lyrics:
only enable it for a diagnostic run that you intend to share.
"""
from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import pathlib
import platform
import re
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field

_SCHEMA = 'yue2-forensic/2.0'
_LOCK = threading.RLock()
_TLS = threading.local()
_BY_PROGRESS = {}
_RECENT = []
_MAX_RUNS = 32
_MAX_EVENTS = 120
_MAX_SNAPSHOT_BYTES = 2_000_000
_MAX_TOTAL_SNAPSHOTS = 12_000_000
_PATH_RE = re.compile(r'(?i)(?:[A-Z]:\\|/home/|/Users/)[^\s"\']+')
_ALLOWED_TEXT_FIELDS = frozenset({
    'abc', 'source_abc', 'target_abc', 'reference_abc', 'generated_abc',
    'composite_abc', 'prepared_abc', 'lyrics', 'target_lyrics',
    'reference_lyrics', 'composite_lyrics', 'reference_text', 'style',
    'source_report', 'plan_report', 'target_plan_report', 'lyrics_report',
    'ar_report', 'tokenizer_report', 'report', 'execution_report',
    'quality_report', 'auto_score_report', 'diagnostic_plan_report',
    'selected_mode_report', 'reference_clip_report', 'reference_style_report',
    'reason',
})
_SAFE_PUBLIC_STRING_FIELDS = frozenset({
    'ckpt_name', 'audio_encoder_name', 'mert_model', 'tokenizer_name',
    'nar_lora_name', 'whisper_model', 'whisper_mode', 'mode',
    'conditioning_mode', 'reference_clip_policy', 'clip_policy',
    'section_policy', 'instrumental_handling', 'existing_vocal',
    'vocal_melody_mode', 'diagnostic_mode', 'reference_quality_gate',
    'quality_gate', 'target_language', 'stage', 'budget_policy',
})
_SKIP_KEYS = frozenset({'source_audio', 'reference_audio', 'audio', 'clip', 'model',
                        'vae', 'latent', 'semantic_prefix', 'positive', 'negative',
                        'aligned_reference_audio', 'waveform'})
_NATIVE_FILES = (
    'comfy_extras/nodes_yue2.py',
    'comfy/text_encoders/yue2.py',
    'comfy_extras/nodes_audio.py',
    'comfy_extras/nodes_sheetsage.py',
    'comfy_extras/nodes_sheetsage2.py',
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _short_path(s: str) -> str:
    # Never export absolute workstation paths in metadata by default.
    return _PATH_RE.sub(lambda m: '[local]/' + re.split(r'[\\/]', m.group(0))[-1], s)


def _compact(value, depth=0):
    if depth >= 3:
        return {'type': type(value).__name__}
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        bs = value.encode('utf-8', errors='replace')
        return {'type': 'text', 'sha256': sha256(bs), 'bytes': len(bs),
                'lines': value.count('\n')+1, 'text_available': False}
    if isinstance(value, (bytes, bytearray)):
        return {'type': 'binary', 'bytes': len(value), 'sha256': sha256(bytes(value))}
    if isinstance(value, dict):
        if 'waveform' in value:
            wf = value['waveform']
            return {'type':'AUDIO', 'sample_rate':_compact(value.get('sample_rate'), depth+1),
                    'shape':list(getattr(wf,'shape',())), 'dtype':str(getattr(wf,'dtype','unknown'))}
        if 'samples' in value and len(value) <= 5:
            return {'type':'AUDIO_OR_LATENT','keys':sorted(str(x) for x in value)}
        if 'expand' in value and 'result' in value:
            graph = value.get('expand')
            if isinstance(graph,dict):
                items = []
                for key, node in list(graph.items())[:80]:
                    if isinstance(node,dict):
                        items.append({'graph_key':str(key),
                                      'class_type':node.get('class_type'),
                                      'inputs':sorted(str(k) for k in node.get('inputs',{}))})
                return {'type':'DYNAMIC_GRAPH', 'node_count':len(graph),
                        'nodes':items,'result_count':len(value.get('result',()))}
            return {'type':'DYNAMIC_GRAPH','expand_type':type(graph).__name__}
        return {'type':'dictionary','keys':sorted(str(k) for k in value)[:60],
                'details':{str(k):_compact(v,depth+1) for k,v in list(value.items())[:18]
                           if k in ('status','reason','mode','decision','score_source','score_durations',
                                    'semantic_truncated','diagnostic_bundle','sampling_mode')}}
    if isinstance(value, (list, tuple)):
        return {'type':type(value).__name__, 'count':len(value),
                'items':[_compact(x,depth+1) for x in value[:5]]}
    return {'type':type(value).__name__,
            'shape':list(getattr(value,'shape',())) if hasattr(value,'shape') else None}


def _text_stats(label: str, content: str) -> dict:
    stats = _compact(content)
    if label.endswith('abc') or label == 'abc':
        stats['abc_metadata'] = {
            'groups': sum(bool(re.match(r'^\s*%\s*(intro|verse|chorus|bridge|interlude|outro|instrumental)',l,re.I)) for l in content.splitlines()),
            'vocal_labels': content.count('V: Vocal'),
            'instrumental_labels': content.count('V: Ins'),
            'estimated_bar_lines': sum(l.count('|') for l in content.splitlines()),
            'has_ties': '-' in content,
        }
    if 'lyrics' in label or label == 'reference_text':
        stats['lyrics_metadata'] = {
            'section_markers': len(re.findall(r'(?m)^\s*\[[^\]]+\]', content)),
            'word_count_proxy': len(re.findall(r'\S+', re.sub(r'(?m)^\s*\[[^\]]+\]', '', content))),
        }
    return stats


def _info_snapshot(trace, node: str, direction: str, key: str, value):
    label = str(key).lower()
    if label in _SKIP_KEYS or not isinstance(value, str):
        return _compact(value)
    info = _text_stats(label, value)
    if label in _SAFE_PUBLIC_STRING_FIELDS:
        # Safe enums and model *names*, never absolute filesystem paths.
        info['selected_value'] = _short_path(value)[:180]
    if not trace.replay or label not in _ALLOWED_TEXT_FIELDS:
        return info
    data = value.encode('utf-8',errors='replace')
    total = sum(len(v) for v in trace.snapshots.values())
    if len(data) > _MAX_SNAPSHOT_BYTES or total + len(data) > _MAX_TOTAL_SNAPSHOTS:
        info['replay_missing_reason'] = 'size_limit_exceeded'
        trace.replay_complete = False
        return info
    stage = len(trace.events)
    ext = '.abc' if 'abc' in label else '.json' if ('report' in label or label == 'reason') else '.txt'
    name = f'snapshots/{stage:03d}_{re.sub("[^a-zA-Z0-9_-]", "_",node)}_{direction}_{label}{ext}'
    trace.snapshots[name] = data
    info['snapshot'] = name
    info['text_available'] = True
    return info


def _environment():
    result = {'python':platform.python_version(), 'platform':platform.system(),
              'plugin_schema':_SCHEMA}
    for name in ('torch','transformers','comfyui-frontend-package','safetensors'):
        try: result['package_'+name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return result


def _native_sources(trace):
    """Identify actual installed official node *code*, never checkpoint weights.

    ComfyUI may move definitions between versions. Discover by the class name
    within the official comfy_extras directory, instead of assuming master.
    """
    base = pathlib.Path(__file__).resolve().parents[2]
    candidates = [base/rel for rel in _NATIVE_FILES]
    extras=base/'comfy_extras'
    if extras.is_dir():
        for f in extras.glob('*.py'):
            if f in candidates or not f.is_file(): continue
            try:
                if f.stat().st_size > 900_000: continue
                data=f.read_bytes()
                if any(marker in data for marker in (
                    b'class YuE2GenerateABC', b'class YuE2GenerateMusic',
                    b'class SheetSage2AudioToABC', b'class YuE2TEModel_')):
                    candidates.append(f)
            except OSError: continue
    result = []
    for f in candidates:
        if not f.is_file(): continue
        rel=f.relative_to(base).as_posix()
        try:
            content = f.read_bytes()
            entry = {'path_relative_to_ComfyUI':rel, 'sha256':sha256(content), 'bytes':len(content)}
            if trace.replay and len(content) <= 900_000:
                filename='official_sources/'+rel
                trace.snapshots[filename] = content
                entry['embedded_source']=filename
            result.append(entry)
        except OSError as exc:
            result.append({'path_relative_to_ComfyUI':rel,'status':'unreadable',
                           'error_type':type(exc).__name__})
    plugin_dir=pathlib.Path(__file__).resolve().parent
    for f in sorted(plugin_dir.glob('*.py')):
        try:
            content=f.read_bytes()
            entry={'path_relative_to_plugin':f.name,'sha256':sha256(content),'bytes':len(content)}
            if trace.replay and len(content)<=900_000:
                filename='custom_node_sources/'+f.name
                trace.snapshots[filename]=content
                entry['embedded_source']=filename
            result.append(entry)
        except OSError: continue
    return result


def _output_dir():
    target = os.getenv('YUE2_FORENSIC_OUTPUT')
    if target:
        return pathlib.Path(target)
    try:
        import folder_paths
        return pathlib.Path(folder_paths.get_output_directory()) / 'yue2_forensic'
    except ImportError:
        import tempfile
        return pathlib.Path(tempfile.gettempdir())/'yue2_forensic'


@dataclass
class Trace:
    id: str
    replay: bool = False
    started: str = field(default_factory=lambda:_dt.datetime.now(_dt.timezone.utc).isoformat())
    events: list = field(default_factory=list)
    snapshots: dict = field(default_factory=dict)
    replay_complete: bool = True
    error: dict | None = None
    status: str = 'running'
    latest_zip: str = ''
    native_files: list = field(default_factory=list)
    _recorded_environment: dict = field(default_factory=_environment)
    _output_dir: str = ''


def _start(replay=False):
    run = Trace(uuid.uuid4().hex[:16], bool(replay))
    run._output_dir = str(_output_dir())
    run.native_files = _native_sources(run)
    _RECENT.append(run)
    if len(_RECENT)>_MAX_RUNS:
        _RECENT.pop(0)
    _TLS.current = run
    return run


def attach_progress(progress_id: str):
    """Bind GraphBuilder nodes to their parent forensics after start_run()."""
    with _LOCK:
        run = getattr(_TLS,'current',None)
        if run is not None and progress_id:
            _BY_PROGRESS[str(progress_id)] = run
            if len(_BY_PROGRESS) > 128:
                _BY_PROGRESS.clear()
                _BY_PROGRESS[str(progress_id)] = run


def _choose_run(node, named):
    progress = str(named.get('progress_node_id') or '')
    replay = bool(named.get('forensic_replay', False))
    with _LOCK:
        if node == 'YuE2ReferenceVoiceCover':
            return _start(replay)
        run = _BY_PROGRESS.get(progress) if progress else None
        if node == 'YuE2AutoVocalPRO' and not named.get('reference_composite') and not named.get('reference_sample_only'):
            if not progress or run is None:
                return _start(replay)
        if run is None:
            run = getattr(_TLS,'current',None)
        if run is None or run.status == 'failed':
            run = _start(replay)
        if replay and not run.replay:
            run.replay = True
            run.native_files = _native_sources(run)
        _TLS.current = run
        return run


def _error_record(exc):
    return {'type':type(exc).__name__,
            'message':_short_path(str(exc))[:1600],
            'traceback':_short_path(''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)))[:30000]}


def _scrub_error_for_metadata(error: dict, named: dict):
    """Mask literal supplied Lyrics/ABC if a validation exception echoes them."""
    error = dict(error)
    for label, value in named.items():
        if label not in _ALLOWED_TEXT_FIELDS or not isinstance(value, str) or len(value) < 4:
            continue
        digest = sha256(value.encode('utf-8',errors='replace'))[:12]
        replacement='[private '+str(label)+' sha256:'+digest+']'
        for part in ('message','traceback'):
            error[part] = error[part].replace(value,replacement)
    return error


def _dump(trace, *, status=None):
    # Atomic ZIP checkpoint: survives exceptions in later OFFICIAL nodes and
    # is available for user upload without a completed AUDIO result.
    if status:
        trace.status = status
    out = pathlib.Path(trace._output_dir)
    zip_path=out/f'yue2_forensic_{trace.id}.zip'
    temp_path=out/f'.{zip_path.name}.{uuid.uuid4().hex[:8]}.tmp'
    manifest = {
        'schema':_SCHEMA, 'run_id':trace.id, 'started_utc':trace.started,
        'status':trace.status,'replay_requested':trace.replay,
        'symbolic_replay_inputs_captured':trace.replay_complete if trace.replay else False,
        'replay_complete':trace.replay_complete, 'audio_included':False,
        'weights_included':False,'installed_source_code_hashes':trace.native_files,
        'environment':trace._recorded_environment,'events':trace.events,
        'error':trace.error,
        'limitations':[
            'No GPU inference is replayable from this bundle alone.',
            'An exception in an official ComfyUI node may leave this at the last checkpoint.',
            'Word-to-audio fidelity requires separate inference/ASR/listening.',
        ],
    }
    try:
        out.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(temp_path,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            z.writestr('trace.json',json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
            text = (f'YuE2 Forensic\nTrace ID: {trace.id}\nStatus: {trace.status}\n'
                    f'Replay data: {trace.replay} / complete: {trace.replay_complete}\n'
                    f'Events: {len(trace.events)}\n'
                    'No audio or model weights are included.\n')
            if trace.error:
                text += '\nLast error: '+trace.error['type']+': '+trace.error['message']+'\n'
            z.writestr('LEGGIMI.txt',text)
            for filename, content in trace.snapshots.items():
                z.writestr(filename,content)
        os.replace(temp_path, zip_path)
        trace.latest_zip=str(zip_path)
    except OSError as exc:
        try: temp_path.unlink(missing_ok=True)
        except OSError: pass
        print('[YuE2 Forensic] ZIP unavailable: '+type(exc).__name__+' '+str(exc)[:200])
    return trace.latest_zip


def _named(func, args, kwargs):
    try:
        bound = inspect.signature(func).bind_partial(*args,**kwargs)
        bound.apply_defaults()
        return {k:v for k,v in bound.arguments.items() if k!='self'}
    except (ValueError,TypeError):
        return dict(kwargs)


def record_manual(stage, fields, progress_node_id=''):
    """Optional evidence from a stage which is not a ComfyUI registered class."""
    named=dict(fields)
    if progress_node_id: named['progress_node_id']=progress_node_id
    run=_choose_run(stage,named)
    event={'index':len(run.events),'stage':stage,'phase':'marker','inputs':{
        str(k):_info_snapshot(run,stage,'input',k,v) for k,v in named.items()}}
    with _LOCK:
        run.events.append(event)
        _dump(run)
    return run.id


def install(node_classes: dict):
    """Wrap each registered ComfyUI node FUNCTION. Installs once per class."""
    for label, cls in node_classes.items():
        attr=getattr(cls,'FUNCTION',None)
        if not attr: continue
        old=getattr(cls,attr,None)
        if old is None or getattr(old,'__yue2_forensic_wrapped__',False):
            continue
        def make_wrapper(method, node, node_class):
            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                # The recorder must NEVER introduce a new production failure.
                trace = None
                event = None
                try:
                    named = _named(method, args, kwargs)
                    trace = _choose_run(node, named)
                    event = {'index':len(trace.events),'stage':node,
                             'method':getattr(method,'__name__',''),
                             'phase':'started','inputs':{},
                             'time_utc':_dt.datetime.now(_dt.timezone.utc).isoformat()}
                    with _LOCK:
                        for key, value in named.items():
                            event['inputs'][str(key)] = _info_snapshot(trace,node,'input',str(key),value)
                        trace.events.append(event)
                        _dump(trace)
                except Exception as diagnostic_exc:
                    print('[YuE2 Forensic] Non-fatal recorder start failure: '+str(diagnostic_exc)[:200])
                    trace = None
                    event = None
                started = time.monotonic()
                try:
                    response = method(*args, **kwargs)
                except BaseException as exc:
                    if trace is not None and event is not None:
                        try:
                            with _LOCK:
                                event['phase'] = 'failed'
                                event['duration_cpu_seconds'] = round(time.monotonic()-started,6)
                                trace.error = _error_record(exc)
                                if not trace.replay:
                                    trace.error = _scrub_error_for_metadata(trace.error,named)
                                _dump(trace,status='failed')
                            print('[YuE2 Forensic][ERROR ZIP] '+str(trace.latest_zip))
                        except Exception as diagnostic_exc:
                            print('[YuE2 Forensic] Non-fatal recorder error failure: '+str(diagnostic_exc)[:200])
                    raise  # Preserve original exception and traceback.
                if trace is not None and event is not None:
                    try:
                        result_names = getattr(node_class,'RETURN_NAMES',None)
                        if not result_names:
                            result_names = tuple(f'output_{i}' for i in range(len(getattr(node_class,'RETURN_TYPES',()))))
                        if isinstance(response,dict) and 'result' in response:
                            values = response.get('result',())
                        elif isinstance(response,(tuple,list)):
                            values = response
                        else:
                            values = (response,)
                        with _LOCK:
                            event['phase'] = 'completed'
                            event['duration_cpu_seconds'] = round(time.monotonic()-started,6)
                            event['outputs'] = {
                                str(result_names[i]) if i<len(result_names) else f'output_{i}':
                                _info_snapshot(trace,node,'output',str(result_names[i]) if i<len(result_names) else f'output_{i}',item)
                                for i,item in enumerate(values)
                            }
                            if isinstance(response,dict) and 'expand' in response:
                                event['dynamic_graph'] = _compact(response)
                            if node in ('YuE2NativeExecutionReport','YuE2VocalFidelityAudit'):
                                trace.status = 'completed'
                            _dump(trace)
                    except Exception as diagnostic_exc:
                        print('[YuE2 Forensic] Non-fatal recorder result failure: '+str(diagnostic_exc)[:200])
                return response
            wrapped.__yue2_forensic_wrapped__ = True
            return wrapped
        setattr(cls,attr,make_wrapper(old,label,cls))


def _reset_for_tests():
    with _LOCK:
        _BY_PROGRESS.clear();_RECENT.clear()
        if hasattr(_TLS,'current'):
            delattr(_TLS,'current')
