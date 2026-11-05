"""Choose actual ComfyUI files, not hard-coded model folder names.

MERT 2 FullSong can be a single safetensors file OR an upstream directory.
Tokenizer and MERT enumerate audio_encoders, NAR LoRA enumerates loras,
Whisper enumerates whisper. All paths honor extra_model_paths.yaml.
"""
from __future__ import annotations
from pathlib import Path
import json
import re
from .model_store import NATIVE_SAGE, HEAD_NAME, NATIVE_NAR, MERT_WEIGHTS, MERT_SCRIPTS, valid_mert_code, VENDOR_ROOT, locations


class MissingModelError(RuntimeError):
    pass


def _category_roots(category):
    roots=[]
    try:
        import folder_paths
        roots.extend(Path(p) for p in folder_paths.get_folder_paths(category))
    except (ImportError, KeyError, ValueError, AttributeError):
        pass
    loc=locations()
    roots.append(loc.get(category, loc['root']/category))
    return list(dict.fromkeys(p.resolve() for p in roots))


def _is_safe_name(name):
    value=str(name)
    p=Path(value)
    return bool(value) and not p.is_absolute() and not any(x in ('..','.') for x in p.parts) and '\\' not in value and ':' not in value


def _model_files(category, extensions=('.safetensors',)):
    names=[]
    try:
        import folder_paths
        names=folder_paths.get_filename_list(category)
    except (ImportError,KeyError,AttributeError,ValueError):
        pass
    for root in _category_roots(category):
        if root.is_dir():
            names.extend(str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*')
                         if p.is_file() and p.suffix.lower() in extensions)
    return sorted({str(n).replace('\\','/') for n in names if str(n).lower().endswith(extensions)})


def _resolve(category,name):
    if not _is_safe_name(name):
        raise MissingModelError(f'Invalid model selection for {category}: {name!r}')
    try:
        import folder_paths
        full=folder_paths.get_full_path(category,name)
        if full and Path(full).is_file():
            return Path(full).resolve()
    except (ImportError,KeyError,AttributeError,ValueError):
        pass
    for root in _category_roots(category):
        path=(root/name).resolve()
        if path.is_relative_to(root) and path.is_file():
            return path
    raise MissingModelError(f'Model {name!r} does not exist under registered ComfyUI {category} folders')


def _option(found, default):
    found=sorted(set(found))
    if not found:
        return ['[Nessun modello trovato: riavvia ComfyUI]']  # never invent a file in the dropdown
    return [default]+[x for x in found if x!=default] if default in found else found


def checkpoint_options():
    return _option((x for x in _model_files('checkpoints') if 'yue2' in x.lower()),'yue2_3b_bf16.safetensors')


def sheetsage_options():
    return _option((x for x in _model_files('audio_encoders') if 'sheetsage' in x.lower()),NATIVE_SAGE)


def tokenizer_options():
    """Exactly the ComfyUI audio_encoders files list, like the LoRA dropdown.

    The selected file is validated against the tokenizer architecture when used;
    the UI never hides installed audio encoders based on their filenames.
    """
    return _option(_model_files('audio_encoders'), HEAD_NAME)


def lora_options():
    # Show actual files from models/loras, not only one pre-selected filename.
    # Incompatibility, if any, is reported when ComfyUI applies the selected LoRA.
    return _option(_model_files('loras'),NATIVE_NAR)


def _looks_like_mert(directory):
    if 'mert' in directory.name.lower():
        return True
    cfg=directory/'config.json'
    try:
        data=json.loads(cfg.read_text(encoding='utf-8'))
        return 'mert' in str(data.get('model_type','')).lower() or any('mert' in str(s).lower() for s in data.get('architectures',[]))
    except (OSError, UnicodeError, ValueError):
        return False


def mert_options():
    """All files in ComfyUI audio_encoders, plus complete legacy model dirs."""
    found=_model_files('audio_encoders')
    for root in _category_roots('audio_encoders'):
        if root.is_dir():
            found.extend(str(d.relative_to(root)).replace('\\','/') for d in root.rglob('*')
                         if d.is_dir() and _looks_like_mert(d) and all((d/k).is_file() for k in MERT_WEIGHTS))
    return _option(found,'MERT-v2-FullSong.safetensors')


def whisper_mode_options():
    """Transcription behaviour, independent from the installed model filename."""
    return ['auto', 'selected', 'none']


def whisper_options():
    """Only real files from models/whisper, analogous to the LoRA selector."""
    found = _model_files('whisper', extensions=('.safetensors', '.pt'))
    return _option(found, 'whisper-large-v3-turbo.safetensors')


def resolve_mert(name):
    if not _is_safe_name(name):
        raise MissingModelError(f'Invalid MERT selection: {name!r}')
    if str(name).lower().endswith('.safetensors'):
        path=_resolve('audio_encoders',name)
        if path.stat().st_size<100_000_000:
            raise MissingModelError(f'MERT weights are incomplete or unusually small: {path}')
        return path
    for root in _category_roots('audio_encoders'):
        directory=(root/name).resolve()
        if directory.is_relative_to(root) and directory.is_dir() and _looks_like_mert(directory):
            missing=[f for f in MERT_WEIGHTS if not (directory/f).is_file()]
            if not missing and (directory/'model.safetensors').stat().st_size>100_000_000:
                return directory
            raise MissingModelError(f'MERT directory {directory} is incomplete: {missing}')
    raise MissingModelError(f'MERT model {name!r} not present under models/audio_encoders. Select the actual .safetensors file.')


def resolve_tokenizer(name):
    if not str(name).lower().endswith('.safetensors'):
        raise MissingModelError('The tokenizer must be a .safetensors file from audio_encoders')
    path=_resolve('audio_encoders',name)
    if path.stat().st_size<10_000_000:
        raise MissingModelError(f'Tokenizer weights appear incomplete: {path}')
    return path


def resolve_nar_lora(name):
    if not str(name).lower().endswith('.safetensors'):
        raise MissingModelError('NAR LoRA must be a ComfyUI .safetensors file')
    path=_resolve('loras',name)
    if path.stat().st_size<10_000:
        raise MissingModelError(f'LoRA appears incomplete: {path}')
    return path


def resolve_whisper(name, *, required=False, mode='auto'):
    """Resolve a local ASR checkpoint without mixing modes into filename choices.

    Legacy programmatic calls resolve_whisper('auto'/'none') remain supported.
    When the user provides reference_lyrics, the caller does not require ASR.
    """
    if name in ('auto', 'none') and mode == 'auto':
        mode, name = name, ''
    if mode not in ('auto', 'selected', 'none'):
        raise MissingModelError(f'Unsupported Whisper mode: {mode!r}')
    if mode == 'none':
        if required:
            raise MissingModelError('Reference lyrics are empty and Whisper mode is none. Enter the reference lyrics or enable Whisper.')
        return None
    choices = [x for x in whisper_options() if not x.startswith('[Nessun modello')]
    if mode == 'selected':
        if name not in choices:
            raise MissingModelError(f'Selected Whisper file {name!r} is not installed under models/whisper')
        chosen = name
    else:
        preference = ('whisper-large-v3-turbo.safetensors', 'large-v3-turbo.pt')
        chosen = next((x for x in preference if x in choices), choices[0] if choices else None)
    if not chosen:
        if required:
            raise MissingModelError('Whisper model not found in models/whisper; enter reference_lyrics or install a model')
        return None
    path = _resolve('whisper', chosen)
    if path.suffix.lower() not in ('.pt', '.safetensors'):
        raise MissingModelError(f'Unsupported Whisper checkpoint extension: {path.suffix}')
    return path


def validate_reference_models(ckpt_name,audio_encoder_name,mert_model,tokenizer_name,nar_lora_name,whisper_model,reference_lyrics,whisper_mode="auto"):
    checkpoint=_resolve('checkpoints',ckpt_name)
    if 'yue2' not in checkpoint.name.lower():
        raise MissingModelError('Select a native YuE2 checkpoint')
    sage=_resolve('audio_encoders',audio_encoder_name)
    if 'sheetsage' not in sage.name.lower():
        raise MissingModelError('Select SheetSage2 audio encoder')
    mert=resolve_mert(mert_model)
    tokenizer=resolve_tokenizer(tokenizer_name)
    lora=resolve_nar_lora(nar_lora_name)
    from .model_store import ensure_mert_support
    try:
        code=ensure_mert_support()
    except RuntimeError as exc:
        raise MissingModelError(str(exc)) from exc
    whisper=resolve_whisper(whisper_model,required=not bool(str(reference_lyrics).strip()),mode=whisper_mode)
    return {'checkpoint':str(checkpoint),'sheetsage':str(sage),'mert':str(mert),
            'tokenizer':str(tokenizer),'nar_lora':str(lora),
            'whisper':str(whisper) if whisper else None,
            'mert_python':{name:str(path) for name,path in code.items()}}
