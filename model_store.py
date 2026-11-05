"""Single ComfyUI model manager. Model directories contain no executable Python files.

The joint-v9 tokenizer inference implementation is bundled with this custom
node. Only MERT's separately published model architecture is retrieved via
the same model manager; it is stored in the vendor code tree, never models/.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HUB_NATIVE = 'Comfy-Org/YuE2'
HUB_MERT = 'm-a-p/MERT-v2-FullSong'
HUB_TOKENIZER = 'Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4'
TOKENIZER_REVISION = 'e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5'
# The MERT-v2-FullSong repo's initial release; keep code and weights on same revision.
MERT_REVISION = 'd8ba1c7'
NATIVE_CKPTS = ('yue2_3b_bf16.safetensors', 'yue2_3b_int8_convrot.safetensors')
NATIVE_SAGE = 'sheetsage2_bf16.safetensors'
NATIVE_NAR = 'nar_lora_joint_v9_comfyui.safetensors'
HEAD_NAME = 'tokenizer_head_joint_v9.safetensors'
TOKENIZER_SCRIPTS = ('prep_real.py', 'ar_prep.py', 'ckpt_io.py')
MERT_SCRIPTS = ('configuration_mert2.py', 'modeling_mert2.py')
MERT_WEIGHTS = ('config.json', 'preprocessor_config.json', 'model.safetensors')
HASHES = {
    'prep_real.py': '961b3d826a06f05836a0b8a435f8a4af811ac561356440545800e7597e9431cf',
    'ar_prep.py': '952c85bff9c49939564120cd0064573706f68eec0abc5219dab1ee26092db775',
    'ckpt_io.py': '44d48fc08aa7c4a2e0f3120adbe4a05c5f6e1ec54442153bd8a7bc00a24a0e9a',
    HEAD_NAME: '06440f25605c6c12c4e517ef033d7b70b8d9e961b3b06aa3848b1f8028fd7eb8',
}
VENDOR_ROOT = Path(__file__).resolve().parent / 'vendor' / 'model_code'


def comfy_models_dir():
    try:
        import folder_paths
        return Path(folder_paths.models_dir)
    except ImportError:
        return Path(__file__).resolve().parent / 'models'


def locations(root=None, code_root=None):
    base = Path(root) if root is not None else comfy_models_dir()
    code = Path(code_root) if code_root is not None else VENDOR_ROOT
    return {
        'root': base,
        'checkpoints': base / 'checkpoints',
        'audio_encoders': base / 'audio_encoders',
        'mert': base / 'audio_encoders' / 'MERT-v2-FullSong.safetensors',
        'legacy_mert_dir': base / 'audio_encoders' / 'MERT-v2-FullSong',
        'loras': base / 'loras',
        'nar_lora': base / 'loras' / NATIVE_NAR,
        'head': base / 'audio_encoders',
        'legacy_head': base / 'yue2_reference_voice' / 'realaudio-tokenizer',
        'whisper': base / 'whisper',
        'code': code,
        'mert_scripts': code / 'mert2',
    }


def checksum(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def valid_tokenizer_file(path, name, verify_hashes=True):
    return (path.is_file() and path.stat().st_size > 10 and
            (not verify_hashes or checksum(path) == HASHES[name]))


def ready_mert_weights(path):
    path=Path(path)
    if path.is_file() and path.suffix.lower()=='.safetensors':
        return path.stat().st_size > 100_000_000
    return (path.is_dir() and all((path / f).is_file() for f in MERT_WEIGHTS) and
            (path / 'model.safetensors').stat().st_size > 100_000_000)



def valid_mert_code(path):
    """Basic parser/class sanity check. Source origin is pinned by HF revision."""
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    try:
        tree=ast.parse(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, SyntaxError):
        return False
    classes={n.name for n in tree.body if isinstance(n,ast.ClassDef)}
    expected='MERT2Config' if path.name=='configuration_mert2.py' else 'MERT2Model'
    return expected in classes


def _status(loc, verify_hashes):
    ckpts = [loc['checkpoints'] / x for x in NATIVE_CKPTS]
    found = [p.name for p in ckpts if p.is_file() and p.stat().st_size > 1_000_000]
    head = loc['head'] / HEAD_NAME
    head_candidates = ([p for p in loc['audio_encoders'].rglob('tokenizer_head_joint_v9*.safetensors')
                        if p.is_file() and p.stat().st_size > 10_000_000]
                       if loc['audio_encoders'].is_dir() else [])
    # The published v9 fp32 head has a pinned SHA. The compatible BF16
    # release is detected by name+size; expose that weaker check in the report.
    compatible_heads = [p for p in head_candidates if
                        (p.name != HEAD_NAME or valid_tokenizer_file(p, HEAD_NAME, verify_hashes))]
    lora_candidates = ([p for p in loc['loras'].rglob(NATIVE_NAR)
                        if p.is_file() and p.stat().st_size > 10_000_000]
                       if loc['loras'].is_dir() else [])
    mert_files = ([p for p in loc['audio_encoders'].rglob('*.safetensors')
                   if 'mert' in p.name.lower() and p.is_file() and p.stat().st_size>100_000_000]
                  if loc['audio_encoders'].is_dir() else [])
    mert_folders = ([d for d in loc['audio_encoders'].rglob('*')
                     if d.is_dir() and 'mert' in d.name.lower() and ready_mert_weights(d)]
                    if loc['audio_encoders'].is_dir() else [])
    whisper_files = ([p for p in loc['whisper'].rglob('*')
                      if p.is_file() and p.suffix.lower() in ('.pt','.safetensors')]
                     if loc['whisper'].is_dir() else [])
    mert_code = {f:valid_mert_code(loc['mert_scripts'] / f) for f in MERT_SCRIPTS}
    return {
        'YuE2 checkpoint': {'ready': bool(found), 'path':str(loc['checkpoints']), 'found':found},
        'SheetSage2 encoder': {'ready':(loc['audio_encoders'] / NATIVE_SAGE).is_file(),
                               'path':str(loc['audio_encoders'] / NATIVE_SAGE)},
        'MERT-v2-FullSong': {'ready':bool(mert_files or mert_folders), 'path':str(loc['audio_encoders']), 'found':[str(p) for p in mert_files+mert_folders]},
        'joint-v9 tokenizer': {'ready':bool(compatible_heads), 'path':str(head),
                               'found':[str(p) for p in compatible_heads],
                               'integrity':'exact pinned SHA for fp32, size/name only for BF16 alternatives'},
        'joint-v9 ComfyUI NAR LoRA': {'ready':bool(lora_candidates),
                                      'path':str(loc['nar_lora']),
                                      'found':[str(p) for p in lora_candidates]},
        'MERT Python (custom node)': {'ready':all(mert_code.values()), 'path':str(loc['mert_scripts']),
                                      'files':mert_code, 'kind':'code'},
        'Whisper (optional, blank reference lyrics)': {'ready':bool(whisper_files),
                                                       'path':str(loc['whisper']), 'found':[str(p) for p in whisper_files], 'optional':True},
        'Whisper processor metadata (HF safetensors)': {'ready':(VENDOR_ROOT/'whisper_processor'/'config.json').is_file()
                                                       and (VENDOR_ROOT/'whisper_processor'/'tokenizer.json').is_file(),
                                                       'path':str(VENDOR_ROOT/'whisper_processor'), 'optional':True},
    }


def inventory(root=None, verify_hashes=True, code_root=None):
    loc = locations(root, code_root)
    status = _status(loc, verify_hashes)
    pkg = {'torch':'torch','torchaudio':'torchaudio','transformers':'transformers',
           'numpy':'numpy','scipy':'scipy','soundfile':'soundfile',
           'huggingface_hub':'huggingface_hub','safetensors':'safetensors',
           'librosa':'librosa','whisper':'whisper'}
    def installed(module):
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False
    deps = {name:installed(module) for name,module in pkg.items()}
    return {'models':status,
            'ready':all(x['ready'] for x in status.values() if not x.get('optional')),
            'packages':deps,'root':str(loc['root']),
            'code_root':str(loc['code']),
            'note':f'Pesi in ComfyUI/models; codice Python esclusivamente in {VENDOR_ROOT}.'}


def _migrate_tokenizer_legacy(loc, progress):
    """Move only the known, verified v9 weights into shared audio_encoders.

    Never overwrite an existing model, and never move/download Python files
    to the models tree. Known legacy scripts may be removed only when their
    checksums match the pinned historical releases.
    """
    old_weights = loc['legacy_head'] / HEAD_NAME
    new_weights = loc['head'] / HEAD_NAME
    if (old_weights.is_file() and not new_weights.exists() and
            valid_tokenizer_file(old_weights, HEAD_NAME, verify_hashes=True)):
        new_weights.parent.mkdir(parents=True, exist_ok=True)
        old_weights.replace(new_weights)
        progress(f'[YuE2 Models] Moved verified tokenizer weights to: {new_weights}')
    old = loc['legacy_head'] / 'scripts'
    if not old.is_dir():
        return
    for name in TOKENIZER_SCRIPTS:
        legacy=old / name
        if not legacy.is_file() or checksum(legacy)!=HASHES[name]:
            continue
        legacy.unlink()
        progress(f'[YuE2 Models] Removed verified, obsolete tokenizer script: {name} (native codec engine bundled)')
    if not any(old.iterdir()):
        old.rmdir()


def _migrate_mert_legacy(loc, progress):
    """Remove legacy MERT scripts ONLY if identical to the pinned vendored copy.

    Never import or copy unverified code from a models directory into executable code.
    """
    old=loc['legacy_mert_dir']
    dest=loc['mert_scripts']
    if not old.exists():
        return
    for name in MERT_SCRIPTS:
        src=old/name; dst=dest/name
        if src.is_file() and dst.is_file() and checksum(src)==checksum(dst):
            src.unlink()
            progress(f'[YuE2 Models] Removed duplicate MERT script from weights folder: {name}')



def ensure_mert_support(*, code_root=None, progress=print):
    """Install *architecture support only* beside this custom node, if absent.

    Does not download MERT model weights, replace existing code, or write
    executable Python under models/. Uses the same Hugging Face revision as
    the full-song checkpoint. Safe to invoke before generation.
    """
    dest=Path(code_root) if code_root is not None else VENDOR_ROOT
    dest=dest/'mert2'
    code={name:dest/name for name in MERT_SCRIPTS}
    metadata=('config.json','preprocessor_config.json')
    required=list(code)+list(metadata)
    missing=[name for name in required if not ((valid_mert_code(dest/name)) if name in code else (dest/name).is_file())]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError('MERT support missing: install the requirements.txt of this custom node with ComfyUI Python') from exc
        dest.mkdir(parents=True,exist_ok=True)
        progress('[YuE2 Reference Voice] Installing MERT model architecture into custom node (not models): '+', '.join(missing))
        errors=[]
        for name in missing:
            try:
                hf_hub_download(repo_id=HUB_MERT, revision=MERT_REVISION,
                                filename=name, local_dir=str(dest))
            except Exception as exc:
                errors.append(f'{name}: {type(exc).__name__}: {exc}')
        if errors:
            raise RuntimeError('MERT support could not be installed automatically. '
                               'Check Hugging Face access, then use YuE2ReferenceVoiceModels → download_missing. '
                               +' | '.join(errors))
    invalid=[name for name,path in code.items() if not valid_mert_code(path)]
    if invalid:
        raise RuntimeError('MERT model architecture files are missing/corrupt inside custom node: '+', '.join(invalid))
    return code


def ensure_whisper_processor(*, progress=print):
    """Fetch only lightweight Whisper HF tokenizer/config metadata, never weights."""
    dest=VENDOR_ROOT/'whisper_processor'
    required=('config.json','tokenizer.json','preprocessor_config.json')
    if not all((dest/x).is_file() for x in required):
        try:
            from huggingface_hub import snapshot_download
            progress('[YuE2 Reference Voice] Installing Whisper processor metadata inside custom node (no model weights)')
            snapshot_download(repo_id='openai/whisper-large-v3-turbo',
                    allow_patterns=['config.json','generation_config.json','preprocessor_config.json',
                                    'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                                    'normalizer.json','vocab.json','merges.txt','added_tokens.json'],
                    local_dir=str(dest))
        except Exception as exc:
            raise RuntimeError(f'Whisper processor metadata is missing and automatic installation failed: {exc}') from exc
    absent=[x for x in required if not (dest/x).is_file()]
    if absent:
        raise RuntimeError('Whisper metadata missing in custom node: '+', '.join(absent))
    return dest

def download_missing(root=None, include_whisper=False, progress=print, code_root=None):
    """Only the custom node's manager downloads resources, including model-specific code.

    Model-specific `.py` is always stored under this custom node, never under ComfyUI/models.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(f'Install {Path(__file__).resolve().parent / "requirements.txt"} in ComfyUI Python.') from exc
    loc=locations(root,code_root)
    errors=[]
    def attempt(label, action):
        try:
            progress('[YuE2 Models] '+label)
            action()
        except Exception as exc:
            msg=f'{label}: {type(exc).__name__}: {exc}'
            errors.append(msg);progress('[YuE2 Models][ERROR] '+msg)

    _migrate_tokenizer_legacy(loc,progress)
    state=_status(loc,False)
    if not state['YuE2 checkpoint']['ready']:
        attempt('YuE2 BF16', lambda:hf_hub_download(repo_id=HUB_NATIVE,
            filename='checkpoints/yue2_3b_bf16.safetensors',local_dir=str(loc['root'])))
    if not state['SheetSage2 encoder']['ready']:
        attempt('SheetSage2', lambda:hf_hub_download(repo_id=HUB_NATIVE,
            filename='audio_encoders/'+NATIVE_SAGE,local_dir=str(loc['root'])))
    if not state['MERT-v2-FullSong']['ready']:
        def download_mert_flat():
            # Obtain the upstream safetensors without storing any executable
            # source code in models; copy only the weight payload to audio_encoders.
            import shutil
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(prefix='yue2-mert-') as tmp:
                src=hf_hub_download(repo_id=HUB_MERT,revision=MERT_REVISION,
                                    filename='model.safetensors',local_dir=tmp)
                loc['mert'].parent.mkdir(parents=True,exist_ok=True)
                if not loc['mert'].exists():
                    shutil.copy2(src,loc['mert'])
        attempt('MERT flat model weights under audio_encoders',download_mert_flat)
    metadata = ('config.json', 'preprocessor_config.json')
    if not all((loc['mert_scripts']/file).is_file() for file in metadata):
        attempt('Official MERT architecture metadata into custom node',
                lambda:snapshot_download(repo_id=HUB_MERT,revision=MERT_REVISION,
                    allow_patterns=list(metadata),local_dir=str(loc['mert_scripts'])))
    for name in MERT_SCRIPTS:
        if not valid_mert_code(loc['mert_scripts']/name):
            def install_mert_code(name=name):
                path=hf_hub_download(repo_id=HUB_MERT,revision=MERT_REVISION,
                                     filename=name,local_dir=str(loc['mert_scripts']))
                if not valid_mert_code(Path(path)):
                    raise ValueError(f'MERT architecture source {name} is invalid')
            attempt('MERT architecture code inside custom node: '+name,install_mert_code)
    head=loc['head']/HEAD_NAME
    if not state['joint-v9 tokenizer']['ready']:
        attempt('Joint-v9 tokenizer weights',lambda:hf_hub_download(repo_id=HUB_TOKENIZER,
            revision=TOKENIZER_REVISION,filename=HEAD_NAME,local_dir=str(loc['head'])))
    if not state['joint-v9 ComfyUI NAR LoRA']['ready']:
        attempt('NAR joint-v9 ComfyUI LoRA',lambda:hf_hub_download(repo_id=HUB_TOKENIZER,
            filename=NATIVE_NAR,revision=TOKENIZER_REVISION,local_dir=str(loc['loras'])))
    if include_whisper:
        if not state['Whisper (optional, blank reference lyrics)']['ready']:
            def whisper_download():
                import whisper
                loc['whisper'].mkdir(parents=True,exist_ok=True)
                whisper._download(whisper._MODELS['large-v3-turbo'],str(loc['whisper']),False)
            attempt('Whisper large-v3-turbo .pt (only if no Whisper already installed)',whisper_download)
        if not (VENDOR_ROOT/'whisper_processor'/'config.json').is_file() or not (VENDOR_ROOT/'whisper_processor'/'tokenizer.json').is_file():
            def whisper_processor_download():
                snapshot_download(repo_id='openai/whisper-large-v3-turbo',
                    allow_patterns=['config.json','generation_config.json','preprocessor_config.json',
                                    'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                                    'normalizer.json','vocab.json','merges.txt','added_tokens.json'],
                    local_dir=str(VENDOR_ROOT/'whisper_processor'))
            attempt('Whisper processor metadata into custom node (no duplicate weights)',whisper_processor_download)
    _migrate_tokenizer_legacy(loc,progress)
    _migrate_mert_legacy(loc,progress)
    final=inventory(root,code_root=code_root)
    final['errors']=errors
    if errors:
        raise RuntimeError('Some downloads failed:\n'+'\n'.join(errors)+'\n'+json.dumps(final,ensure_ascii=False,default=str)[:2000])
    return final


class YuE2ReferenceVoiceModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'action':(['check','download_missing','download_missing_with_whisper'],{'default':'check'})}}
    RETURN_TYPES=('STRING',);RETURN_NAMES=('report',);FUNCTION='manage'
    CATEGORY='audio/YuE2';OUTPUT_NODE=True
    def manage(self,action='check'):
        result=inventory() if action=='check' else download_missing(include_whisper=action.endswith('with_whisper'))
        from .model_selection import (checkpoint_options, sheetsage_options, mert_options,
                                      tokenizer_options, lora_options, whisper_options)
        result['selectable_models']={
            'YuE2 checkpoints':checkpoint_options(), 'SheetSage2 encoders':sheetsage_options(),
            'MERT models (files/directories)':mert_options(), 'joint-v9 tokenizers':tokenizer_options(),
            'joint-v9 NAR LoRA':lora_options(), 'Whisper':whisper_options(),
        }
        report=json.dumps(result,ensure_ascii=False,indent=2)
        print('[YuE2 Auto Vocal PRO Models]\n'+report)
        return (report,)

NODE_CLASS_MAPPINGS={'YuE2ReferenceVoiceModels':YuE2ReferenceVoiceModels}
NODE_DISPLAY_NAME_MAPPINGS={'YuE2ReferenceVoiceModels':'YuE2 Reference Voice: Check / Download Models'}
