"""Transcribe locally selected Whisper .pt or .safetensors in ComfyUI/models/whisper.

No runtime selection override and no model weight download.  The OpenAI-native
Whisper format can be loaded directly. HF-format safetensors uses the official
small processor/config files downloaded by the custom node's model manager
into vendor/model_code/whisper_processor (not into models/).
"""
from __future__ import annotations
from pathlib import Path
import gc

PROCESSOR_DIR=Path(__file__).resolve().parent/'vendor'/'model_code'/'whisper_processor'


def sniff_weights(filename):
    from safetensors import safe_open
    with safe_open(str(filename),framework='pt',device='cpu') as f:
        keys=set(f.keys())
        if 'model.encoder.conv1.weight' in keys:
            return 'transformers'
        if 'encoder.conv1.weight' in keys:
            return 'openai'
    raise RuntimeError(f'Unrecognized Whisper safetensors architecture: {filename}. '
                       'Expected Hugging Face or native OpenAI Whisper tensor names.')


def _load_openai_safetensors(path):
    import whisper
    from safetensors import safe_open
    from safetensors.torch import load_model
    import torch
    with safe_open(str(path),framework='pt',device='cpu') as weights:
        def shape(name): return weights.get_slice(name).get_shape()
        audio_state=shape('encoder.conv1.weight')[0]
        n_mels=shape('encoder.conv1.weight')[1]
        audio_ctx=shape('encoder.positional_embedding')[0]
        vocab,text_state=shape('decoder.token_embedding.weight')
        text_ctx=shape('decoder.positional_embedding')[0]
        keys=weights.keys()
        audio_layers=1+max(int(k.split('.')[2]) for k in keys if k.startswith('encoder.blocks.') and k.split('.')[2].isdigit())
        text_layers=1+max(int(k.split('.')[2]) for k in keys if k.startswith('decoder.blocks.') and k.split('.')[2].isdigit())
    if audio_state%64 or text_state%64:
        raise RuntimeError('Selected Whisper has unexpected attention head size; native loader cannot infer dimensions')
    dims=whisper.ModelDimensions(n_mels=n_mels,n_audio_ctx=audio_ctx,
        n_audio_state=audio_state,n_audio_head=audio_state//64,n_audio_layer=audio_layers,
        n_vocab=vocab,n_text_ctx=text_ctx,n_text_state=text_state,
        n_text_head=text_state//64,n_text_layer=text_layers)
    model=whisper.Whisper(dims)
    try:
        load_model(model,str(path),strict=True,device='cpu')
    except Exception as exc:
        del model
        raise RuntimeError(f'Whisper OpenAI safetensors incompatibility: {path}: {exc}') from exc
    return model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()


def _transcribe_hf(path, mono16):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoProcessor, WhisperForConditionalGeneration
    from .model_store import ensure_whisper_processor
    ensure_whisper_processor()
    config=AutoConfig.from_pretrained(str(PROCESSOR_DIR),trust_remote_code=False,local_files_only=True)
    processor=AutoProcessor.from_pretrained(str(PROCESSOR_DIR),trust_remote_code=False,local_files_only=True)
    model=WhisperForConditionalGeneration(config)
    try:
        missing,unexpected=load_model(model,str(path),strict=False,device='cpu')
        if unexpected or any(k not in ('proj_out.weight','model.decoder.embed_tokens.weight') for k in missing):
            raise RuntimeError(f'Tensor mismatch: missing={missing[:12]}; unexpected={unexpected[:12]}')
        model=model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        chunks=[]
        sample_window=28*16000
        for start in range(0,len(mono16),sample_window):
            audio=mono16[start:start+sample_window]
            if len(audio)<1600: continue
            tensors=processor(audio,sampling_rate=16000,return_tensors='pt')
            features=tensors.input_features.to(model.device,dtype=model.dtype)
            with torch.inference_mode():
                ids=model.generate(features,max_new_tokens=192)
            line=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
            if line: chunks.append(line)
        return ' '.join(chunks)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def transcribe_selected(reference_audio,selected_file):
    from scipy.signal import resample_poly
    from .native_plan import waveform_mono
    import numpy as np
    import torch
    filename=Path(selected_file)
    mono24=waveform_mono(reference_audio,24000)
    mono16=resample_poly(mono24,2,3).astype(np.float32)
    if filename.suffix.lower()=='.pt':
        import whisper
        model=whisper.load_model(str(filename))
        try:
            text=model.transcribe(mono16,verbose=False,condition_on_previous_text=False)['text']
        finally:
            del model;gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    elif filename.suffix.lower()=='.safetensors':
        kind=sniff_weights(filename)
        if kind=='transformers':
            text=_transcribe_hf(filename,mono16)
        else:
            import whisper
            model=_load_openai_safetensors(filename)
            try:
                text=model.transcribe(mono16,verbose=False,condition_on_previous_text=False)['text']
            finally:
                del model;gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        raise RuntimeError(f'Unsupported Whisper model: {filename}')
    return str(text).strip()


def _transcribe_hf_windows(path, windows):
    """Diagnostic-only HF Whisper inference: one model load for all windows.

    The normal Reference Voice Whisper loader above is intentionally unchanged.
    """
    import torch
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoProcessor, WhisperForConditionalGeneration
    from .model_store import ensure_whisper_processor
    ensure_whisper_processor()
    config = AutoConfig.from_pretrained(str(PROCESSOR_DIR), trust_remote_code=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(PROCESSOR_DIR), trust_remote_code=False, local_files_only=True)
    model = WhisperForConditionalGeneration(config)
    try:
        missing, unexpected = load_model(model, str(path), strict=False, device='cpu')
        if unexpected or any(k not in ('proj_out.weight', 'model.decoder.embed_tokens.weight') for k in missing):
            raise RuntimeError(f'Whisper HF tensor mismatch: missing={missing[:12]}; unexpected={unexpected[:12]}')
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        output = []
        for mono16 in windows:
            tensors = processor(mono16, sampling_rate=16000, return_tensors='pt')
            features = tensors.input_features.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                ids = model.generate(features, max_new_tokens=192)
            output.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return output
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def transcribe_selected_windows(audio, selected_file, *, max_windows=6, window_seconds=20.0):
    """Segmented Whisper ASR on decoded target without storing/exporting audio.

    Windows cover the full audio when <=120 s; longer audio is sampled at
    stratified positions, and the output explicitly exposes sampled windows.
    A single model load is used for the entire diagnostic run.
    """
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from .native_plan import waveform_mono
    from .reference_diagnostics import temporal_windows

    filename = Path(selected_file)
    mono24 = waveform_mono(audio, 24000)
    mono16 = resample_poly(mono24, 2, 3).astype(np.float32)
    boundaries = temporal_windows(len(mono16) / 16000, window_seconds=window_seconds,
                                  max_windows=max_windows)
    segments = [mono16[int(s * 16000):int(e * 16000)] for s, e in boundaries]
    if filename.suffix.lower() == '.pt':
        import whisper
        model = whisper.load_model(str(filename))
    elif filename.suffix.lower() == '.safetensors':
        kind = sniff_weights(filename)
        if kind == 'transformers':
            transcripts = _transcribe_hf_windows(filename, segments)
            return [{'start_s': s, 'end_s': e, 'text': text}
                    for (s, e), text in zip(boundaries, transcripts)]
        model = _load_openai_safetensors(filename)
    else:
        raise RuntimeError(f'Unsupported Whisper model: {filename}')
    try:
        transcripts = [str(model.transcribe(chunk, verbose=False,
                                             condition_on_previous_text=False)['text']).strip()
                       for chunk in segments]
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return [{'start_s': s, 'end_s': e, 'text': text}
            for (s, e), text in zip(boundaries, transcripts)]
