"""Native tokenizer-head inference for YuE2 Reference Voice.

Reimplementation of the inference-only portion of the Mothersuperior
joint-v9 tokenizer. No dynamically downloaded trainer scripts, no execution of
code in a model directory and no extra YuE2 checkpoint.

Architecture: MERT-v2-FullSong hidden_states[20] -> per-song standardization
-> 8-layer 512-wide encoder -> 32768-way greedy codes @25 Hz.
Reference: Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4 (CC BY-NC 4.0)

The MERT2 architecture is a distinct model with its own official Python
implementation; that implementation is installed under this node's vendor tree
by YuE2ReferenceVoiceModels, not under models/audio_encoders/.
"""
from __future__ import annotations

from math import gcd
import numpy as np

CODEBOOK_SIZE = 32768
FEATURE_DIM = 1024
MODEL_WIDTH = 512
WINDOW = 512
LAYERS = 8
HEADS = 8
FRAME_HZ = 25


def normalize_features(features):
    """Per-song population-statistics normalization, shape [T,1024]."""
    features = np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or features.shape[1]!=FEATURE_DIM or len(features)<1:
        raise ValueError('MERT features must be nonempty [frames,1024]')
    if not np.isfinite(features).all():
        raise ValueError('MERT features contain nonfinite values')
    return (features-features.mean(axis=0))/(features.std(axis=0)+1e-5)


def build_tokenizer_head():
    """Return the exact checkpoint-compatible v9 head architecture.

    Lazy construction avoids importing PyTorch during ComfyUI's node discovery.
    """
    import torch
    from torch import nn

    class TokenizerHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp=nn.Linear(FEATURE_DIM,MODEL_WIDTH)
            self.pos=nn.Parameter(torch.zeros(1,WINDOW,MODEL_WIDTH))
            layer=nn.TransformerEncoderLayer(MODEL_WIDTH,HEADS,4*MODEL_WIDTH,
                    dropout=0.1,batch_first=True,norm_first=True,activation='gelu')
            self.enc=nn.TransformerEncoder(layer,LAYERS)
            self.norm=nn.LayerNorm(MODEL_WIDTH)
            self.head=nn.Linear(MODEL_WIDTH,CODEBOOK_SIZE)
        def forward(self,frames):
            y=self.inp(frames)
            return self.head(self.norm(self.enc(y+self.pos[:,:y.shape[1]])))

    return TokenizerHead()


def extract_mert_features(mono24, model_dir):
    import torch
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
    from .vendor.model_code.mert2.configuration_mert2 import MERT2Config
    from .vendor.model_code.mert2.modeling_mert2 import MERT2Model

    mono24=np.asarray(mono24,dtype=np.float32)
    if mono24.ndim != 1 or len(mono24)<24000 or not np.isfinite(mono24).all():
        raise ValueError('Reference audio must be finite mono 24 kHz and at least one second')
    if not torch.cuda.is_available():
        raise RuntimeError('YuE2 reference tokenizer requires CUDA')
    from pathlib import Path
    model_path=Path(model_dir)
    if model_path.is_file() and model_path.suffix.lower()=='.safetensors':
        # Official MERT-v2-FullSong architecture settings, identical to the
        # published config.json (variant fs). No HF model directory is needed.
        from .mert_checkpoint import load_mert_checkpoint
        metadata_dir=Path(__file__).resolve().parent/'vendor'/'model_code'/'mert2'
        if (metadata_dir/'config.json').is_file() and (metadata_dir/'preprocessor_config.json').is_file():
            # Read the exact model config and waveform preprocessing from the
            # official MERT repository; neither lives under ComfyUI/models.
            config=MERT2Config.from_pretrained(str(metadata_dir),local_files_only=True)
            processor=AutoFeatureExtractor.from_pretrained(str(metadata_dir),
                         trust_remote_code=False,local_files_only=True)
        else:
            # Backwards compatible while the manager installs tiny official JSON.
            config=MERT2Config(variant='fs',context_seconds=360.0)
            processor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=24000,
                         padding_value=0.0,do_normalize=True,return_attention_mask=True)
        model=MERT2Model(config)
        try:
            # The safetensors.load_model helper fails *before* loading when
            # torchaudio's window buffer is a view into larger storage.
            # Our loader validates every learned tensor and refuses actual
            # architecture mismatches, without that shared-storage scan.
            load_mert_checkpoint(model,model_path)
        except Exception:
            del model
            raise
    elif model_path.is_dir():
        processor=AutoFeatureExtractor.from_pretrained(str(model_path),
                      trust_remote_code=False,local_files_only=True)
        config=MERT2Config.from_pretrained(str(model_path),local_files_only=True)
        model=MERT2Model.from_pretrained(str(model_path),config=config,
                      trust_remote_code=False,local_files_only=True)
    else:
        raise RuntimeError(f'Selected MERT model does not exist: {model_path}')
    model=model.to('cuda').eval()
    chunk_samples=24000*30
    chunks=[mono24[x:x+chunk_samples] for x in range(0,len(mono24),chunk_samples)]
    # Official MERT extraction discards tails shorter than one second.
    if len(chunks)>1 and len(chunks[-1])<24000:
        raise ValueError('MERT would drop a reference tail shorter than 1 second; trim the reference differently')
    full=[c for c in chunks if len(c)==chunk_samples]
    tail=[c for c in chunks if 24000<=len(c)<chunk_samples]
    feats=[]
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            batches=([full] if full else [])+[[c] for c in tail]
            for batch in batches:
                model_input={name:value.to('cuda') for name,value in processor(batch,
                          sampling_rate=24000,return_tensors='pt').items()}
                out=model(**model_input,output_hidden_states=True)
                feats.append(out.hidden_states[20].reshape(-1,FEATURE_DIM))
            raw=torch.cat(feats,dim=0).float()
            count=int(round(len(mono24)/24000*FRAME_HZ))
            interpolated=F.interpolate(raw.T[None],size=count,mode='linear',
                                        align_corners=False)[0].T.half().cpu().numpy()
            return interpolated
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def infer_codes(features, weights_path):
    """Load the joint-v9 head from safetensors and infer greedily in windows.

    Preserves the training window policy (512, hop 256, interior crop 128).
    """
    import torch
    from safetensors.torch import load_file
    x=normalize_features(features)
    count=len(x)
    predictions=np.zeros(count,dtype=np.int32)
    head=build_tokenizer_head().to('cuda').eval()
    try:
        tensors=load_file(str(weights_path),device='cpu')
        head.load_state_dict(tensors,strict=True)
        del tensors
        with torch.inference_mode():
            begins=list(range(0,max(1,count-WINDOW+1),WINDOW//2))
            if begins[-1]+WINDOW<count:
                begins.append(max(0,count-WINDOW))
            for beginning in begins:
                frame=x[beginning:beginning+WINDOW]
                number=len(frame)
                if number<WINDOW:
                    frame=np.pad(frame,((0,WINDOW-number),(0,0)))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    values=head(torch.as_tensor(frame[None],device='cuda'))[0,:number]
                    greedy=values.float().argmax(-1).cpu().numpy().astype(np.int32)
                left=beginning+(0 if beginning==0 else WINDOW//4)
                right=beginning+number-(0 if beginning+number>=count else WINDOW//4)
                predictions[left:right]=greedy[left-beginning:right-beginning]
        if predictions.size!=count or predictions.min()<0 or predictions.max()>=CODEBOOK_SIZE:
            raise RuntimeError('Invalid YuE2 semantic code range/length')
        return tuple(int(x) for x in predictions)
    finally:
        del head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_audio(reference_audio, model_dir, head_file):
    """ComfyUI AUDIO -> 24 kHz mono -> MERT -> v9 codec prefix.

    The caller's reference audio may be 24 kHz or another positive integer rate.
    """
    from scipy.signal import resample_poly
    import torch
    rate=int(reference_audio['sample_rate'])
    wave=reference_audio['waveform']
    if rate<1000 or wave.ndim!=3 or wave.shape[0]!=1:
        raise ValueError('Expected AUDIO batch=1, channels x samples with valid sample rate')
    mono=wave[0].float().mean(0).detach().cpu().numpy()
    if rate!=24000:
        common=gcd(rate,24000)
        mono=resample_poly(mono,24000//common,rate//common).astype(np.float32)
    features=extract_mert_features(mono,model_dir)
    codes=infer_codes(features,head_file)
    expected=int(round(len(mono)/24000*FRAME_HZ))
    if len(codes)!=expected:
        raise RuntimeError(f'Codec framing mismatch: {len(codes)} vs {expected}')
    return codes
