"""Forced YuE2 semantic-history adapter for ComfyUI's native YuE2 text encoder.

Based on ComfyUI comfy/text_encoders/yue2.py (YuE2TEModel._generate) and
Ornn8/yue2-cover/yue2_cover/prefix_sampling.py, Apache-2.0.

This is a context-local override, not a new YuE2 model and not yue2-infer.
It injects reference codec tokens into the actual autoregressive KV history,
then lets the stock encoder build NAR conditioning for the complete song.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import logging
import threading

_active = ContextVar('yue2_native_reference_prefix', default=None)
_install_lock = threading.RLock()
_original = None


@dataclass
class PrefixCall:
    codes: tuple[int, ...]
    used: bool = False
    progress_node_id: str = ''
    last_progress: int = 49
    sampled_frames: int = 0
    forced_frames: int = 0
    required_target_frames: int = 0


def checked_codes(codes):
    if not isinstance(codes,(tuple,list)) or not codes:
        raise ValueError('Reference prefix must contain at least one frame')
    if len(codes)>2500:  # 100 seconds maximum for one vocal reference
        raise ValueError('Reference prefix exceeds 100 seconds')
    if any(type(c) is not int or c<0 or c>=32768 for c in codes):
        raise ValueError('Reference prefix contains invalid raw codec IDs (0..32767)')
    return tuple(codes)


def _semantic_with_reference(self, prefix, seed, max_tokens, phase, dtype,
                             negative=None, cfg_scale=1.0, legacy_off=False, **sampling):
    """Mirror current stock YuE2TEModel._generate, with KV prefill extended by codec frames."""
    import torch
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.utils
    from comfy.text_encoders.llama import FixedKV, rope_matrix
    from comfy.text_encoders.yue2 import (ABC_END, MUSIC_END, EOD, CODEC_OFFSET, distribution)

    call=_active.get()
    if call is None or phase != 'semantic':
        return _original(self,prefix,seed,max_tokens,phase,dtype,
                         negative=negative,cfg_scale=cfg_scale,legacy_off=legacy_off,**sampling)
    if call.used:
        raise RuntimeError('YuE2 prefix adapter invoked twice in one generation')
    call.used=True
    if max_tokens <= len(call.codes):
        raise ValueError('max_duration is shorter than voice reference; increase max_duration')
    forced=[CODEC_OFFSET+x for x in call.codes]
    max_new=max_tokens-len(forced)
    call.forced_frames = len(forced)
    # Stock YuE2 counts the semantic tokens before considering MUSIC_END.
    # The legacy adapter restarted this guard at zero immediately after a large
    # forced voice history and could terminate after only a few song seconds.
    minimum_target = min(max_new - 1, max(0, call.required_target_frames))
    sampling['min_tokens'] = max(int(sampling.get('min_tokens', 0)), len(forced) + minimum_target)
    branches=[list(prefix)+forced] if cfg_scale==1.0 else [list(prefix)+forced, list(negative)+forced]
    prefix_length=max(map(len,branches))
    if prefix_length+max_new > self.config.max_position_embeddings:
        raise ValueError('YuE2 combined score/prefix exceeds context length; shorten song or reference')
    device=self.execution_device
    rng_device=device if torch.device(device).type!='mps' else 'cpu'
    generator=torch.Generator(device=rng_device).manual_seed(int(seed))
    logits,cache,mask=self._prefill(branches,prefix_length+max_new,dtype)
    fixed_kv=isinstance(cache[0], FixedKV)
    positions=torch.tensor([[len(p)] for p in branches],device=device,dtype=torch.long)
    decode_buffers=(torch.empty((len(branches),1,self.config.hidden_size),device=device,dtype=dtype),
                    rope_matrix(self.model.compute_freqs_cis(positions,device)))
    end=MUSIC_END
    # Do not create a second ComfyUI progress stream for a GraphBuilder child.
    # Parent UI has one correlated websocket progress channel.
    penalty_window=int(sampling.pop('penalty_window',50))
    recent=torch.full((1,penalty_window),EOD,device=device,dtype=torch.long)
    n=min(len(forced),penalty_window)
    if n:
        forced_tail=torch.tensor(forced[-n:],device=device,dtype=torch.long)
        for i in range(n):
            recent[0, (len(forced) - n + i) % penalty_window] = forced_tail[i]
    token_buffer=torch.empty((max_new,),device=device,dtype=torch.long)
    step_index=torch.zeros((),device=device,dtype=torch.long)
    history=[]
    def head(step):
        guided=logits if cfg_scale==1.0 else logits[1:]+cfg_scale*(logits[:1]-logits[1:])
        scores=distribution(guided,recent,step + len(forced),'semantic',legacy_off=legacy_off,**sampling)
        if sampling['temperature']==0:
            next_id=scores.argmax(-1,keepdim=True)
        else:
            prob=scores.softmax(-1).to(rng_device)
            next_id=torch.multinomial(prob,1,generator=generator).to(device)
        token_buffer.index_copy_(0,step_index.view(1),next_id.view(1))
        recent.scatter_(1,((step_index+len(forced))%penalty_window).view(1,1),next_id)
        step_index.add_(1)
        decode_buffers[0].copy_(self.model.embed_tokens(next_id.expand(len(branches),1),out_dtype=dtype))
    try:
        for step in comfy.utils.model_trange(max_new,desc='YuE2 reference-voice generation',unit='token'):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if fixed_kv:
                comfy.model_prefetch.malloc_graph_begin(device)
            head_queue=comfy.model_prefetch.make_prefetch_queue([self.model.embed_tokens],device,
                  {'prefetch_dynamic_vbars':self.model.prefetch_dynamic_vbars})
            comfy.model_prefetch.prefetch_queue_pop(head_queue,device,self.model.embed_tokens,dtype,
                      core=lambda:head(step),enable_graph=(step + len(forced))>=sampling['min_tokens'],
                      generator=generator,malloc_scope='head')
            comfy.model_prefetch.prefetch_queue_pop(head_queue,device,None,malloc_scope='head')
            # Native model_trange remains the local token loop; parent status is emitted below.
            if call.progress_node_id:
                # Emit at most 18 parent updates for thousands of AR tokens.
                milestone=49+int(27*(step+1)/max_new)
                if milestone > call.last_progress:
                    from .reference_progress import report_progress
                    report_progress(call.progress_node_id,'ar_tokens',milestone)
                    call.last_progress=milestone
            output=self.model(None,embeds=decode_buffers[0],past_key_values=cache,dtype=dtype,
                position_ids=positions,
                attention_mask=mask[:,:prefix_length+step+1] if mask is not None and not fixed_kv else None,
                decode_buffers=decode_buffers)
            logits.copy_(self.model.lm_head(output[0][:,-1]))
            cache=output[2]
            del output
            if fixed_kv:
                comfy.model_prefetch.malloc_graph_end()
            positions.add_(1)
            if (step+1)%16==0 or step+1==max_new:
                for token in token_buffer[len(history):step+1].tolist():
                    if token==end:
                        call.sampled_frames = len(history)
                        return forced+history,False
                    history.append(token)
    finally:
        comfy.model_prefetch.cleanup_prefetch_queues()
    call.sampled_frames = len(history)
    logging.warning('YuE2 voice prefix exhausted its generation budget')
    return forced+history,True


def encode_with_reference(clip, style, lyrics, abc, codes, seed, mode, max_duration,
                          temperature=1.0, top_p=.95, top_k=100, repetition_penalty=1.2,
                          progress_node_id='', min_target_seconds=0.0):
    """One synchronous scheduled-CLIP call. Fail closed if native internals drift."""
    import comfy.text_encoders.yue2 as native
    from comfy.text_encoders.yue2 import FRAMES_PER_SECOND
    global _original
    codes=checked_codes(codes)
    if mode not in ('melody','full'):
        raise ValueError('YuE2 native mode must be melody/full')
    if not abc.strip() or not lyrics.strip():
        raise ValueError('Composite ABC and Lyrics must both be nonempty')
    with _install_lock:
        if _original is None:
            _original=native.YuE2TEModel._generate
            def bridge(self,prefix,seed,max_tokens,phase,dtype,negative=None,cfg_scale=1.0,legacy_off=False,**sampling):
                return _semantic_with_reference(self,prefix,seed,max_tokens,phase,dtype,
                      negative=negative,cfg_scale=cfg_scale,legacy_off=legacy_off,**sampling)
            native.YuE2TEModel._generate=bridge
        call=PrefixCall(codes,progress_node_id=str(progress_node_id or ''),required_target_frames=max(0,int(round(float(min_target_seconds)*FRAMES_PER_SECOND))))
        token=_active.set(call)
        try:
            tokens=clip.tokenize(style,lyrics=lyrics,cot=mode,seed=int(seed),abc=abc,
                max_tokens=max(1,round(float(max_duration)*FRAMES_PER_SECOND)),
                temperature=float(temperature),top_p=float(top_p),top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),cfg_scale=1.0)
            result=clip.encode_from_tokens_scheduled(tokens)
            if not call.used:
                raise RuntimeError('Native YuE2 text encoder did not invoke the reference-prefix adapter. '
                                   'Verify that this ComfyUI version supports YuE2TEModel._generate.')
            try:
                frames=result[0][1]['yue2_frames']
            except (TypeError,IndexError,KeyError) as exc:
                raise RuntimeError('Native YuE2 conditioning layout changed') from exc
            from .reference_quality import validate_semantic_duration
            validate_semantic_duration(frames, len(codes), min_target_seconds)
            print('[YuE2 Reference Voice][SEMANTIC DURATION] ' + str({
                'prefix_s': round(len(codes) / FRAMES_PER_SECOND, 2),
                'target_s': round((frames - len(codes)) / FRAMES_PER_SECOND, 2),
                'minimum_target_s': round(float(min_target_seconds), 2),
                'truncated': bool(result[0][1].get('yue2_truncated'))}))
            return result,frames/FRAMES_PER_SECOND
        finally:
            _active.reset(token)
