"""Pure Apple Silicon MLX CosyVoice3 Synthesizer for Echo-S2ST.

Replaces PyTorch CosyVoice3 and ONNX models with 100% native MLX implementations:
- MLX Qwen2 LLM backbone
- MLX DiT Flow Matching
- MLX Causal HiFi-GAN (HiFT) vocoder
- MLX CAMPlus speaker encoder (no campplus.onnx)
- MLX S3Tokenizer V3 (no speech_tokenizer_v3.onnx)
- MLX Hidden2CV Mapper
"""

import json
import os
import sys
import time
from typing import Dict, Any, Tuple, Optional

import mlx.core as mx
import numpy as np

from mlx_audio.tts.utils import load_model
from mlx_audio.codec.models.s3gen.mel import mel_spectrogram as cosyvoice_mel_spectrogram
from mlx_audio.codec.models.s3tokenizer import log_mel_spectrogram_compat as log_mel_spectrogram
import soundfile as sf
import librosa

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from video_dub_mlx.mapper_mlx import MLXHidden2CVMapper

INSTRUCT_CV3 = "You are a helpful assistant.<|endofprompt|>"


def char_pool(spans_src, h_src, spans_dst):
    out = []
    h_src_np = np.asarray(h_src, dtype=np.float32)
    for (ds, de) in spans_dst:
        idx = [i for i, (ss, se) in enumerate(spans_src) if ss < de and se > ds]
        if not idx:
            i = min(range(len(spans_src)), key=lambda j: abs(spans_src[j][0] - ds))
            idx = [i]
        out.append(h_src_np[idx].mean(axis=0))
    return np.stack(out, axis=0)


class PureMLXCosyVoiceSynthesizer:
    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)

        # 1. Load MLX Hidden2CV Mapper
        cfg_path = os.path.join(self.model_dir, 'bridge/mapper_config.json')
        weights_path = os.path.join(self.model_dir, 'bridge/mapper.safetensors')
        print('[PureMLXCosyVoice] Loading MLXHidden2CVMapper...', flush=True)
        self.mapper = MLXHidden2CVMapper.from_pretrained(cfg_path, weights_path)

        with open(cfg_path) as f:
            mcfg = json.load(f)
        langs = list(mcfg.get('langs', ['en', 'es', 'ja', 'zh']))
        self.lang2idx = {t: i for i, t in enumerate(langs)}

        # 2. Load Pure MLX CosyVoice3
        print('[PureMLXCosyVoice] Loading Pure MLX CosyVoice3 (LLM, Flow, HiFT, CAMPlus, S3Tokenizer)...', flush=True)
        self.model_wrapper = load_model('mlx-community/Fun-CosyVoice3-0.5B-2512-fp16')
        self.model_wrapper._ensure_model_loaded()
        self.model_wrapper._ensure_tokenizers_loaded()

        self.cv3 = self.model_wrapper._model
        self.tokenizer = self.model_wrapper._tokenizer
        self.speaker_encoder = self.model_wrapper._speaker_encoder
        self.s3_tokenizer = self.model_wrapper._s3_tokenizer

        self.sample_rate = 24000
        print('[PureMLXCosyVoice] Ready! All speech components are 100% Pure MLX.', flush=True)

    def synth(self, ext: Dict[str, Any], out_wav: Optional[str] = None, max_tokens: int = 1500, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Synthesize voice-preserved speech using 100% Pure MLX components."""
        lang = ext.get('lang', 'en')
        tgt_cv = ext['tgt_cv']

        # 1. Text Tokenization & Character Alignment
        enc = self.tokenizer(tgt_cv, return_offsets_mapping=True, add_special_tokens=False)
        offset_mapping = enc['offset_mapping'] if 'offset_mapping' in enc else enc.offset_mapping
        input_ids = enc['input_ids'] if 'input_ids' in enc else enc.input_ids

        h_al = char_pool(ext['spans_st'], ext['h_st'], offset_mapping)

        # 2. Target Text Embedding from Pure MLX Qwen2
        ids_mx = mx.array([input_ids], dtype=mx.int32)
        e_mx = self.cv3.llm.llm.embed_tokens(ids_mx)  # (1, T_tgt, 896)

        # 3. Pure MLX Mapper Forward
        h_mx = mx.array(h_al)[None, ...]  # (1, T_tgt, 4096)
        lang_mx = mx.array([self.lang2idx[lang]])
        e_hat_mx = self.mapper(h_mx, e_mx, lang=lang_mx)[0]  # (T_tgt, 896)

        # 4. Reference Audio Feature Extraction (16k & 24k)
        pwav_path = ext['wav']
        if isinstance(pwav_path, str):
            wav_16k, _ = librosa.load(pwav_path, sr=16000)
            wav_24k, _ = librosa.load(pwav_path, sr=24000)
        else:
            wav_raw = np.asarray(pwav_path, dtype=np.float32)
            wav_16k = librosa.resample(wav_raw, orig_sr=self.sample_rate, target_sr=16000)
            wav_24k = wav_raw

        # Limit to 30s
        wav_16k = wav_16k[:16000 * 30]
        wav_24k = wav_24k[:24000 * 30]

        ref_16k_mx = mx.array(wav_16k, dtype=mx.float32)
        ref_24k_mx = mx.array(wav_24k, dtype=mx.float32)

        # Speaker embedding via Pure MLX CAMPlus
        spk_emb = self.speaker_encoder(ref_16k_mx, sample_rate=16000)

        # Speech tokens via Pure MLX S3Tokenizer V3
        mel_128 = log_mel_spectrogram(ref_16k_mx, n_mels=128)
        mel_128 = mx.expand_dims(mel_128, 0)
        mel_len_128 = mx.array([mel_128.shape[2]])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.s3_tokenizer(mel_128, mel_len_128)

        # Mel spectrogram (80-bin) for Flow DiT prompt
        mel_80 = cosyvoice_mel_spectrogram(ref_24k_mx, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)
        mel_80 = mx.swapaxes(mel_80, 1, 2)  # (1, T, 80)

        token_len = int(prompt_speech_tokens_lens[0].item())
        max_mel_len = int(mel_80.shape[1])
        if max_mel_len < token_len * 2:
            token_len = max_mel_len // 2

        mel_len = token_len * 2
        prompt_mel = mel_80[:, :mel_len, :]
        prompt_mel_len = mx.array([mel_len], dtype=mx.int32)
        prompt_speech_tokens = prompt_speech_tokens[:, :token_len]
        prompt_speech_tokens_len = mx.array([token_len], dtype=mx.int32)

        # Prompt text embedding
        src_line = INSTRUCT_CV3 + ext['zh']
        prompt_text_tokens = self.tokenizer.encode(src_line, add_special_tokens=False)
        prompt_text_mx = mx.array([prompt_text_tokens], dtype=mx.int32)
        prompt_text_emb = self.cv3.llm.llm.embed_tokens(prompt_text_mx)

        # 5. Construct Initial LM Input with e_hat
        sos_emb = self.cv3.llm.speech_embedding.weight[self.cv3.llm.sos].reshape(1, 1, -1)
        task_id_emb = self.cv3.llm.speech_embedding.weight[self.cv3.llm.task_id].reshape(1, 1, -1)
        if prompt_speech_tokens_len.item() > 0:
            prompt_speech_emb = self.cv3.llm.speech_embedding(prompt_speech_tokens)
        else:
            prompt_speech_emb = mx.zeros((1, 0, self.cv3.llm.llm_input_size), dtype=sos_emb.dtype)

        # [sos, prompt_text, e_hat, task_id, prompt_speech]
        lm_input = mx.concatenate([
            sos_emb,
            prompt_text_emb,
            e_hat_mx.reshape(1, -1, 896).astype(sos_emb.dtype),
            task_id_emb,
            prompt_speech_emb
        ], axis=1)

        m = e_hat_mx.shape[0]
        min_len = max(int(m * 2), 10)
        max_len = min(int(m * 20), max_tokens)

        # 6. Autoregressive Speech Tokens Generation in Pure MLX
        t0 = time.time()
        mx.random.seed(seed)
        tokens = list(self.cv3.llm._inference_loop(lm_input, sampling=25, min_len=min_len, max_len=max_len))
        gen_s = time.time() - t0
        hit_eos = len(tokens) < max_len

        # 7. Flow Matching DiT & HiFT Vocoder in Pure MLX
        token_array = mx.array([tokens], dtype=mx.int32)
        token_lens = mx.array([len(tokens)], dtype=mx.int32)

        mel, _ = self.cv3.tokens_to_mel(
            tokens=token_array,
            token_len=token_lens,
            prompt_token=prompt_speech_tokens,
            prompt_token_len=prompt_speech_tokens_len,
            prompt_feat=prompt_mel,
            prompt_feat_len=prompt_mel_len,
            embedding=spk_emb,
            finalize=True,
            n_timesteps=10,
        )

        audio = self.cv3.mel_to_audio(mel)
        wav_np = np.array(audio.squeeze(), dtype=np.float32)

        if out_wav:
            sf.write(out_wav, wav_np, self.sample_rate)

        info = {
            'n_tokens': len(tokens),
            'hit_eos': hit_eos,
            'gen_s': round(gen_s, 2),
            'lang': lang,
            'src_lang': ext.get('src_lang'),
            'zh': ext['zh'],
            'tgt_raw': ext['tgt_raw'],
            'tgt_cv': tgt_cv,
            'sample_rate': self.sample_rate,
        }
        return wav_np, info
