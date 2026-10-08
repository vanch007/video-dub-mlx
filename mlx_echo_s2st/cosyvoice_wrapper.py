"""CosyVoice3 wrapper with MLX-powered Hidden2CV mapper for voice-preserved speech synthesis."""

import os
import sys
import time
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import mlx.core as mx

# Inject pyworld mock if missing
if "pyworld" not in sys.modules:
    import types
    sys.modules["pyworld"] = types.ModuleType("pyworld")

from mlx_echo_s2st.mapper_mlx import MLXHidden2CVMapper

INSTRUCT_CV3 = "You are a helpful assistant.<|endofprompt|>"


def char_pool(spans_src, h_src, spans_dst):
    """Character boundary overlap mean-pooling."""
    out = []
    h_src_np = np.asarray(h_src, dtype=np.float32)
    for (ds, de) in spans_dst:
        idx = [i for i, (ss, se) in enumerate(spans_src) if ss < de and se > ds]
        if not idx:
            i = min(range(len(spans_src)), key=lambda j: abs(spans_src[j][0] - ds))
            idx = [i]
        out.append(h_src_np[idx].mean(axis=0))
    return np.stack(out, axis=0)


def _patch_cv3_tf56(cv):
    """CosyVoice3 patch for transformers 5.x."""
    import transformers
    major = int(transformers.__version__.split('.')[0])
    if major < 5:
        return
    cv.model.llm = cv.model.llm.float()
    llm = cv.model.llm
    orig_fos = llm.llm.forward_one_step

    def fos_fixed(xs, masks, cache=None):
        past = cache.get_seq_length() if cache is not None else 0
        full = torch.ones(1, 1, past + xs.shape[1], dtype=torch.bool, device=xs.device)
        return orig_fos(xs, masks=full, cache=cache)

    llm.llm.forward_one_step = fos_fixed


class MLXCosyVoiceSynthesizer:
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")

        # Set up paths for CosyVoice repo
        repo_paths = [
            os.path.join(model_dir, "code"),
            os.path.join(model_dir, "code/cosyvoice_repo"),
            os.path.join(model_dir, "code/cosyvoice_repo/third_party/Matcha-TTS"),
            os.path.join(model_dir, "code/ja_ext"),
        ]
        for p in repo_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        from cosyvoice.cli.cosyvoice import CosyVoice3
        cv3_dir = os.path.join(model_dir, "cosyvoice3")
        print(f"[MLXCosyVoiceSynthesizer] Loading CosyVoice3 from {cv3_dir}...", flush=True)
        self.cv = CosyVoice3(cv3_dir, load_trt=False, load_vllm=False, fp16=False)
        _patch_cv3_tf56(self.cv)
        self.llm = self.cv.model.llm
        self.llm.eval()

        # Load MLX Hidden2CVMapper
        cfg_path = os.path.join(model_dir, "bridge/mapper_config.json")
        weights_path = os.path.join(model_dir, "bridge/mapper.safetensors")
        print(f"[MLXCosyVoiceSynthesizer] Loading MLXHidden2CVMapper...", flush=True)
        self.mapper = MLXHidden2CVMapper.from_pretrained(cfg_path, weights_path)

        with open(cfg_path) as f:
            mcfg = json.load(f)
        langs = list(mcfg.get("langs", ["en", "es", "ja", "zh"]))
        self.lang2idx = {}
        for i, t in enumerate(langs):
            self.lang2idx.setdefault(t, i)

        self.tok = self.cv.frontend.tokenizer.tokenizer
        self.emb_t = self.llm.llm.model.model.embed_tokens
        self.emb_s = self.llm.speech_embedding
        self.sample_rate = self.cv.sample_rate
        print(f"[MLXCosyVoiceSynthesizer] Ready! Sample rate: {self.sample_rate}Hz", flush=True)

    @torch.inference_mode()
    def synth(self, ext: Dict[str, Any], out_wav: Optional[str] = None, max_tokens: int = 1500, seed: int = 42) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Synthesize voice-preserved speech using MLX mapper + CosyVoice3.
        Returns: (waveform_numpy_1d, info)
        """
        lang = ext.get("lang", "en")
        tgt_cv = ext["tgt_cv"]
        enc = self.tok(tgt_cv, return_offsets_mapping=True, add_special_tokens=False)

        if lang == "ja" and ext.get("ja_words"):
            from ja_kata import kata_align
            h_al = kata_align(
                ext["ja_words"],
                torch.from_numpy(np.asarray(ext["h_st"], dtype=np.float32)),
                ext["spans_st"],
                enc.offset_mapping,
                len(tgt_cv),
                len(ext["tgt_raw"]),
            ).numpy()
        else:
            h_al = char_pool(ext["spans_st"], ext["h_st"], enc.offset_mapping)

        # CV3 text embedding
        ids = torch.tensor(enc.input_ids)
        e = self.emb_t(ids).float()

        # MLX Mapper forward
        h_mx = mx.array(h_al)[None, ...]
        e_mx = mx.array(e.numpy())[None, ...]
        lang_mx = mx.array([self.lang2idx[lang]])
        e_hat_mx = self.mapper(h_mx, e_mx, lang=lang_mx)[0]
        e_hat = torch.from_numpy(np.array(e_hat_mx)).to(self.device)

        # Zero-shot frontend condition
        src_line = ext["zh"]
        pwav = ext["wav"]
        mi = self.cv.frontend.frontend_zero_shot(
            tgt_cv, INSTRUCT_CV3 + src_line, pwav, self.sample_rate, ""
        )

        prompt_emb = self.emb_t(mi["prompt_text"].long().to(self.device))
        sos_emb = self.emb_s.weight[self.llm.sos].reshape(1, 1, -1)
        task_emb = self.emb_s.weight[self.llm.task_id].reshape(1, 1, -1)
        psp = mi["llm_prompt_speech_token"].long().to(self.device)
        psp_emb = self.emb_s(psp) if psp.shape[1] > 0 else torch.zeros(1, 0, self.llm.llm_input_size, device=self.device)

        lm_input = torch.cat([
            sos_emb, prompt_emb,
            e_hat.unsqueeze(0).to(prompt_emb.dtype),
            task_emb, psp_emb
        ], dim=1)

        m = e_hat.shape[0]
        min_len = max(int(m * 2), 10)
        max_len = min(int(m * 20), max_tokens)
        t0 = time.time()
        torch.manual_seed(seed)
        tokens = list(self.llm.inference_wrapper(lm_input, 25, min_len, max_len, ""))
        gen_s = time.time() - t0
        hit_eos = len(tokens) < max_len

        toks = list(tokens)

        def fake_inference(**_kw):
            return iter(toks)

        orig = self.llm.inference
        self.llm.inference = fake_inference
        try:
            torch.manual_seed(seed)
            wavs = [r["tts_speech"] for r in self.cv.model.tts(**mi, stream=False, speed=1.0)]
        finally:
            self.llm.inference = orig

        w = torch.cat(wavs, dim=1)
        wav_np = w.squeeze(0).cpu().float().numpy()

        if out_wav:
            import soundfile as sf
            sf.write(out_wav, wav_np, self.sample_rate)

        info = {
            "n_tokens": len(tokens),
            "hit_eos": hit_eos,
            "gen_s": round(gen_s, 2),
            "lang": lang,
            "src_lang": ext.get("src_lang"),
            "zh": ext["zh"],
            "tgt_raw": ext["tgt_raw"],
            "tgt_cv": tgt_cv,
            "sample_rate": self.sample_rate,
        }
        return wav_np, info
