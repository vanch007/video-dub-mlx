"""MLX-accelerated Speech-Translation Language Model (STLM) for Echo-S2ST.

Combines AuT (Audio Tower) + AudioConnector with MLX Qwen3.5-9B for fast Apple Silicon inference.
"""

import json
import os
import sys
import time
from typing import List, Tuple, Optional, Dict, Any

import mlx.core as mx
import mlx_lm.models.qwen3_5 as qwen3_5
from transformers import AutoTokenizer, WhisperFeatureExtractor
import torch
import torch.nn as nn
import numpy as np

AUDIO_PAD = 248076  # <|audio_pad|>
AUDIO_DIM = 2048    # AuT output_dim
EMBED_RMS_9B = 0.0137
AUT_RMS = 0.0143

INSTR = {
    "en": "把这段语音翻译成英文，先输出原文，换行输出英文译文。",
    "es": "把这段语音翻译成西班牙语，先输出原文，换行输出西班牙语译文。",
    "ja": "把这段语音翻译成日语，先输出原文，换行输出日语译文。",
    "zh": "把这段语音翻译成中文，先输出原文，换行输出中文译文。",
}


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def parse_hyp(hyp: str) -> Tuple[str, str]:
    lines = [l.strip() for l in hyp.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        if len(lines) == 1:
            return lines[0], lines[0]
        return "", ""
    return lines[0], lines[-1]


class AudioConnector9B(nn.Module):
    def __init__(self, in_dim=AUDIO_DIM, out_dim=4096, gain=EMBED_RMS_9B / AUT_RMS / (AUDIO_DIM / 4096) ** 0.5):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h):
        return self.proj(h)


class AudioEncoder:
    def __init__(self, omni_config_path: str, ckpt_path: str, device="mps"):
        self.device = torch.device(device if torch.backends.mps.is_available() else "cpu")
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder, Qwen3OmniMoeAudioEncoderConfig)
        from safetensors.torch import load_file

        with open(omni_config_path) as f:
            cfg = json.load(f)
        audio_cfg = Qwen3OmniMoeAudioEncoderConfig(**{
            k: v for k, v in cfg["thinker_config"]["audio_config"].items()
            if not k.startswith("_")
        })
        self.audio_tower = Qwen3OmniMoeAudioEncoder(audio_cfg).eval()
        self.connector = AudioConnector9B(out_dim=4096).eval()

        # Load trained weights
        st_file = os.path.join(ckpt_path, "model.safetensors") if os.path.isdir(ckpt_path) else ckpt_path
        sd = load_file(st_file)
        au = {k[len("audio_model."):]: v for k, v in sd.items() if k.startswith("audio_model.")}
        co = {k[len("connector."):]: v for k, v in sd.items() if k.startswith("connector.")}
        self.audio_tower.load_state_dict(au, strict=True)
        self.connector.load_state_dict(co, strict=True)

        self.audio_tower.to(self.device).float()
        self.connector.to(self.device).float()

        self.fe = WhisperFeatureExtractor(
            feature_size=128, hop_length=160, n_fft=400,
            sampling_rate=16000, padding_value=0.0,
            return_attention_mask=True
        )
        self.fe.n_samples = 4800000
        self.fe.nb_max_frames = 30000

    @torch.inference_mode()
    def encode(self, wav_path_or_data) -> mx.array:
        """Encode audio file or 1D float numpy array (16kHz) -> mx.array (T, 4096)."""
        import librosa
        if isinstance(wav_path_or_data, str):
            wav, _ = librosa.load(wav_path_or_data, sr=16000)
        else:
            wav = np.asarray(wav_path_or_data, dtype=np.float32)

        f = self.fe(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        T = int(f.attention_mask.sum(-1)[0])
        feats = f.input_features[0][:, :T]

        out = self.audio_tower(
            input_features=feats.to(self.device),
            feature_lens=torch.tensor([T], device=self.device)
        )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        emb = self.connector(h.to(self.device))  # (T_down, 4096)
        return mx.array(emb.cpu().numpy().astype(np.float32))


class MLXEchoSTLM:
    def __init__(self, llm_path: str, omni_config_path: str, ckpt_path: str):
        self.llm_path = llm_path
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

        print("[MLXEchoSTLM] Loading Qwen3.5-9B into MLX...", flush=True)
        with open(os.path.join(llm_path, "config.json")) as f:
            cfg = json.load(f)
        args = qwen3_5.ModelArgs.from_dict(cfg)
        self.model = qwen3_5.Model(args)

        # Load pure MLX weights
        st_file = os.path.join(llm_path, "model.safetensors")
        print(f"[MLXEchoSTLM] Loading pure MLX weights from {st_file}...", flush=True)
        weights = mx.load(st_file)
        if any(k.startswith("language_model.") for k in weights.keys()):
            sanitized = weights
        else:
            sanitized = self.model.sanitize(weights)
        self.model.load_weights(list(sanitized.items()), strict=False)
        print("[MLXEchoSTLM] Qwen3.5-9B loaded successfully into MLX!", flush=True)

        print("[MLXEchoSTLM] Loading Audio Tower...", flush=True)
        self.audio_encoder = AudioEncoder(omni_config_path, ckpt_path)
        print("[MLXEchoSTLM] Audio Tower loaded successfully!", flush=True)

    def _prepare_prompt_and_embeddings(self, audio_emb: mx.array, lang: str) -> Tuple[str, mx.array, mx.array]:
        instr = INSTR[lang]
        user_content = f"<audio>\n{instr}"
        n_audio = audio_emb.shape[0]
        block = "<|audio_start|>" + "<|audio_pad|>" * n_audio + "<|audio_end|>"
        user_content = user_content.replace("<audio>", block, 1)
        prompt = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        tokens = np.array(self.tokenizer(prompt, return_tensors="np").input_ids[0])
        pad_positions = np.where(tokens == AUDIO_PAD)[0]
        assert len(pad_positions) == n_audio, f"Pad tokens count {len(pad_positions)} != {n_audio}"
        start_pad = int(pad_positions[0])
        end_pad = int(pad_positions[-1]) + 1

        prefix_ids = tokens[:start_pad]
        suffix_ids = tokens[end_pad:]

        embed_fn = self.model.language_model.model.embed_tokens
        dt = embed_fn.weight.dtype

        prefix_emb = embed_fn(mx.array(prefix_ids)[None, :])
        suffix_emb = embed_fn(mx.array(suffix_ids)[None, :])
        audio_emb_typed = audio_emb.astype(dt)[None, :]

        inputs_embeds = mx.concatenate([prefix_emb, audio_emb_typed, suffix_emb], axis=1)
        input_ids = mx.array(tokens)[None, :]
        return prompt, input_ids, inputs_embeds

    def generate(self, audio_path_or_data, lang="en", max_new_tokens=1024) -> Tuple[str, str, str, mx.array, str]:
        assert lang in INSTR, f"Unsupported target lang: {lang}"
        audio_emb = self.audio_encoder.encode(audio_path_or_data)
        prompt, input_ids, inputs_embeds = self._prepare_prompt_and_embeddings(audio_emb, lang)

        cache = self.model.make_cache()

        # Prefill prompt + audio embeddings in MLX
        logits = self.model(None, cache=cache, input_embeddings=inputs_embeds)
        next_token = mx.argmax(logits[:, -1:], axis=-1)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_id = self.tokenizer.eos_token_id

        out_tokens = []
        for _ in range(max_new_tokens):
            t_id = next_token.item()
            if t_id in (im_end_id, eos_id):
                break
            out_tokens.append(t_id)
            logits = self.model(next_token, cache=cache)
            next_token = mx.argmax(logits[:, -1:], axis=-1)

        hyp = self.tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        src_text, tgt_raw = parse_hyp(hyp)
        return hyp, src_text, tgt_raw, audio_emb, prompt

    def extract_hidden_states(self, prompt: str, audio_emb: mx.array, src_text: str, tgt_st: str) -> Tuple[mx.array, List[Tuple[int, int]]]:
        target = f"{src_text}\n{tgt_st}"
        tok = self.tokenizer
        t_enc = tok(target, return_offsets_mapping=True, add_special_tokens=False)

        prompt_tokens = np.array(tok(prompt, return_tensors="np").input_ids[0])
        pad_positions = np.where(prompt_tokens == AUDIO_PAD)[0]
        start_pad = int(pad_positions[0])
        end_pad = int(pad_positions[-1]) + 1

        prefix_ids = prompt_tokens[:start_pad]
        suffix_ids = prompt_tokens[end_pad:]
        target_ids = np.array(t_enc.input_ids)

        embed_fn = self.model.language_model.model.embed_tokens
        dt = embed_fn.weight.dtype

        prefix_emb = embed_fn(mx.array(prefix_ids)[None, :])
        suffix_emb = embed_fn(mx.array(suffix_ids)[None, :])
        audio_emb_typed = audio_emb.astype(dt)[None, :]
        target_emb = embed_fn(mx.array(target_ids)[None, :])

        inputs_embeds = mx.concatenate([prefix_emb, audio_emb_typed, suffix_emb, target_emb], axis=1)

        # Forward pass through MLX Qwen3.5-9B to get last hidden states
        h_last = self.model.language_model.model(None, input_embeddings=inputs_embeds)[0]

        tgt_start = len(src_text) + 1
        sel = [j for j, (s, e) in enumerate(t_enc.offset_mapping) if e > tgt_start]
        assert sel, "No target tokens selected"

        prompt_len = len(prompt_tokens)
        idx = mx.array(prompt_len + np.array(sel))
        h_st = h_last[idx]
        spans_st = [
            (max(s - tgt_start, 0), max(e - tgt_start, 0))
            for j, (s, e) in enumerate(t_enc.offset_mapping) if j in sel
        ]

        return h_st, spans_st
