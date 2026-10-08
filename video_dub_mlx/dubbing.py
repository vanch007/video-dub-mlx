"""Unified MLX-based Echo-S2ST pipeline for translating and dubbing local video or audio.

Supports vocal separation (Demucs) to isolate clean speech for voice cloning,
and preserves the background music / sound effects to mix back onto the final timeline.
"""

import os
import sys
import tempfile
import time
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import soundfile as sf
import mlx.core as mx

from mlx_echo_s2st.stlm_mlx import MLXEchoSTLM, _has_cjk
from mlx_echo_s2st.cosyvoice_wrapper import MLXCosyVoiceSynthesizer
from mlx_echo_s2st import media, segment, timeline, subtitles, separate

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".ts"}
TARGET_LANGS = ("en", "es", "ja", "zh")


class MLXEchoS2ST:
    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(self.model_dir) or not os.path.exists(os.path.join(self.model_dir, 'bridge', 'mapper.safetensors')):
            print(f"[MLXEchoS2ST] Local model weights not found at {self.model_dir}. Auto-downloading from vanch007/Index-Echo-S2ST-9B-MLX...")
            from huggingface_hub import snapshot_download
            self.model_dir = snapshot_download('vanch007/Index-Echo-S2ST-9B-MLX', local_dir=self.model_dir)

        # Set environment for vendored components
        os.environ["DUBBING_HOME"] = self.model_dir

        llm_path = os.path.join(self.model_dir, "stlm_llm")
        omni_cfg = os.path.join(self.model_dir, "stlm_omni/config.json")
        ckpt_path = os.path.join(self.model_dir, "stlm_ckpt")

        # Initialize MLX STLM
        print("=" * 60)
        print("[MLXEchoS2ST] Initializing MLX Speech Translation Model...")
        self.stlm = MLXEchoSTLM(llm_path=llm_path, omni_config_path=omni_cfg, ckpt_path=ckpt_path)

        # Initialize MLX-backed CosyVoice3 Synthesizer
        print("=" * 60)
        print("[MLXEchoS2ST] Initializing MLX CosyVoice Synthesizer...")
        self.synthesizer = MLXCosyVoiceSynthesizer(self.model_dir)
        self.sample_rate = self.synthesizer.sample_rate
        print("=" * 60)
        print("[MLXEchoS2ST] System Ready for High-Speed Inference!")

    def _normalize_target_text(self, tgt_raw: str, lang: str) -> Tuple[str, Any]:
        """Normalize target text according to language rules."""
        ja_words = None
        if lang == "en":
            try:
                from text_norm import normalize_en
                tgt_cv = normalize_en(tgt_raw)
            except Exception:
                tgt_cv = tgt_raw
        elif lang in ("es", "zh"):
            try:
                from text_norm import strip_meta
                text = strip_meta(tgt_raw)
                tgt_cv = self.synthesizer.cv.frontend.text_normalize(text, split=False)
            except Exception:
                tgt_cv = tgt_raw
        elif lang == "ja":
            try:
                from ja_kata import ja2kata_map
                tgt_cv, ja_words, _ = ja2kata_map(tgt_raw)
            except Exception:
                tgt_cv = tgt_raw
        else:
            tgt_cv = tgt_raw
        return tgt_cv, ja_words

    def dub_clip(self, audio_path: str, lang: str = "en", out_wav: Optional[str] = None) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Dubs a single speech audio clip (<= 30s) preserving the original voice.
        Returns: (waveform_numpy_1d, sample_rate, info)
        """
        assert lang in TARGET_LANGS, f"Invalid target lang {lang}, must be in {TARGET_LANGS}"

        # 1. MLX Speech Translation
        hyp, src_text, tgt_raw, audio_emb, prompt = self.stlm.generate(audio_path, lang=lang)
        src_lang = "zh" if _has_cjk(src_text) else "en"

        # 2. Text normalization
        tgt_cv, ja_words = self._normalize_target_text(tgt_raw, lang)
        tgt_st = (tgt_raw if lang == "ja" else tgt_cv) if src_lang == "zh" else tgt_raw

        # 3. Extract target hidden states with teacher-forcing forward pass in MLX
        h_st, spans_st = self.stlm.extract_hidden_states(prompt, audio_emb, src_text, tgt_st)

        # 4. Synthesize voice-preserved speech
        ext = {
            "wav": audio_path,
            "lang": lang,
            "src_lang": src_lang,
            "zh": src_text,
            "tgt_raw": tgt_raw,
            "tgt_cv": tgt_cv,
            "ja_words": ja_words,
            "h_st": np.array(h_st.astype(mx.float32)),
            "spans_st": spans_st,
        }
        wav_np, info = self.synthesizer.synth(ext, out_wav=out_wav)
        return wav_np, self.sample_rate, info

    def dub_video(self, video_path: str, lang: str = "en", out_video: Optional[str] = None,
                  max_seg: float = 9.5, srt: bool = True, burn_subtitles: bool = True, do_separate: bool = True,
                  keep_bgm: bool = True, bgm_vol: float = 0.9, voice_vol: float = 1.0,
                  speech_wav: Optional[str] = None, instr_wav: Optional[str] = None) -> str:
        """
        Dubs a video into the target language with optional Demucs vocal separation and BGM mixing.
        """
        video_path = os.path.abspath(video_path)
        if not out_video:
            base, ext = os.path.splitext(video_path)
            out_video = f"{base}.dub_{lang}{ext}"
        out_video = os.path.abspath(out_video)

        total_dur = media.probe_duration(video_path)
        print(f"[MLXEchoS2ST] Processing video: {video_path} (Duration: {total_dur:.1f}s)")

        with tempfile.TemporaryDirectory(prefix="mlx_dub_") as workdir:
            # 1. Extract audio
            orig_wav = os.path.join(workdir, "extracted_raw.wav")
            media.extract_audio(video_path, orig_wav, sr=16000)

            # 2. Vocal separation & BGM stem isolation
            vocal_track = orig_wav
            bgm_track = None

            if speech_wav:
                print(f"[MLXEchoS2ST] Reusing speech track: {speech_wav}")
                vocal_16k = os.path.join(workdir, "vocals16k.wav")
                media.extract_audio(speech_wav, vocal_16k, sr=16000)
                vocal_track = vocal_16k
                if instr_wav and keep_bgm:
                    bgm_track = instr_wav
            elif do_separate:
                try:
                    print("[MLXEchoS2ST] Separating vocals and background audio via Demucs...")
                    demucs_dir = os.path.join(workdir, "demucs")
                    raw_vocals, raw_instr = separate.separate_vocals(orig_wav, demucs_dir)
                    vocal_16k = os.path.join(workdir, "vocals16k.wav")
                    media.extract_audio(raw_vocals, vocal_16k, sr=16000)
                    vocal_track = vocal_16k
                    if keep_bgm:
                        bgm_track = raw_instr
                    print(f"[MLXEchoS2ST] Separation complete: vocals isolated, BGM stem preserved.")
                except Exception as e:
                    print(f"[MLXEchoS2ST] Warning: Vocal separation unavailable ({e}), continuing with original audio.")
                    vocal_track = orig_wav
                    bgm_track = None
            else:
                print("[MLXEchoS2ST] Vocal separation disabled, dubbing directly from mixed audio.")

            # Short video (<= 12s)
            if total_dur <= 12.0:
                print(f"[MLXEchoS2ST] Short clip, dubbing in single shot...")
                dub_wav = os.path.join(workdir, "dubbed_voice.wav")
                wav_np, sr, info = self.dub_clip(vocal_track, lang=lang, out_wav=dub_wav)
                print(f"  [Transcript]  {info['zh']}")
                print(f"  [Translation] {info['tgt_raw']}")

                final_audio = dub_wav
                if bgm_track and keep_bgm:
                    print(f"[MLXEchoS2ST] Mixing original background sound with dubbed voice...")
                    mixed_audio = os.path.join(workdir, "mixed_track.wav")
                    media.mix_audio(bgm_track, dub_wav, mixed_audio, base_vol=bgm_vol, over_vol=voice_vol, sr=sr)
                    final_audio = mixed_audio

                media.mux_video_audio(video_path, final_audio, out_video)
                if srt:
                    srt_base = os.path.splitext(out_video)[0]
                    subtitles.write_srt(f"{srt_base}.srt", [(0.0, total_dur)], [info["tgt_raw"]])
                    subtitles.write_bilingual_srt(f"{srt_base}.bilingual.srt", [(0.0, total_dur)], [info["zh"]], [info["tgt_raw"]])
                print(f"[MLXEchoS2ST] Dubbed video saved: {out_video}")
                return out_video

            # Segment long video with VAD on clean vocal track
            print(f"[MLXEchoS2ST] Segmenting speech timeline with Silero VAD (max segment: {max_seg}s)...")
            spans = segment.segment_audio(vocal_track, max_dur=max_seg, backend="silero")
            print(f"[MLXEchoS2ST] Detected {len(spans)} speech segments.")

            dub_wavs = []
            sub_spans, sub_srcs, sub_tgts = [], [], []

            for i, (s, e) in enumerate(spans):
                dur = e - s
                seg_in = os.path.join(workdir, f"seg_{i:04d}_in.wav")
                seg_out = os.path.join(workdir, f"seg_{i:04d}_out.wav")
                media.extract_segment(vocal_track, seg_in, s, e, sr=16000)

                print(f"\n[Segment {i+1}/{len(spans)}] Time: {s:.2f}s - {e:.2f}s (len={dur:.2f}s)")
                try:
                    w, sr, info = self.dub_clip(seg_in, lang=lang, out_wav=seg_out)
                    print(f"  [Original]   {info['zh']}")
                    print(f"  [Translated] {info['tgt_raw']}")
                    dub_wavs.append(seg_out)
                    sub_spans.append((s, e))
                    sub_srcs.append(info.get("zh", ""))
                    sub_tgts.append(info.get("tgt_raw", ""))
                except Exception as err:
                    print(f"  [Warning] Segment {i+1} dubbing skipped/fallback: {err}")
                    dub_wavs.append(None)

            # Re-assemble dubbed dialogue timeline
            print("\n[MLXEchoS2ST] Assembling dubbed audio onto timeline...")
            full_track, sr = timeline.build_track(spans, dub_wavs, total_dur, workdir, sr=self.sample_rate)
            dubbed_audio_path = os.path.join(workdir, "full_dubbed_track.wav")
            sf.write(dubbed_audio_path, full_track, sr)

            final_audio = dubbed_audio_path
            # Mix background audio back under the dubbed voice
            if bgm_track and keep_bgm:
                print(f"[MLXEchoS2ST] Mixing original background sound ({bgm_vol*100:.0f}%) with dubbed speech ({voice_vol*100:.0f}%)...")
                mixed_audio = os.path.join(workdir, "final_mixed_track.wav")
                media.mix_audio(bgm_track, dubbed_audio_path, mixed_audio, base_vol=bgm_vol, over_vol=voice_vol, sr=sr)
                final_audio = mixed_audio

            # Export subtitles & segments.json manifest (aligning with Index-Translate/video-dub)
            bi_srt = None
            if srt and sub_spans:
                srt_base = os.path.splitext(out_video)[0]
                subtitles.write_srt(f"{srt_base}.srt", sub_spans, sub_tgts)
                bi_srt = f"{srt_base}.bilingual.srt"
                subtitles.write_bilingual_srt(bi_srt, sub_spans, sub_srcs, sub_tgts)
                manifest_path = f"{srt_base}.segments.json"
                with open(manifest_path, 'w', encoding='utf-8') as mf:
                    json.dump([{"start": s, "end": e, "src": src, "text": tgt}
                               for (s, e), src, tgt in zip(sub_spans, sub_srcs, sub_tgts)],
                              mf, ensure_ascii=False, indent=2)
                print(f"[MLXEchoS2ST] Subtitles & Manifest saved: {srt_base}.srt, {bi_srt}, {manifest_path}")

            # Mux back into video container (burning bilingual subtitles by default)
            if burn_subtitles and bi_srt and os.path.isfile(bi_srt):
                print(f"[MLXEchoS2ST] Burning bilingual subtitles into video with VideoToolbox hardware acceleration...")
                media.mux_video_audio_with_subtitles(video_path, final_audio, out_video, srt_path=bi_srt, burn_subtitles=True)
            else:
                print(f"[MLXEchoS2ST] Muxing final audio track into video container...")
                media.mux_video_audio(video_path, final_audio, out_video)

            print(f"[MLXEchoS2ST] Successfully generated dubbed video: {out_video}")
            return out_video

    def dub(self, input_path: str, lang: str = "en", out_path: Optional[str] = None, **kwargs) -> str:
        """
        Universal dubbing entry point: handles either video or audio.
        """
        input_path = os.path.abspath(input_path)
        assert os.path.exists(input_path), f"File not found: {input_path}"

        ext = os.path.splitext(input_path)[1].lower()
        if ext in VIDEO_EXTS:
            return self.dub_video(input_path, lang=lang, out_video=out_path, **kwargs)
        else:
            if not out_path:
                base = os.path.splitext(input_path)[0]
                out_path = f"{base}.dub_{lang}.wav"
            out_path = os.path.abspath(out_path)
            dur = media.probe_duration(input_path)
            if dur <= 15.0:
                self.dub_clip(input_path, lang=lang, out_wav=out_path)
            else:
                with tempfile.TemporaryDirectory(prefix="mlx_dub_") as workdir:
                    in_16k = os.path.join(workdir, "in_16k.wav")
                    media.extract_audio(input_path, in_16k, sr=16000)
                    spans = segment.segment_audio(in_16k, max_dur=9.5, backend="silero")
                    dub_wavs = []
                    for i, (s, e) in enumerate(spans):
                        seg_in = os.path.join(workdir, f"seg_{i:04d}_in.wav")
                        seg_out = os.path.join(workdir, f"seg_{i:04d}_out.wav")
                        media.extract_segment(in_16k, seg_in, s, e, sr=16000)
                        try:
                            w, sr, info = self.dub_clip(seg_in, lang=lang, out_wav=seg_out)
                            dub_wavs.append(seg_out)
                        except Exception:
                            dub_wavs.append(None)
                    full_track, sr = timeline.build_track(spans, dub_wavs, dur, workdir, sr=self.sample_rate)
                    sf.write(out_path, full_track, sr)
            print(f"[MLXEchoS2ST] Dubbed audio saved: {out_path}")
            return out_path
