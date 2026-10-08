#!/usr/bin/env python3
"""video-dub-mlx: Unified Video & Audio Dubbing on Apple Silicon (MLX Native + Official video-dub Compatible)."""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from video_dub_mlx import media, segment, separate, subtitles, timeline
from video_dub_mlx.dubbing import MLXEchoS2ST, TARGET_LANGS, VIDEO_EXTS
from video_dub_mlx.s2st import S2STClient


def main():
    parser = argparse.ArgumentParser(
        description="video-dub-mlx: Translate and dub videos with voice cloning on Apple Silicon.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Input video (mp4, mkv, mov) or audio file")
    parser.add_argument("-l", "--lang", required=True, choices=TARGET_LANGS,
                        help="Target language (en: English, es: Spanish, ja: Japanese, zh: Chinese)")
    parser.add_argument("-o", "--output", "--out", dest="output", default=None,
                        help="Output file path (default: <input>.<lang>.mp4)")
    parser.add_argument("--s2st-url", default=os.environ.get("S2ST_URL", "local"),
                        help="S2ST service URL. Set to 'local' (default) to run in-process on Apple Silicon MLX GPU directly.")
    parser.add_argument("--model-dir", default=os.environ.get("S2ST_MODEL_DIR", os.path.join(_ROOT, "weights/Index-Echo-S2ST-9B")),
                        help="Path to Index-Echo-S2ST-9B weights (auto-downloads from HF if missing in local mode)")
    parser.add_argument("--endpoint", default="/s2st", choices=["/s2st", "/s2tt"],
                        help="Service endpoint (/s2st for full dubbing, /s2tt for text translation only)")
    parser.add_argument("--max-seg", type=float, default=9.5,
                        help="Max segment duration in seconds for VAD splitting (default: 9.5)")
    parser.add_argument("--max-stretch", type=float, default=1.5,
                        help="Max time-stretch factor to fit a dub into its slot (default: 1.5)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel requests when using remote HTTP S2ST service (default: 4)")
    parser.add_argument("--vad", default="auto", choices=["auto", "silero", "ffmpeg"],
                        help="Segmentation backend (default: auto)")
    parser.add_argument("--separate", dest="separate", action="store_true", default=True,
                        help="Separate vocals with Demucs first to ensure clean speech cloning (default: on)")
    parser.add_argument("--no-separate", dest="separate", action="store_false",
                        help="Skip vocal separation (faster; directly dubs from mixed audio)")
    parser.add_argument("--no-bgm", action="store_true", default=False,
                        help="Drop background audio stem instead of mixing it back under dubbed speech")
    parser.add_argument("--bgm-vol", type=float, default=0.9,
                        help="Volume factor of background sound / music (default: 0.9)")
    parser.add_argument("--voice-vol", type=float, default=1.0,
                        help="Volume factor of dubbed speech voice (default: 1.0)")
    parser.add_argument("--burn-subtitles", dest="burn_subtitles", action="store_true", default=True,
                        help="Burn bilingual subtitles into video frames with Apple Silicon VideoToolbox (default: on)")
    parser.add_argument("--no-burn-subtitles", dest="burn_subtitles", action="store_false",
                        help="Do not burn subtitles into video (only export .srt files)")
    parser.add_argument("--srt", action="store_true", default=True,
                        help="Export .srt and .bilingual.srt subtitles (default: on)")
    parser.add_argument("--speech-wav", default=None,
                        help="Reuse an existing pre-separated vocal wav file")
    parser.add_argument("--instr-wav", default=None,
                        help="Reuse an existing pre-separated background instrumental wav file")
    parser.add_argument("--keep-temp", action="store_true", default=False,
                        help="Keep temporary intermediate directory")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        parser.error(f"Input file not found: {input_path}")

    if not args.output:
        base, ext = os.path.splitext(input_path)
        args.output = f"{base}.{args.lang}{ext}"
    out_path = os.path.abspath(args.output)

    is_video = os.path.splitext(input_path)[1].lower() in VIDEO_EXTS

    print("=" * 65)
    print("video-dub-mlx · Apple Silicon Native Speech & Video Dubbing")
    print(f"Input:            {input_path}")
    print(f"Target Lang:      {args.lang}")
    print(f"Output:           {out_path}")
    print(f"Mode:             {'In-Process Apple Silicon MLX GPU' if args.s2st_url == 'local' else f'Remote Service ({args.s2st_url})'}")
    print(f"Vocal Separation: {'Enabled (Demucs MPS)' if args.separate else 'Disabled'}")
    print(f"BGM Preservation: {'Enabled (vol=' + str(args.bgm_vol) + ')' if not args.no_bgm else 'Disabled'}")
    print(f"Burn Subtitles:   {'Enabled (VideoToolbox)' if (is_video and args.burn_subtitles) else 'Disabled'}")
    print("=" * 65)

    # Branch 1: Local In-Process MLX Mode (Default, Highest Performance on Apple Silicon)
    if args.s2st_url.lower() == "local":
        model = MLXEchoS2ST(args.model_dir)
        if is_video:
            res = model.dub_video(
                input_path,
                lang=args.lang,
                out_video=out_path,
                max_seg=args.max_seg,
                srt=args.srt,
                burn_subtitles=args.burn_subtitles,
                do_separate=args.separate,
                keep_bgm=not args.no_bgm,
                bgm_vol=args.bgm_vol,
                voice_vol=args.voice_vol,
                speech_wav=args.speech_wav,
                instr_wav=args.instr_wav
            )
        else:
            res = model.dub(input_path, lang=args.lang, out_path=out_path)
        print(f"Task Complete! Finished video-dub-mlx output: {res}")
        return

    # Branch 2: Client Mode talking to S2ST HTTP Service (Aligned with Official video-dub)
    client = S2STClient(args.s2st_url)
    try:
        hz = client.healthz()
        print(f"[video-dub] S2ST service reachable: {hz}")
    except Exception as e:
        sys.exit(f"[video-dub] ERROR: Cannot reach S2ST service at {args.s2st_url}: {e}")

    tmp = tempfile.mkdtemp(prefix="dub_cli_")
    try:
        # 1. extract audio
        in_16k = os.path.join(tmp, "in.16k.wav")
        media.extract_audio(input_path, in_16k, sr=16000)
        total_dur = media.probe_duration(in_16k)

        # 2. vocal separation
        speech_wav = args.speech_wav
        instr_wav = args.instr_wav
        if speech_wav is None:
            if args.separate:
                print("[2/6] separating vocals with demucs (MPS/CPU)...")
                speech_wav, instr_wav = separate.separate_vocals(in_16k, tmp)
            else:
                speech_wav = in_16k
                instr_wav = None

        # 3. segmentation
        print(f"[3/6] segmenting audio (backend={args.vad}, max_dur={args.max_seg}s)...")
        spans = segment.segment_audio(speech_wav, backend=args.vad, max_dur=args.max_seg)
        if not spans:
            sys.exit("No speech segments detected in audio!")
        print(f"      found {len(spans)} segments")

        # 4. per-segment dubbing via client
        print(f"[4/6] dubbing {len(spans)} segments via {args.s2st_url} (workers={args.workers})...")
        seg_dir = os.path.join(tmp, "segs")
        os.makedirs(seg_dir, exist_ok=True)
        dub_wavs = [None] * len(spans)
        translations = [None] * len(spans)
        sources = [None] * len(spans)

        def dub_one(i, s, e):
            seg_in = os.path.join(seg_dir, f"in_{i:04d}.wav")
            media.extract_segment(speech_wav, seg_in, s, e)
            try:
                r = client.dub(seg_in, args.lang, endpoint=args.endpoint)
            except Exception as err:
                print(f"      [{i+1}/{len(spans)}] {s:.1f}-{e:.1f}s FAILED: {err}")
                dub_wavs[i] = seg_in
                translations[i] = ""
                sources[i] = ""
                return
            b = r.get("_wav_bytes")
            if b:
                seg_out = os.path.join(seg_dir, f"out_{i:04d}.wav")
                with open(seg_out, "wb") as f:
                    f.write(b)
                dub_wavs[i] = seg_out
            else:
                dub_wavs[i] = seg_in
            translations[i] = r.get("text", "")
            sources[i] = r.get("zh", "")
            print(f"      [{i+1}/{len(spans)}] {s:.1f}-{e:.1f}s | {sources[i][:30]} -> {translations[i][:30]}")

        if args.workers <= 1:
            for i, (s, e) in enumerate(spans):
                dub_one(i, s, e)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: dub_one(item[0], item[1][0], item[1][1]), enumerate(spans)))

        # 5. timeline assembly
        print("[5/6] assembling dubbed timeline...")
        track, sr = timeline.build_track(spans, dub_wavs, total_dur, tmp, sr=24000, max_stretch=args.max_stretch)
        dub_track = os.path.join(tmp, "dub_track.wav")
        import soundfile as sf
        sf.write(dub_track, track, sr, subtype="PCM_16")
        if instr_wav and not args.no_bgm:
            mixed = os.path.join(tmp, "final_track.wav")
            media.mix_audio(instr_wav, dub_track, mixed, base_vol=args.bgm_vol, over_vol=args.voice_vol, sr=sr)
            dub_track = mixed

        # 6. subtitles & export
        srt_base = os.path.splitext(out_path)[0]
        bi_srt = f"{srt_base}.bilingual.srt"
        if args.srt:
            ok_spans = [sp for sp, t in zip(spans, translations) if t]
            ok_tgts = [t for t in translations if t]
            ok_srcs = [s for s in sources if s]
            subtitles.write_srt(f"{srt_base}.srt", ok_spans, ok_tgts)
            subtitles.write_bilingual_srt(bi_srt, ok_spans, ok_srcs, ok_tgts)
            manifest = f"{srt_base}.segments.json"
            with open(manifest, "w", encoding="utf-8") as mf:
                json.dump([{"start": s, "end": e, "src": src, "text": t}
                           for (s, e), src, t in zip(spans, sources, translations)],
                          mf, ensure_ascii=False, indent=2)
            print(f"      subtitles and manifest saved: {srt_base}.srt, {bi_srt}, {manifest}")

        # 7. mux & subtitle burning
        print("[6/6] muxing final video container...")
        if is_video:
            if args.burn_subtitles and os.path.isfile(bi_srt):
                media.mux_video_audio_with_subtitles(input_path, dub_track, out_path, srt_path=bi_srt, burn_subtitles=True)
            else:
                media.mux_video_audio(input_path, dub_track, out_path)
        else:
            shutil.copyfile(dub_track, out_path)

        print(f"Task Complete! Output successfully saved to: {out_path}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
