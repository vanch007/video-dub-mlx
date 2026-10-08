#!/usr/bin/env python3
"""CLI for Echo-S2ST MLX: Speech-to-Speech Translation & Video Dubbing on Apple Silicon."""

import argparse
import os
import sys

# Ensure echo-s2st-mlx is in sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlx_echo_s2st.dubbing import MLXEchoS2ST, TARGET_LANGS, VIDEO_EXTS


def main():
    parser = argparse.ArgumentParser(
        description="Echo-S2ST MLX: Speech-to-speech translation and video dubbing preserving the original speaker's voice."
    )
    parser.add_argument("input", help="Path to input audio (.wav, .mp3, etc.) or video (.mp4, .mkv, .mov, etc.)")
    parser.add_argument("-l", "--lang", required=True, choices=TARGET_LANGS,
                        help="TARGET language: en (English), es (Spanish), ja (Japanese), zh (Chinese)")
    parser.add_argument("-o", "--out", default=None,
                        help="Output path (default: <input>.dub_<lang>.<ext>)")
    parser.add_argument("-m", "--model-dir",
                        default=os.environ.get("S2ST_MODEL_DIR", os.path.join(_ROOT, "weights/Index-Echo-S2ST-9B")),
                        help="Path to downloaded Index-Echo-S2ST-9B directory")
    parser.add_argument("--max-seg", type=float, default=9.5,
                        help="Max segment duration in seconds for VAD splitting (default: 9.5)")
    parser.add_argument("--separate", dest="separate", action="store_true", default=True,
                        help="Separate vocals with Demucs first to ensure clean speech cloning (default: on)")
    parser.add_argument("--no-separate", dest="separate", action="store_false",
                        help="Skip vocal separation (faster; directly dubs from mixed audio)")
    parser.add_argument("--no-bgm", action="store_true", default=False,
                        help="Drop the background audio stem instead of mixing it back under the dubbed speech")
    parser.add_argument("--bgm-vol", type=float, default=0.9,
                        help="Volume factor of background sound / music (default: 0.9)")
    parser.add_argument("--voice-vol", type=float, default=1.0,
                        help="Volume factor of dubbed speech voice (default: 1.0)")
    parser.add_argument("--speech-wav", default=None,
                        help="Reuse an existing pre-separated vocal wav file")
    parser.add_argument("--instr-wav", default=None,
                        help="Reuse an existing pre-separated background instrumental/ambient wav file")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        parser.error(f"Input file not found: {input_path}")

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir) or not os.path.exists(os.path.join(model_dir, 'bridge', 'mapper.safetensors')):
        print(f"[Echo-S2ST] Local model weights not found at {model_dir}. Auto-downloading from vanch007/Index-Echo-S2ST-9B-MLX...")
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download('vanch007/Index-Echo-S2ST-9B-MLX', local_dir=model_dir)

    print("=" * 60)
    print(f"Echo-S2ST MLX (Apple Silicon M3 Max Native Acceleration)")
    print(f"Input File:        {input_path}")
    print(f"Target Language:   {args.lang}")
    print(f"Model Directory:   {model_dir}")
    print(f"Vocal Separation:  {'Enabled (Demucs)' if args.separate else 'Disabled'}")
    print(f"BGM Preservation:  {'Enabled' if not args.no_bgm else 'Disabled'}")
    print("=" * 60)

    model = MLXEchoS2ST(model_dir)

    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTS:
        out_path = model.dub_video(
            input_path,
            lang=args.lang,
            out_video=args.out,
            max_seg=args.max_seg,
            do_separate=args.separate,
            keep_bgm=not args.no_bgm,
            bgm_vol=args.bgm_vol,
            voice_vol=args.voice_vol,
            speech_wav=args.speech_wav,
            instr_wav=args.instr_wav,
        )
    else:
        out_path = model.dub(input_path, lang=args.lang, out_path=args.out)

    print("\n" + "=" * 60)
    print(f"Task Complete! Output successfully saved to:\n  {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
