"""ffmpeg helpers: audio extraction, muxing, tempo adjustment, and hardware-accelerated subtitle burning."""

import json
import os
import subprocess
import tempfile
import platform


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        err_msg = proc.stderr[-2000:] if proc.stderr else ''
        raise RuntimeError(f"command failed: {' '.join(cmd)}: {err_msg}")
    return proc


def get_ffmpeg_binary() -> str:
    """Locate an ffmpeg binary with libass subtitle support (referencing VideoLingo step12)."""
    candidates = [
        "/Users/vanch/pinokio/bin/miniconda/envs/videolingo/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1",
    ]
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            candidates.append(exe)
    except Exception:
        pass

    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    try:
        res = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True, timeout=5)
        if "subtitles" in res.stdout:
            return "ffmpeg"
    except Exception:
        pass
    return "ffmpeg"


def probe_duration(path):
    """Return media duration in seconds."""
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path,
    ])
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_audio(video_path, out_wav, sr=16000):
    """Extract mono PCM wav from a video (or audio) file."""
    run([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", out_wav,
    ])
    return out_wav


def atempo_chain(ratio):
    """Build an atempo filter string; each atempo factor must be in [0.5, 2.0]."""
    ratio = max(0.25, min(4.0, ratio))
    parts = []
    r = ratio
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.4f}")
    return ",".join(parts)


def speed_to_duration(in_wav, out_wav, ratio, sr):
    """Time-stretch in_wav by ratio (>1 = faster/shorter) and write sr-Hz wav."""
    run([
        "ffmpeg", "-y", "-i", in_wav, "-af", atempo_chain(ratio),
        "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", out_wav,
    ])
    return out_wav


def mux_video_audio(video_path, audio_path, out_path, audio_codec="aac"):
    """Replace the audio track of video_path with audio_path."""
    run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", audio_codec, "-shortest", out_path,
    ])
    return out_path


def mux_video_audio_with_subtitles(video_path, audio_path, out_path, srt_path=None,
                                    burn_subtitles=True, audio_codec="aac"):
    """Replace audio track and burn subtitles using VideoToolbox hardware acceleration."""
    ff_bin = get_ffmpeg_binary()
    if burn_subtitles and srt_path and os.path.isfile(srt_path):
        # Cinema-grade style referencing VideoLingo step12
        style = "FontSize=15\,FontName=Arial Unicode MS\,PrimaryColour=&H00FFFFFF&\,OutlineColour=&H000000&\,OutlineWidth=1.5\,ShadowColour=&H80000000&\,Alignment=2\,MarginV=18\,BorderStyle=1"
        vf = f"subtitles='{srt_path}':force_style='{style}'"
        cmd = [
            ff_bin, "-y", "-i", video_path, "-i", audio_path,
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "h264_videotoolbox", "-b:v", "4M",
            "-c:a", audio_codec, "-b:a", "192k",
            "-shortest", out_path,
        ]
        try:
            run(cmd)
            return out_path
        except Exception as e:
            print(f"[media] Subtitle burning failed ({e}), falling back to plain muxing...")

    return mux_video_audio(video_path, audio_path, out_path, audio_codec)


def mix_audio(base_wav, over_wav, out_wav, base_vol=1.0, over_vol=1.0, sr=24000):
    """Mix over_wav on top of base_wav (e.g. dubbed voice over instrumental)."""
    run([
        "ffmpeg", "-y", "-i", base_wav, "-i", over_wav,
        "-filter_complex",
        f"[0:a]volume={base_vol}[b];[1:a]volume={over_vol}[o];"
        f"[b][o]amix=inputs=2:duration=first:normalize=0[mix]",
        "-map", "[mix]", "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", out_wav,
    ])
    return out_wav


def extract_segment(in_wav, out_wav, start, end, sr=16000):
    """Cut [start, end) seconds from in_wav."""
    run([
        "ffmpeg", "-y", "-i", in_wav, "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le", out_wav,
    ])
    return out_wav
