"""Optional vocal and background separation (demucs).

Isolates vocals from background music/effects so S2ST only clones clean voice,
and preserves the instrumental/ambient stem to mix back under the dubbed voice.
"""

import os
import shutil
import subprocess
import sys
import torch


def separate_vocals(in_wav: str, workdir: str, model: str = "htdemucs", device: str = None) -> tuple:
    """Run demucs two-stem separation.

    Returns: (vocals_wav, instrumental_wav)
    """
    if not device:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    cmd = [
        sys.executable, "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", model,
        "-o", workdir,
        "-d", device,
        in_wav
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Fallback to cpu if mps fails
        if device == "mps":
            cmd[-2] = "cpu"
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            raise RuntimeError(f"demucs failed: {(e.stderr or e.stdout or '').strip()[-500:]}") from e

    stem = os.path.join(workdir, model, os.path.splitext(os.path.basename(in_wav))[0])
    vocals = os.path.join(stem, "vocals.wav")
    instr = os.path.join(stem, "no_vocals.wav")
    if not os.path.exists(vocals) or not os.path.exists(instr):
        raise RuntimeError(f"demucs output missing in {stem}")
    return vocals, instr
