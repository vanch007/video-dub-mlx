#!/usr/bin/env python3
"""Entrypoint for running the video-dub-mlx S2ST service (Apple Silicon)."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from video_dub_mlx.serve import main

if __name__ == '__main__':
    main()
