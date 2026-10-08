"""Apple Silicon MLX-powered S2ST HTTP service compatible with official Index-Translate video-dub API."""

import argparse
import base64
import os
import subprocess
import sys
import tempfile
import threading
import traceback

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from video_dub_mlx.dubbing import MLXEchoS2ST, TARGET_LANGS

LANGS = TARGET_LANGS
S2ST_MAX_S = 10.5
S2TT_MAX_S = 60.0
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def to_wav16k(raw: bytes, suffix: str) -> str:
    suffix = suffix if suffix and len(suffix) <= 8 else '.bin'
    fd, src = tempfile.mkstemp(prefix='s2s_in_', suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw)
    out = src + '.16k.wav'
    r = subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', src,
                        '-ac', '1', '-ar', '16000', out],
                       capture_output=True, text=True)
    os.unlink(src)
    if r.returncode != 0:
        raise RuntimeError('audio decode failed: ' + (r.stderr or '')[-200:])
    return out


def wav_duration(path: str) -> float:
    r = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries',
                        'format=duration', '-of', 'csv=p=0', path],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0.0)


def create_app(model_dir: str = None) -> FastAPI:
    if not model_dir:
        model_dir = os.environ.get('S2ST_MODEL_DIR', os.path.join(_ROOT, 'weights/Index-Echo-S2ST-9B'))
    print(f'[serve_s2st] Initializing MLXEchoS2ST from {model_dir}...', flush=True)
    model = MLXEchoS2ST(model_dir)
    print('[serve_s2st] MLX Model ready!', flush=True)

    lock = threading.Lock()
    app = FastAPI(title='video-dub-mlx S2ST Service')

    @app.middleware('http')
    async def reject_oversized(request, call_next):
        declared = request.headers.get('content-length')
        if declared:
            try:
                too_big = int(declared) > MAX_UPLOAD_BYTES
            except ValueError:
                too_big = False
            if too_big:
                return JSONResponse(
                    {'ok': False, 'error': f'upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit'}, 413)
        return await call_next(request)

    async def read_audio(file: UploadFile):
        raw = await file.read()
        if not raw:
            raise RuntimeError('empty file')
        if len(raw) > MAX_UPLOAD_BYTES:
            raise RuntimeError('file too large')
        src = to_wav16k(raw, os.path.splitext(file.filename or '')[1])
        return src, wav_duration(src)

    @app.get('/healthz')
    def healthz():
        return {'ok': True, 'langs': list(LANGS), 'backend': 'mlx-apple-silicon'}

    @app.post('/s2tt')
    async def s2tt(file: UploadFile, lang: str = Form('en')):
        if lang not in LANGS:
            return JSONResponse({'ok': False, 'error': f'lang must be one of {LANGS}'}, 400)
        src = None
        try:
            src, d = await read_audio(file)
            if d > S2TT_MAX_S:
                return JSONResponse({'ok': False, 'error': f'audio {d:.1f}s exceeds {S2TT_MAX_S:.0f}s limit'}, 400)
            with lock:
                res = model.stlm.translate(src, lang=lang)
            return {'ok': True, 'zh': res['zh'], 'text': res['tgt_raw'], 'lang': lang, 'dur': round(d, 2)}
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({'ok': False, 'error': str(e)[-400:]}, 500)
        finally:
            if src and os.path.exists(src):
                os.unlink(src)

    @app.post('/s2st')
    async def s2st(file: UploadFile, lang: str = Form('en')):
        if lang not in LANGS:
            return JSONResponse({'ok': False, 'error': f'lang must be one of {LANGS}'}, 400)
        src = out = None
        try:
            src, d = await read_audio(file)
            if d > S2ST_MAX_S:
                return JSONResponse({'ok': False, 'error': f'audio {d:.1f}s exceeds {S2ST_MAX_S:.0f}s limit'}, 400)
            fd, out = tempfile.mkstemp(prefix='s2s_out_', suffix='.wav')
            os.close(fd)
            with lock:
                w, sr, info = model.dub_clip(src, lang=lang, out_wav=out)
            with open(out, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            return {
                'ok': True,
                'audio_b64': audio_b64,
                'sr': sr,
                'zh': info.get('zh', ''),
                'text': info.get('tgt_raw', ''),
                'lang': lang,
                'gen_s': info.get('gen_s'),
                'hit_eos': info.get('hit_eos')
            }
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({'ok': False, 'error': str(e)[-400:]}, 500)
        finally:
            for p in (src, out):
                if p and os.path.exists(p):
                    os.unlink(p)

    return app


def main():
    parser = argparse.ArgumentParser(description='Run video-dub-mlx S2ST service on Apple Silicon')
    parser.add_argument('--model-dir', default=None, help='Path to Index-Echo-S2ST-9B weights directory')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8094, help='Port to listen on (default: 8094)')
    args = parser.parse_args()

    app = create_app(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
