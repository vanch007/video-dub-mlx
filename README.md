# Echo-S2ST MLX · Apple Silicon 语音与视频同传配音系统

基于 Apple Silicon MLX 深度优化的 **Echo-S2ST-9B** 语音到语音同传配音实现。参考 [bilibili/Index-Translate](https://github.com/bilibili/Index-Translate/tree/main/inference/echo-s2st) 与 [Index-Echo-S2ST-9B](https://huggingface.co/IndexTeam/Index-Echo-S2ST-9B)，可在 Mac (Apple Silicon M系列芯片) 上原生运行，将中文/英文语音或视频直接配音为目标语言（英/西/日/中），并保留原说话人音色。

## 核心特性

- ⚡️ **MLX 原生加速**：将 9B STLM（Qwen3.5-9B）与 Hidden2CV Mapper 完全移植并运行于 Apple Silicon 统一内存架构，摆脱 CUDA 显存限制。
- 🎙️ **原声克隆配音**：自包含 Hidden2CV Mapper + CosyVoice3，合成目标语种时保留原声说话人的音色质感与语调特征。
- 🎬 **视频与音频全能**：支持指定本地任意视频（.mp4, .mov, .mkv 等）或音频（.wav, .mp3 等）。
- ⏱️ **时间轴自动对齐**：针对长视频内置 Silero-VAD 语句级切分、分段翻译配音、时间轴动态拉伸对齐与音视频容器封装。
- 📝 **字幕自动导出**：配音同时自动生成单语与双语 .srt 字幕。

## 架构

1. **Audio Tower (AuT) + Connector**：128-bin Mel 提取 -> 32层 Conformer 编码 -> 4096维语音 Embedding。
2. **MLX STLM (Qwen3.5-9B)**：多模态输入 -> 自回归解码转写与翻译 -> 教师强制提取目标语隐层表征。
3. **MLX Hidden2CV Mapper**：3层 Pre-LN Transformer 将 STLM 隐层映射至语音生成空间。
4. **CosyVoice3**：基于映射特征与原音频说话人 Embedding 生成 24kHz 高保真目标语语音。
5. **Timeline Assembler**：多片段时序组装与原画音视频封装。

## 快速使用

### 命令行 CLI

```bash
# 1. 视频直接配音为英语
python echo_s2st_cli.py /path/to/video.mp4 --lang en -o /path/to/output_en.mp4

# 2. 音频配音为中文
python echo_s2st_cli.py /path/to/english.wav --lang zh -o /path/to/dub_zh.wav
```

### Python API

```python
from mlx_echo_s2st import MLXEchoS2ST

s2st = MLXEchoS2ST("weights/Index-Echo-S2ST-9B")

# 配音短音频
wav, sr, info = s2st.dub_clip("input.wav", lang="en", out_wav="output.wav")
print("原文:", info["zh"])
print("译文:", info["tgt_raw"])

# 配音长视频
s2st.dub_video("input.mp4", lang="en", out_video="output_dubbed.mp4")
```
