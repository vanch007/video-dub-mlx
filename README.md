# Echo-S2ST MLX · 视频翻译配音工作流 (video-dub MLX 原生版)

本项目是 [bilibili/Index-Translate/video-dub](https://github.com/bilibili/Index-Translate/tree/main/video-dub) 与 [inference/echo-s2st](https://github.com/bilibili/Index-Translate/tree/main/inference/echo-s2st) 的 Apple Silicon MLX 原生加速版。

- Hugging Face 模型权重: [vanch007/Index-Echo-S2ST-9B-MLX](https://huggingface.co/vanch007/Index-Echo-S2ST-9B-MLX)
- GitHub 开源代码库: [vanch007/video-dub-mlx](https://github.com/vanch007/video-dub-mlx)

## 核心特性

- ⚡️ Apple Silicon MLX 原生端到端: 9B STLM 与 Hidden2CV Mapper 直接运行在 Apple Silicon 统一内存上，无需外部 CUDA 推理服务器。
- 🎙️ 原声克隆配音: 保持原说话人的音色质感、语调与情感。
- 🎼 人声分离与伴奏回贴: Demucs 分离人声克隆，伴奏自动混音回贴。
- 🔤 默认双语字幕硬烧录: 采用 VideoToolbox 硬件加速将中英双语字幕直接烧录入视频像素中。
- 📄 产物自动导出: 自动生成 .mp4、.srt、.bilingual.srt 与 .segments.json。

## 快速上手

```bash
# 对齐官方 video-dub 习惯
python dub_video.py input.mp4 --lang en -o output_dubbed.mp4
```
