# 🎙️ VoiceForge — Neural Voice Synthesis & Cloning Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a4hamza/voiceforge-neural-engine/blob/main/VoiceForge_Cloud_Colab.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=a4hamza/voiceforge-neural-engine)

A high-fidelity, reverse-engineered neural text-to-speech (TTS) and voice cloning backend based on Kyutai's **Pocket-TTS** (24kHz neural audio). Engineered to run 100% crash-free with pure in-memory WAV generation and zero external ffmpeg binary dependencies.

---

## ⚡ 1-Click Free Cloud Deployment (Zero PC Required)

### Option 1: Google Colab (Free T4 GPU / 12GB RAM)
1. Click the **Open In Colab** badge above: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a4hamza/voiceforge-neural-engine/blob/main/VoiceForge_Cloud_Colab.ipynb)
2. Press **Play (▶)** on **Cell 1** to install dependencies.
3. Press **Play (▶)** on **Cell 2** to start the neural engine.
4. You will get a free public HTTPS Cloudflare link (`https://xxxx.trycloudflare.com`) that connects directly to your frontend or mobile app.

### Option 2: GitHub Codespaces (Free 60 Hours/Month Cloud Linux)
1. Click the **Open in GitHub Codespaces** badge above: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=a4hamza/voiceforge-neural-engine)
2. Codespaces automatically builds the Python 3.10 environment with PyTorch and Pocket-TTS.
3. Run `python app.py`.
4. Forward port `7860` as **Public** to get your live HTTPS endpoint (`https://<codespace>-7860.app.github.dev`).

---

## 🚀 Features

* **High-Fidelity 24kHz Synthesis**: Crystal clear speech using Flow Matching neural generation.
* **Instant Voice Cloning**: Upload any 3-10 second audio sample to extract vocal embeddings and synthesize custom speech in that exact voice.
* **33 Built-in Voices**: Full Les Misérables character suite (`alba`, `cosette`, `fantine`, `marius`, `javert`, `jean`, etc.) + Japanese acoustic presets (`jp_sakura`, `jp_kenji`, etc.).
* **Pure In-Memory Audio Pipeline**: Uses `scipy.io.wavfile` in RAM — eliminates `[Errno 2] No such file or directory: 'ffmpeg'` crashes completely.
* **FastAPI Backend**: Interactive Swagger UI documentation at `/docs`.

---

## 📡 API Reference

### Health & Setup
* `GET /api/health` — Returns server status, model readiness, and total active voices.
* `GET /api/setup-status` — Setup and loading progress.
* `GET /api/clone-status` — Availability of voice cloning weights.

### Speech Generation & Voices
* `POST /api/generate` (or `/api/tts`) — Synthesize speech tensors into binary 24kHz WAV.
  ```json
  {
    "text": "Hello world, this is a neural voice test.",
    "voice_id": "alba",
    "speed": 1.0,
    "pitch": 1.0
  }
  ```
* `GET /api/voices` — Returns list of all 33 built-in voices + custom user-cloned voices.
* `GET /api/preview/{voice_id}` — Generates preview audio sample for any voice.

### Custom Voice Cloning
* `POST /api/clone` — Multipart form-data with `audio` (file), `name` (string), and `language` (string). Extracts safetensors conditioning state and registers the new voice.
* `GET /api/cloned-voices` — Retrieve custom cloned voices.
* `DELETE /api/cloned-voices/{voice_id}` — Delete a cloned voice.

---

## 🐳 Docker Deployment

```bash
docker build -t voiceforge-neural .
docker run -p 7860:7860 voiceforge-neural
```

---

## 📄 License
MIT License. Built with [Kyutai Pocket-TTS](https://github.com/kyutai-labs/pocket-tts).
