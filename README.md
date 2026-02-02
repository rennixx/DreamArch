# Dream Architect

> Turn your hummed melodies into full AI-generated songs

**Optimized for 4GB VRAM • ~60s generation time**

---

## Overview

Dream Architect is a full-stack AI music generation application that transforms a 2-3 second hummed melody into a complete 30-45 second produced track with:

- 🎵 **Melody-conditioned instrumental** using MusicGen Small
- 🎤 **AI-generated lyrics** using GPT-3.5-turbo
- 🎙️ **Synthesized vocals** with OpenAI TTS + effects
- 🎚️ **Professional mixing** with Pedalboard
- 📊 **Style presets** for Pink Floyd, Daft Punk, Billie Eilish, Frank Zappa

---

## Features

| Feature | Technology | VRAM Usage |
|---------|------------|------------|
| Pitch Detection | CREPE (small) | ~0.5GB |
| Music Generation | MusicGen Small + fp16 | ~1.5GB |
| Lyrics | GPT-3.5-turbo | CPU |
| Vocals | OpenAI TTS | CPU |
| Mixing | Pedalboard | CPU |
| **Total** | | **~2-2.5GB** |

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **GPU** with 4GB+ VRAM (NVIDIA recommended)
- **OpenAI API key**

### Installation

```bash
# 1. Clone or navigate to project
cd DreamArchitect

# 2. Backend setup
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install core dependencies
pip install fastapi uvicorn python-multipart pydantic python-dotenv pyyaml
pip install numpy scipy librosa soundfile
pip install pretty-midi music21 pedalboard

# Install ML dependencies (requires GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install audiocraft crepe openai

# 3. Frontend setup
cd ../frontend
npm install
```

### Configuration

Create `backend/.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
```

### Running

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Visit **http://localhost:3000**

---

## Usage

1. **Record** your melody (2-3 seconds)
2. **Select** a style (Pink Floyd, Daft Punk, etc.)
3. **Generate** and wait ~60 seconds
4. **Download** your MP3

---

## Project Structure

```
DreamArchitect/
├── backend/                    # Python FastAPI
│   ├── config/
│   │   ├── settings.yaml       # Optimized config
│   │   └── style_presets.json  # Style definitions
│   ├── modules/
│   │   ├── pitch_detector.py   # CREPE
│   │   ├── midi_processor.py   # Hz → MIDI
│   │   ├── music_generator.py  # MusicGen
│   │   ├── lyrics_generator.py # GPT-3.5
│   │   ├── vocal_synth.py      # TTS + effects
│   │   └── mixer.py            # Mixing/mastering
│   ├── routers/generate.py     # API endpoint
│   ├── utils/                  # Audio utils, file manager
│   ├── main.py                 # FastAPI app
│   ├── test_modules.py         # Module tests
│   └── demo_generator.py       # Demo creation
│
├── frontend/                   # Next.js 14
│   ├── app/
│   │   ├── page.tsx            # Main UI
│   │   └── layout.tsx
│   ├── components/
│   │   ├── AudioRecorder.tsx   # Mic recording
│   │   ├── StyleSelector.tsx   # Style presets
│   │   └── AudioPlayer.tsx     # Custom player
│   └── package.json
│
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/generate` | POST | Generate track from audio |
| `/api/styles` | GET | Get style presets |
| `/api/health/gpu` | GET | Check GPU status |
| `/outputs/final/{job_id}.mp3` | GET | Download track |

---

## Style Presets

| Style | Description | Tempo | Effects |
|-------|-------------|-------|---------|
| **Pink Floyd** | Psychedelic space rock | 110 BPM | Reverb, delay |
| **Daft Punk** | Electronic funk | 125 BPM | Vocoder, compression |
| **Billie Eilish** | Dark pop | 95 BPM | Minimal reverb |
| **Frank Zappa** | Experimental rock | 140 BPM | Distortion, wah |

---

## Configuration

Edit `backend/config/settings.yaml`:

```yaml
generation:
  musicgen_model: "facebook/musicgen-small"  # ~1.5GB VRAM
  use_fp16: true                             # Half precision
  output_duration: 45                         # Seconds

mixing:
  target_loudness: -14.0                      # LUFS (streaming)
```

---

## Testing

```bash
# Test core modules (no GPU required)
cd backend
python test_modules.py

# Generate demo track (requires GPU + API key)
python demo_generator.py --style pink-floyd
```

---

## Troubleshooting

### Issue: GPU out of memory
**Solution:**
- Use smaller model: `musicgen-small`
- Reduce duration: 30s instead of 45s
- Close other GPU applications

### Issue: CREPE fails to load
**Solution:**
```bash
# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Slow generation
**Solution:**
- Ensure GPU is being used: Check `/api/health/gpu`
- Use fp16 precision (enabled by default)
- Reduce output duration

### Issue: Vocals sound robotic
**Solution:**
- Use ElevenLabs instead of OpenAI TTS
- Apply more reverb effects
- Lower vocals in mix

---

## Requirements Detail

### Python Dependencies

```
# Core
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0

# Audio
librosa==0.10.1
soundfile==0.12.1
pedalboard==0.8.0

# ML (GPU)
torch==2.1.0
audiocraft==1.3.0
crepe==0.0.13

# Utilities
pretty-midi==0.2.10
music21==9.1.0
openai==1.3.0
python-dotenv==1.0.0
pyyaml==6.0.1
```

### Node Dependencies

```json
{
  "next": "^14.0.0",
  "react": "^18.2.0",
  "wavesurfer.js": "^7.0.0",
  "axios": "^1.6.0"
}
```

---

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| VRAM Usage | < 4GB | ~2.5GB ✅ |
| Generation Time | ~60s | ~57s ✅ |
| Success Rate | > 70% | TBD |
| Output Quality | Shareable | TBD |

---

## Future Enhancements

- [ ] Real-time generation preview
- [ ] Custom voice training
- [ ] Stem-level arrangement control
- [ ] Full song structure (verse/chorus/bridge)
- [ ] Export to DAW (MIDI + stems)
- [ ] Mobile app

---

## License

MIT License - feel free to use for personal or commercial projects.

---

## Credits

Built with:
- [MusicGen](https://github.com/facebookresearch/audiocraft) by Meta
- [CREPE](https://github.com/marl/crepe) pitch detection
- [Pedalboard](https://spotify.github.io/pedalboard/) by Spotify
- [OpenAI API](https://openai.com/) for lyrics and vocals

---

## Support

For issues or questions:
- Check the troubleshooting guide
- Review module tests: `python test_modules.py`
- Enable debug logging in `config/settings.yaml`

---

**Built with 🎵 and 🤖**
