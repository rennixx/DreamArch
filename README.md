# Dream Architect

Turn your hummed melodies into full AI-generated songs.

## Features

- Record 2-3 second melodies via web interface
- Extract pitch using CREPE (optimized for low VRAM)
- Generate instrumentals using MusicGen Small
- Create lyrics with GPT-3.5-turbo
- Synthesize vocals with AI TTS
- Mix and master with professional effects

## Requirements

- **GPU:** 4GB VRAM minimum (RTX 3060, 4060, etc.)
- **RAM:** 8GB+ recommended
- **Storage:** ~5GB for models
- **Python:** 3.9+
- **Node.js:** 18+

## Installation

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=sk-...
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

## Running

### Start Backend

```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd frontend
npm run dev
```

Visit http://localhost:3000

## Usage

1. **Record:** Click the microphone button and hum/whistle for 2-3 seconds
2. **Select Style:** Choose from Pink Floyd, Daft Punk, Billie Eilish, or Frank Zappa
3. **Generate:** Click "Generate Track" and wait ~60 seconds
4. **Download:** Get your MP3 file

## Configuration

Optimized for:
- **VRAM:** 4GB (uses MusicGen Small model)
- **Generation Time:** ~60 seconds
- **Output Duration:** 30-45 seconds

Edit `backend/config/settings.yaml` to adjust settings.

## Project Structure

```
dream-architect/
├── backend/           # Python FastAPI
│   ├── modules/       # Processing modules
│   ├── routers/       # API endpoints
│   ├── config/        # Settings & presets
│   └── outputs/       # Generated tracks
├── frontend/          # Next.js app
│   ├── app/          # Pages
│   └── components/   # React components
└── README.md
```

## API Endpoints

- `POST /api/generate` - Generate track from audio
- `GET /api/styles` - Get available style presets
- `GET /api/health/gpu` - Check GPU status
- `GET /outputs/final/{job_id}.mp3` - Download generated track

## License

MIT
