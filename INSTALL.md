# Dream Architect - Installation Guide

Complete setup instructions for Dream Architect.

---

## System Requirements

### Minimum
- **OS:** Windows 10/11, macOS 12+, or Linux
- **CPU:** 4-core processor
- **RAM:** 8GB
- **GPU:** NVIDIA GPU with 4GB VRAM (RTX 3060 or better)
- **Storage:** 10GB free space

### Recommended
- **OS:** Windows 11, macOS 14+, or Ubuntu 22.04
- **CPU:** 8-core processor
- **RAM:** 16GB
- **GPU:** NVIDIA RTX 4070 (12GB VRAM)
- **Storage:** 20GB SSD

---

## Step 1: Install Prerequisites

### Windows

1. **Python 3.9+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Check "Add Python to PATH" during installation

2. **Node.js 18+**
   - Download from [nodejs.org](https://nodejs.org/)
   - Choose LTS version

3. **Git** (optional)
   - Download from [git-scm.com](https://git-scm.com/)

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install Node.js
brew install node
```

### Linux (Ubuntu/Debian)

```bash
# Update packages
sudo apt update

# Install Python
sudo apt install python3 python3-pip python3-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install system dependencies
sudo apt install ffmpeg libsndfile1 portaudio19-dev
```

---

## Step 2: Install CUDA (NVIDIA GPUs)

### Windows

1. Download [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads)
2. Run installer and select:
   - CUDA Toolkit
   - CUDA Runtime
   - CUDA Development

### Linux

```bash
# Download and run CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent --override

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation

```bash
nvidia-smi
```

---

## Step 3: Project Setup

### Clone or Download

```bash
# If using Git
git clone <repository-url>
cd DreamArchitect

# Or download and extract ZIP
```

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements (core only first)
pip install fastapi uvicorn python-multipart pydantic
pip install python-dotenv pyyaml
pip install numpy scipy librosa soundfile
pip install pretty-midi music21 pedalboard

# Install ML dependencies (requires GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install audiocraft crepe
pip install openai
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

---

## Step 4: Configuration

### Create Environment File

Create `backend/.env`:

```bash
# OpenAI API Key (required for lyrics and vocals)
OPENAI_API_KEY=sk-your-key-here

# Optional: Stability AI (alternative to MusicGen)
# STABILITY_API_KEY=your-key-here
```

### Get OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create new secret key
5. Copy to `.env` file

---

## Step 5: Verify Installation

### Test Core Modules

```bash
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python test_modules.py
```

Expected output:
```
[1/6] Audio Utils         [OK]
[2/6] File Manager        [OK]
[3/6] MIDI Processor      [OK]
[4/6] Audio Mixer         [OK]
[5/6] Pitch Detector      [OK] (with ML deps)
[6/6] Configuration       [OK]
```

### Check GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Step 6: Run Application

### Option A: Use Start Script

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
```

### Option B: Manual Start

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Visit Application

Open browser: **http://localhost:3000**

---

## Troubleshooting

### "No module named 'torch'"

**Solution:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory"

**Solution:**
- Edit `backend/config/settings.yaml`
- Change `use_fp16: true`
- Reduce `output_duration: 30`

### "OPENAI_API_KEY not found"

**Solution:**
```bash
# Create .env file in backend directory
echo OPENAI_API_KEY=sk-your-key > backend/.env
```

### "Port 8000 already in use"

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Frontend won't compile

**Solution:**
```bash
cd frontend
rm -rf node_modules .next
npm install
npm run dev
```

---

## Optional: Install Better Vocals

For higher quality vocals, use ElevenLabs:

```bash
pip install elevenlabs
```

Update `.env`:
```bash
ELEVENLABS_API_KEY=your-key-here
```

---

## Uninstall

```bash
# Remove virtual environment
cd backend
rm -rf venv

# Remove node_modules
cd ../frontend
rm -rf node_modules

# Remove .env files
rm backend/.env
```

---

## Next Steps

1. Read the [README.md](README.md)
2. Run `python test_modules.py` to verify setup
3. Run `python demo_generator.py` to create a demo track
4. Visit http://localhost:3000 to start creating music!

---

**Need Help?**
- Check the main README
- Review test output
- Enable debug logging in `config/settings.yaml`
