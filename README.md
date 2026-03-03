# 🏊 SwimTimer — Live at swimtimer.app

Automatic 50m phase analysis for swimming coaches. Upload a video, get all splits instantly.

## Deploy to Railway in 5 minutes

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "SwimTimer initial commit"
git remote add origin https://github.com/YOUR_USERNAME/swimtimer.git
git push -u origin main
```

### 2. Deploy on Railway

1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo** → select your repo
3. Railway auto-detects the Dockerfile ✓

### 3. Add your API key

In Railway dashboard → your service → **Variables** tab:

```
ANTHROPIC_API_KEY = sk-ant-your-key-here
```

That's it. Railway gives you a live URL instantly.

---

## How it works

```
User uploads video(s)
       ↓
FastAPI receives files → saves to temp
       ↓
ffmpeg extracts:
  - Audio → RMS analysis → finds "Go" signal
  - Video frames at 30fps for key moments
       ↓
Claude Vision API analyses frames:
  - Water entry
  - End of coulée (surfacing)
  - Wall touch (25m)
  - Push off
  - Finish touch
       ↓
Results streamed back via polling
       ↓
Browser renders comparison table + stacked bar chart
```

## Local development

```bash
# Install deps
pip install -r requirements.txt
brew install ffmpeg   # or: apt install ffmpeg

# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run
uvicorn main:app --reload --port 8000
# Open http://localhost:8000
```

## Limits (Railway free tier)
- 500MB RAM, 1 vCPU
- Max 5 videos per batch
- Videos up to 500MB each
- ~2min analysis per video (Claude API calls)
