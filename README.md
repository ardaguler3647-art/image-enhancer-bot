🖼️ Image Enhancer Bot

A Telegram bot that enhances, sharpens, denoises, upscales, and restores images — built to run on free-tier hosting with no GPU required.

⚠️ Important: what's "full AI" vs. "lightweight" here

Some of the requested features (true HD super-resolution, face restoration, AI old-photo restoration) are normally done with large deep-learning models like Real-ESRGAN or GFPGAN, which need several GB of RAM and ideally a GPU. Those won't run on a free container.

This bot instead uses real, working, CPU-friendly techniques for every feature:

Feature	How it's implemented here
HD Upscaling	FSRCNN (lightweight open-source super-resolution model, ~small file), with automatic fallback to Lanczos resizing if the model can't load
Sharpening	Unsharp mask filtering
Denoising	OpenCV Non-Local Means denoising
Brightness Correction	Auto-adjusted based on measured image luminance
Contrast / Color Enhancement	Classical enhancement curves (Pillow)
Auto Enhancement	Chains brightness → contrast → color → sharpen
Face Enhancement	OpenCV Haar-cascade face detection + targeted sharpen/denoise on the detected region (not a generative face-restoration model)
Old Photo Restoration	Median blur + denoise + contrast stretch + sharpen — good for mild fading/grain, not torn or heavily damaged photos
Background Cleanup	Edge-preserving bilateral filtering
Batch Enhancement	Auto-enhances all photos sent together as an album
Before/After Preview	Side-by-side comparison image
Format Conversion / Compression / Image Info	Direct Pillow operations

This trade-off means results are good for everyday phone photos, but won't match a dedicated AI upscaler/restorer. If you later want the full deep-learning versions, that requires a paid GPU-backed host — happy to help set that up separately if you need it.

🛠️ Built With
python-telegram-bot
Pillow
OpenCV (headless) — denoising, super-resolution, face detection
Flask — minimal healthcheck server
FSRCNN — lightweight open-source super-resolution model, downloaded automatically on first use
🚀 Local Setup
bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=your_token_here   # macOS/Linux
# set TELEGRAM_BOT_TOKEN=your_token_here    # Windows cmd
python bot.py
☁️ Deploying to Railway (via GitHub)
Push this repo to GitHub.
In Railway, click New Project → Deploy from GitHub repo and select it.
Railway auto-detects Python via Nixpacks and reads requirements.txt and the Procfile.
Go to your service's Variables tab and add:
TELEGRAM_BOT_TOKEN = your token from @BotFather
NIXPACKS_PYTHON_VERSION = 3.11 — important: Railway's Nixpacks builder does not reliably read .python-version or runtime.txt files (unlike some other hosts). Setting this environment variable is the documented, reliable way to pin the Python version. This repo also includes a nixpacks.toml as a backup method.
Deploy. Watch the Deploy Logs for Image Enhancer Bot running via polling... to confirm it started.

Note on Railway's free tier: as of now, Railway offers a limited trial credit rather than a permanent free tier, and card requirements have varied — check Railway's current pricing page before deploying, since this can change.

⚠️ Operational Notes
Never commit your bot token — always use environment variables.
Images are capped at 2500px on the long edge and 15MB upload size to protect free-tier RAM; larger images are automatically downscaled before processing.
The FSRCNN model (~a few hundred KB) downloads once on first upscale request and is cached in /tmp for the life of the container. If your host's network blocks the download, upscaling automatically falls back to a standard high-quality resize instead of failing.
All processing happens in memory — no images are stored on disk or logged.
