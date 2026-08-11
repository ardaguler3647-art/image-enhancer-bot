"""
Image Enhancer Bot
-------------------
A Telegram bot that enhances, sharpens, denoises, and upscales images
using lightweight, CPU-friendly classical image-processing techniques
(Pillow + OpenCV), so it runs comfortably on a free-tier container
with no GPU. See README.md for what's "full" vs "lightweight".
"""

import io
import os
import threading
import logging
import urllib.request
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from flask import Flask

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("image-enhancer-bot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT", 8080))

# Max image size we'll process, to keep memory use safe on free-tier RAM.
MAX_DIMENSION = 2500
MAX_FILE_MB = 15

# ---------------------------------------------------------------------------
# KEEP-ALIVE / HEALTHCHECK WEB SERVER
# Some hosts (including Railway, if deployed as a web-exposed service) expect
# a bound HTTP port to consider the deploy healthy. This tiny Flask app is
# not used for anything else; the bot itself talks to Telegram via polling.
# ---------------------------------------------------------------------------
keep_alive_app = Flask(__name__)


@keep_alive_app.route("/")
def health_check():
    return "Image Enhancer Bot is alive.", 200


def run_keep_alive_server():
    keep_alive_app.run(host="0.0.0.0", port=PORT)


# ---------------------------------------------------------------------------
# SUPER-RESOLUTION MODEL (lightweight FSRCNN, CPU-friendly)
# Downloaded once at first use and cached to /tmp. If the download or the
# model fails for any reason (no internet at build time, model unavailable,
# etc.), we transparently fall back to a high-quality Lanczos resize instead
# of crashing.
# ---------------------------------------------------------------------------
FSRCNN_MODEL_PATH = "/tmp/FSRCNN_x2.pb"
FSRCNN_MODEL_URL = (
    "https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x2.pb"
)
_sr_model_lock = threading.Lock()
_sr_model = None
_sr_model_failed = False


def get_super_res_model():
    """Lazily download and load the FSRCNN super-resolution model.
    Returns None if unavailable, so callers can fall back gracefully."""
    global _sr_model, _sr_model_failed
    if _sr_model is not None:
        return _sr_model
    if _sr_model_failed:
        return None

    with _sr_model_lock:
        if _sr_model is not None:
            return _sr_model
        if _sr_model_failed:
            return None
        try:
            if not os.path.exists(FSRCNN_MODEL_PATH):
                logger.info("Downloading FSRCNN super-resolution model...")
                urllib.request.urlretrieve(FSRCNN_MODEL_URL, FSRCNN_MODEL_PATH)
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(FSRCNN_MODEL_PATH)
            sr.setModel("fsrcnn", 2)
            _sr_model = sr
            logger.info("FSRCNN model loaded successfully.")
            return _sr_model
        except Exception as e:
            logger.warning(f"Super-resolution model unavailable, will use fallback resize: {e}")
            _sr_model_failed = True
            return None


# ---------------------------------------------------------------------------
# IMAGE CONVERSION HELPERS
# ---------------------------------------------------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def clamp_size(img: Image.Image) -> Image.Image:
    """Downscale very large images before processing to protect memory."""
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # respect camera rotation metadata
    return clamp_size(img.convert("RGB"))


def image_to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 90) -> io.BytesIO:
    bio = io.BytesIO()
    fmt = fmt.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    save_kwargs = {}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    img.convert("RGB").save(bio, format=fmt, **save_kwargs)
    bio.seek(0)
    bio.name = f"result.{fmt.lower()}"
    return bio


# ---------------------------------------------------------------------------
# ENHANCEMENT FUNCTIONS
# Each takes a PIL Image and returns a new PIL Image. None of these mutate
# the input image, so the original is always preserved in memory for
# before/after comparisons.
# ---------------------------------------------------------------------------

def enhance_sharpen(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def enhance_denoise(img: Image.Image) -> Image.Image:
    cv_img = pil_to_cv2(img)
    denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 7, 7, 7, 21)
    return cv2_to_pil(denoised)


def enhance_brightness(img: Image.Image) -> Image.Image:
    # Auto-correct exposure based on the image's mean luminance.
    gray = np.array(img.convert("L"), dtype=np.float32)
    mean_luminance = gray.mean()
    target = 128.0
    factor = max(0.5, min(1.8, target / max(mean_luminance, 1.0)))
    return ImageEnhance.Brightness(img).enhance(factor)


def enhance_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(1.25)


def enhance_color(img: Image.Image) -> Image.Image:
    return ImageEnhance.Color(img).enhance(1.3)


def enhance_auto(img: Image.Image) -> Image.Image:
    result = enhance_brightness(img)
    result = enhance_contrast(result)
    result = enhance_color(result)
    result = enhance_sharpen(result)
    return result


def upscale_image(img: Image.Image) -> Image.Image:
    """2x upscale. Uses FSRCNN super-resolution if available, otherwise
    falls back to a high-quality Lanczos resize — never fails outright."""
    model = get_super_res_model()
    if model is not None:
        try:
            cv_img = pil_to_cv2(img)
            upscaled = model.upsample(cv_img)
            return cv2_to_pil(upscaled)
        except Exception as e:
            logger.warning(f"Super-resolution upsample failed, falling back: {e}")
    w, h = img.size
    return img.resize((w * 2, h * 2), Image.LANCZOS)


def enhance_face(img: Image.Image) -> Image.Image:
    """Detects faces with OpenCV's built-in Haar cascade (no extra model
    download needed) and applies targeted sharpening + mild denoise to
    those regions only, leaving the rest of the image untouched."""
    cv_img = pil_to_cv2(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        # No faces detected — fall back to a general auto-enhance instead
        # of doing nothing, so the user still gets a visible result.
        return enhance_auto(img)

    result = cv_img.copy()
    for (x, y, w, h) in faces:
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(cv_img.shape[1], x + w + pad_x), min(cv_img.shape[0], y + h + pad_y)
        region = result[y0:y1, x0:x1]
        region = cv2.fastNlMeansDenoisingColored(region, None, 5, 5, 7, 21)
        region = cv2.detailEnhance(region, sigma_s=10, sigma_r=0.15)
        result[y0:y1, x0:x1] = region

    return cv2_to_pil(result)


def restore_old_photo(img: Image.Image) -> Image.Image:
    """Lightweight restoration: median-blur to soften scratches/grain,
    contrast stretching to revive faded tones, and a mild sharpen pass
    to recover detail lost in the denoise step. This is a classical
    approach, not deep-learning inpainting — it works best on mild
    scratches/fading, not heavy damage or torn photos."""
    cv_img = pil_to_cv2(img)
    denoised = cv2.medianBlur(cv_img, 3)
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)
    pil_img = cv2_to_pil(denoised)
    pil_img = ImageOps.autocontrast(pil_img, cutoff=1)
    pil_img = ImageEnhance.Color(pil_img).enhance(1.15)
    pil_img = enhance_sharpen(pil_img)
    return pil_img


def background_cleanup(img: Image.Image) -> Image.Image:
    """Applies an edge-preserving bilateral filter to reduce background
    noise/grain while keeping subject edges crisp."""
    cv_img = pil_to_cv2(img)
    cleaned = cv2.bilateralFilter(cv_img, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2_to_pil(cleaned)


def make_before_after(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    """Builds a single side-by-side comparison image."""
    h = 500
    def resize_h(im):
        w = int(im.width * (h / im.height))
        return im.resize((w, h), Image.LANCZOS)

    left = resize_h(original)
    right = resize_h(enhanced)
    gap = 8
    combined = Image.new("RGB", (left.width + right.width + gap, h), (30, 30, 30))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width + gap, 0))
    return combined


# ---------------------------------------------------------------------------
# TELEGRAM HANDLERS
# ---------------------------------------------------------------------------

MENU_BUTTONS = [
    [
        InlineKeyboardButton("✨ Auto Enhance", callback_data="auto"),
        InlineKeyboardButton("🔍 HD Upscale 2x", callback_data="upscale"),
    ],
    [
        InlineKeyboardButton("🔪 Sharpen", callback_data="sharpen"),
        InlineKeyboardButton("🧹 Denoise", callback_data="denoise"),
    ],
    [
        InlineKeyboardButton("☀️ Brightness", callback_data="brightness"),
        InlineKeyboardButton("🌓 Contrast", callback_data="contrast"),
    ],
    [
        InlineKeyboardButton("🎨 Color Boost", callback_data="color"),
        InlineKeyboardButton("🙂 Face Enhance", callback_data="face"),
    ],
    [
        InlineKeyboardButton("🕰️ Restore Old Photo", callback_data="restore"),
        InlineKeyboardButton("🖼️ Background Cleanup", callback_data="bgclean"),
    ],
    [
        InlineKeyboardButton("↔️ Before / After", callback_data="beforeafter"),
        InlineKeyboardButton("ℹ️ Image Info", callback_data="info"),
    ],
]

ENHANCERS = {
    "auto": ("Auto Enhance", enhance_auto),
    "upscale": ("HD Upscale 2x", upscale_image),
    "sharpen": ("Sharpen", enhance_sharpen),
    "denoise": ("Denoise", enhance_denoise),
    "brightness": ("Brightness Correction", enhance_brightness),
    "contrast": ("Contrast Enhancement", enhance_contrast),
    "color": ("Color Enhancement", enhance_color),
    "face": ("Face Enhance", enhance_face),
    "restore": ("Old Photo Restoration", restore_old_photo),
    "bgclean": ("Background Cleanup", background_cleanup),
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🖼️ *Image Enhancer Bot*\n\n"
        "Send me a photo and I'll show you enhancement options.\n\n"
        "*Other commands:*\n"
        "• `/convert <jpg|png|webp>` — convert your last image's format\n"
        "• `/compress <quality 1-95>` — recompress your last image\n"
        "• `/batch` — enhance all photos from your last album/media group\n\n"
        "_Note: HD Upscale and restoration use lightweight, CPU-based "
        "techniques so this runs on a free hosting tier — not a full "
        "deep-learning model. Results are good for everyday photos, not "
        "professional restoration._"
    )
    await update.message.reply_markdown(text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    if photo.file_size and photo.file_size > MAX_FILE_MB * 1024 * 1024:
        await update.message.reply_text(f"That image is over {MAX_FILE_MB}MB — please send a smaller one.")
        return

    file = await photo.get_file()
    file_bytes = bytes(await file.download_as_bytearray())

    context.user_data["last_image_bytes"] = file_bytes

    # Track media-group (album) uploads for /batch.
    media_group_id = update.message.media_group_id
    if media_group_id:
        batch = context.user_data.setdefault("batch_groups", {})
        batch.setdefault(media_group_id, []).append(file_bytes)
        context.user_data["last_media_group_id"] = media_group_id

    await update.message.reply_text(
        "Choose an enhancement:",
        reply_markup=InlineKeyboardMarkup(MENU_BUTTONS),
    )


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    action = query.data
    file_bytes = context.user_data.get("last_image_bytes")
    if not file_bytes:
        await query.message.reply_text("I don't have an image to work with — please send one first.")
        return

    try:
        original = load_image_from_bytes(file_bytes)
    except Exception:
        await query.message.reply_text("Sorry, I couldn't read that image. Please try sending it again.")
        return

    if action == "info":
        await send_image_info(query.message, file_bytes, original)
        return

    if action not in ENHANCERS and action != "beforeafter":
        return

    await query.message.chat.send_action("upload_photo")

    try:
        if action == "beforeafter":
            # Use auto-enhance as the "after" for the comparison.
            enhanced = enhance_auto(original)
            combo = make_before_after(original, enhanced)
            bio = image_to_bytes(combo, "JPEG", quality=90)
            await query.message.reply_photo(photo=bio, caption="Left: original — Right: auto-enhanced")
            return

        label, func = ENHANCERS[action]
        enhanced = func(original)
        context.user_data["last_result_image"] = enhanced
        bio = image_to_bytes(enhanced, "JPEG", quality=92)
        await query.message.reply_photo(photo=bio, caption=f"✅ {label} complete.")
    except Exception as e:
        logger.exception("Enhancement failed")
        await query.message.reply_text(
            f"Something went wrong while applying that enhancement ({type(e).__name__}). "
            "Please try a different image or option."
        )


async def send_image_info(message, file_bytes: bytes, img: Image.Image):
    size_kb = len(file_bytes) / 1024
    fmt = img.format or "Unknown (re-encoded)"
    w, h = img.size
    text = (
        "ℹ️ *Image Info*\n"
        f"• Dimensions: `{w} x {h}`\n"
        f"• Format: `{fmt}`\n"
        f"• File size: `{size_kb:.1f} KB`\n"
        f"• Mode: `{img.mode}`"
    )
    await message.reply_markdown(text)


async def cmd_convert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or context.args[0].lower() not in ("jpg", "jpeg", "png", "webp"):
        await update.message.reply_text("Usage: /convert <jpg|png|webp>")
        return
    file_bytes = context.user_data.get("last_image_bytes")
    if not file_bytes:
        await update.message.reply_text("Send me an image first, then use /convert.")
        return
    fmt = context.args[0].lower()
    try:
        img = load_image_from_bytes(file_bytes)
        bio = image_to_bytes(img, fmt, quality=92)
        await update.message.reply_document(document=bio, filename=f"converted.{fmt}")
    except Exception as e:
        logger.exception("Convert failed")
        await update.message.reply_text(f"Conversion failed: {e}")


async def cmd_compress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: /compress <quality 1-95>")
        return
    quality = int(context.args[0])
    if not (1 <= quality <= 95):
        await update.message.reply_text("Please choose a quality between 1 and 95.")
        return
    file_bytes = context.user_data.get("last_image_bytes")
    if not file_bytes:
        await update.message.reply_text("Send me an image first, then use /compress.")
        return
    try:
        img = load_image_from_bytes(file_bytes)
        bio = image_to_bytes(img, "JPEG", quality=quality)
        before_kb = len(file_bytes) / 1024
        after_kb = bio.getbuffer().nbytes / 1024
        await update.message.reply_document(
            document=bio,
            filename="compressed.jpg",
            caption=f"Compressed: {before_kb:.0f} KB → {after_kb:.0f} KB (quality {quality})",
        )
    except Exception as e:
        logger.exception("Compress failed")
        await update.message.reply_text(f"Compression failed: {e}")


async def cmd_batch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    group_id = context.user_data.get("last_media_group_id")
    batch_groups = context.user_data.get("batch_groups", {})
    images = batch_groups.get(group_id) if group_id else None

    if not images or len(images) < 2:
        await update.message.reply_text(
            "Send multiple photos together as an album, then run /batch to "
            "auto-enhance all of them at once."
        )
        return

    await update.message.reply_text(f"Auto-enhancing {len(images)} images...")
    for i, file_bytes in enumerate(images, start=1):
        try:
            img = load_image_from_bytes(file_bytes)
            enhanced = enhance_auto(img)
            bio = image_to_bytes(enhanced, "JPEG", quality=92)
            await update.message.reply_photo(photo=bio, caption=f"Image {i}/{len(images)}")
        except Exception as e:
            logger.exception("Batch item failed")
            await update.message.reply_text(f"Image {i} failed: {e}")

    # Clear the processed batch so /batch doesn't re-send old images.
    batch_groups.pop(group_id, None)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Something went wrong on my end. Please try again."
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")

    threading.Thread(target=run_keep_alive_server, daemon=True).start()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("convert", cmd_convert))
    app.add_handler(CommandHandler("compress", cmd_compress))
    app.add_handler(CommandHandler("batch", cmd_batch))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_error_handler(error_handler)

    logger.info("Image Enhancer Bot running via polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
