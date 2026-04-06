import math
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from starlette.background import BackgroundTask

APP_NAME = "podcast2short"
APP_DIR = Path("/app")
INDEX_FILE = APP_DIR / "index.html"

MAX_UPLOAD_SIZE_MB = 25
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = 1280

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

FPS = 24
AUDIO_SAMPLE_RATE = 12000

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado.")
    return INDEX_FILE.read_text(encoding="utf-8")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": APP_NAME})


def safe_hex_color(color: str, default: str = "#22c55e") -> str:
    if not color:
        return default
    color = color.strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", color):
        return color if color.startswith("#") else f"#{color}"
    return default


async def save_upload_with_limit(
    upload: UploadFile,
    destination: Path,
    max_bytes: int = MAX_UPLOAD_SIZE_BYTES,
) -> None:
    total = 0
    with destination.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"El archivo {upload.filename or 'subido'} supera el límite de {MAX_UPLOAD_SIZE_MB} MB.",
                )
            f.write(chunk)
    await upload.close()


def cover_resize(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_h = target_h
        new_w = int(new_h * src_ratio)
    else:
        new_w = target_w
        new_h = int(new_w / src_ratio)

    resized = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def square_crop_resize(image: Image.Image, side: int) -> Image.Image:
    src_w, src_h = image.size
    crop_side = min(src_w, src_h)
    left = (src_w - crop_side) // 2
    top = (src_h - crop_side) // 2
    cropped = image.crop((left, top, left + crop_side, top + crop_side))
    return cropped.resize((side, side), Image.LANCZOS)


def rounded_mask(width: int, height: int, radius: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    return mask


def add_center_card_border(canvas: Image.Image, x: int, y: int, side: int, radius: int) -> None:
    draw = ImageDraw.Draw(canvas)
    border_w = max(3, side // 140)
    draw.rounded_rectangle(
        (x, y, x + side, y + side),
        radius=radius,
        outline=(255, 255, 255, 242),
        width=border_w,
    )


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    lines: List[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_bottom_title_box(base: Image.Image, title: str) -> None:
    title = (title or "").strip()
    if not title:
        return

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    box_left = int(base.width * 0.07)
    box_right = int(base.width * 0.93)
    box_top = int(base.height * 0.65)
    box_bottom = int(base.height * 0.965)
    radius = 28

    draw_overlay.rounded_rectangle(
        (box_left, box_top, box_right, box_bottom),
        radius=radius,
        fill=(0, 0, 0, 105),
    )

    base.alpha_composite(overlay)

    draw = ImageDraw.Draw(base)
    max_width = int(base.width * 0.74)
    max_height = int(base.height * 0.09)

    chosen_font = None
    chosen_lines: List[str] = []

    for font_size in range(42, 18, -2):
        try:
            font = ImageFont.truetype(FONT_BOLD, font_size)
        except Exception:
            font = ImageFont.load_default()

        lines = wrap_text(draw, title, font, max_width)
        block = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), block, font=font, spacing=6, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= max_width and text_h <= max_height:
            chosen_font = font
            chosen_lines = lines
            break

    if chosen_font is None:
        try:
            chosen_font = ImageFont.truetype(FONT_BOLD, 22)
        except Exception:
            chosen_font = ImageFont.load_default()
        chosen_lines = wrap_text(draw, title, chosen_font, max_width)

    block = "\n".join(chosen_lines)
    bbox = draw.multiline_textbbox((0, 0), block, font=chosen_font, spacing=6, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (base.width - text_w) // 2
    y = box_top + ((box_bottom - box_top - text_h) // 2) - 1

    for dx, dy in [(0, 2), (1, 1), (-1, 1), (0, 0)]:
        draw.multiline_text(
            (x + dx, y + dy),
            block,
            font=chosen_font,
            fill=(0, 0, 0, 200),
            spacing=6,
            align="center",
        )

    draw.multiline_text(
        (x, y),
        block,
        font=chosen_font,
        fill=(255, 255, 255, 245),
        spacing=6,
        align="center",
    )


def extract_audio_samples(audio_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-hide_banner",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError("No se pudo extraer el audio para generar la onda.")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        raise RuntimeError("El audio está vacío o no se pudo procesar.")

    samples /= 32768.0
    return samples


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("No se pudo obtener la duración del audio.")
    try:
        duration = float(result.stdout.strip())
    except Exception as e:
        raise RuntimeError("No se pudo interpretar la duración del audio.") from e
    if duration <= 0:
        raise RuntimeError("La duración del audio no es válida.")
    return duration


def build_wave_envelope_frame(
    samples: np.ndarray,
    bar_count: int,
    center_sample: int,
    samples_per_bar: int,
) -> np.ndarray:
    envelope = np.zeros(bar_count, dtype=np.float32)
    half = bar_count // 2

    for i in range(bar_count):
        offset = (i - half) * samples_per_bar
        start = center_sample + offset
        end = start + samples_per_bar

        if end <= 0 or start >= len(samples):
            continue

        start_clamped = max(0, start)
        end_clamped = min(len(samples), end)
        chunk = samples[start_clamped:end_clamped]
        if chunk.size == 0:
            continue

        abs_chunk = np.abs(chunk)
        peak = float(abs_chunk.max())
        avg = float(abs_chunk.mean())
        envelope[i] = peak * 0.45 + avg * 0.55

    for _ in range(2):
        padded = np.pad(envelope, (1, 1), mode="edge")
        envelope = (
            padded[:-2] * 0.25 +
            padded[1:-1] * 0.50 +
            padded[2:] * 0.25
        )

    envelope = np.clip(envelope * 1.05 + 0.02, 0.0, 1.0)
    envelope = np.power(envelope, 1.05)
    return envelope


def render_wave_frame(
    envelope: np.ndarray,
    width: int,
    height: int,
    wave_color: str,
    bar_width: int,
    bar_gap: int,
) -> bytes:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    draw = ImageDraw.Draw(overlay)
    draw_glow = ImageDraw.Draw(glow)

    slot_width = bar_width + bar_gap
    bar_count = len(envelope)
    used_width = bar_count * bar_width + (bar_count - 1) * bar_gap
    start_x = max(0, (width - used_width) // 2)

    center_y = height / 2
    usable_height = height * 0.72
    rgb = tuple(int(wave_color[i:i + 2], 16) for i in (1, 3, 5))

    for i in range(bar_count):
        h = max(8, int(envelope[i] * usable_height))
        x = start_x + i * slot_width
        top = int(center_y - h / 2)
        bottom = int(center_y + h / 2)

        draw_glow.rounded_rectangle(
            (x - 1, top - 1, x + bar_width + 1, bottom + 1),
            radius=4,
            fill=(*rgb, 70),
        )

        draw.rounded_rectangle(
            (x, top, x + bar_width, bottom),
            radius=4,
            fill=(*rgb, 230),
        )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=4))
    merged = Image.alpha_composite(glow, overlay)
    return merged.tobytes()


def compose_base_image(
    background_path: Path,
    center_image_path: Path,
    title: str,
    width: int,
    height: int,
    output_path: Path,
) -> None:
    bg = Image.open(background_path).convert("RGBA")
    fg = Image.open(center_image_path).convert("RGBA")

    canvas = cover_resize(bg, width, height).convert("RGBA")

    square_side = int(width * 0.43)
    radius = max(18, square_side // 10)

    square = square_crop_resize(fg, square_side).convert("RGBA")
    mask = rounded_mask(square_side, square_side, radius=radius)
    square.putalpha(mask)

    x = (width - square_side) // 2
    y = int(height * 0.18)

    canvas.alpha_composite(square, (x, y))
    add_center_card_border(canvas, x, y, square_side, radius)
    draw_bottom_title_box(canvas, title)

    canvas.save(output_path, format="PNG")


def should_copy_audio(audio_path: Path) -> bool:
    return audio_path.suffix.lower() in {".m4a", ".aac", ".mp4"}


def render_video_with_animated_waves(
    base_image_path: Path,
    audio_path: Path,
    output_path: Path,
    wave_color: str,
    width: int,
    height: int,
) -> None:
    duration = get_audio_duration(audio_path)
    samples = extract_audio_samples(audio_path, AUDIO_SAMPLE_RATE)

    wave_height = int(height * 0.18)
    overlay_y = int(height * 0.44)

    bar_width = 16
    bar_gap = 10
    slot_width = bar_width + bar_gap
    bar_count = max(1, (width + bar_gap) // slot_width)

    samples_per_bar = 120
    frame_count = max(1, math.ceil(duration * FPS))

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-threads", "1",
        "-loop", "1",
        "-framerate", str(FPS),
        "-i", str(base_image_path),
        "-f", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{wave_height}",
        "-r", str(FPS),
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-filter_complex", f"[0:v][1:v]overlay=0:{overlay_y}:shortest=1,format=yuv420p[v]",
        "-map", "[v]",
        "-map", "2:a:0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "26",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
    ]

    if should_copy_audio(audio_path):
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    cmd += [
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]

    print("RUN render_video_with_animated_waves:", " ".join(cmd), flush=True)

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None

        for frame_idx in range(frame_count):
            t = frame_idx / FPS
            center_sample = int(t * AUDIO_SAMPLE_RATE)
            envelope = build_wave_envelope_frame(
                samples=samples,
                bar_count=bar_count,
                center_sample=center_sample,
                samples_per_bar=samples_per_bar,
            )
            frame_bytes = render_wave_frame(
                envelope=envelope,
                width=width,
                height=wave_height,
                wave_color=wave_color,
                bar_width=bar_width,
                bar_gap=bar_gap,
            )
            process.stdin.write(frame_bytes)

        process.stdin.close()
        process.stdin = None

        returncode = process.wait()
        stdout = process.stdout.read() if process.stdout else b""
        stderr = process.stderr.read() if process.stderr else b""

        print("render_video_with_animated_waves returncode:", returncode, flush=True)
        if stdout:
            print("render_video_with_animated_waves stdout:", stdout.decode("utf-8", errors="ignore"), flush=True)
        if stderr:
            print("render_video_with_animated_waves stderr:", stderr.decode("utf-8", errors="ignore"), flush=True)

        if returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(detail or "ffmpeg falló al generar el vídeo.")

    except Exception:
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except Exception:
            pass
        process.kill()
        process.wait()
        raise


@app.post("/generate")
async def generate_video(
    background_image: UploadFile = File(...),
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    title: str = Form(""),
    overlay_text: str = Form(""),
    wave_color: str = Form("#22c55e"),
    width: int = Form(DEFAULT_WIDTH),
    height: int = Form(DEFAULT_HEIGHT),
):
    wave_color = safe_hex_color(wave_color, "#22c55e")

    tmp_dir = Path(tempfile.mkdtemp(prefix="podcast2short_"))
    print("tmp_dir:", tmp_dir, flush=True)

    try:
        bg_ext = Path(background_image.filename or "background.png").suffix or ".png"
        fg_ext = Path(image.filename or "image.png").suffix or ".png"
        audio_ext = Path(audio.filename or "audio.m4a").suffix or ".m4a"

        bg_path = tmp_dir / f"background{bg_ext}"
        fg_path = tmp_dir / f"foreground{fg_ext}"
        audio_path = tmp_dir / f"audio{audio_ext}"

        base_png_path = tmp_dir / "base.png"
        output_mp4_path = tmp_dir / "short.mp4"

        await save_upload_with_limit(background_image, bg_path)
        await save_upload_with_limit(image, fg_path)
        await save_upload_with_limit(audio, audio_path)

        print("saved files:", bg_path, fg_path, audio_path, flush=True)

        # Forzamos siempre 720x1280
        width = DEFAULT_WIDTH
        height = DEFAULT_HEIGHT

        compose_base_image(
            background_path=bg_path,
            center_image_path=fg_path,
            title=title,
            width=width,
            height=height,
            output_path=base_png_path,
        )
        print("base image created:", base_png_path.exists(), flush=True)

        render_video_with_animated_waves(
            base_image_path=base_png_path,
            audio_path=audio_path,
            output_path=output_mp4_path,
            wave_color=wave_color,
            width=width,
            height=height,
        )
        print("video created:", output_mp4_path.exists(), flush=True)

        if not output_mp4_path.exists():
            raise HTTPException(status_code=500, detail="No se pudo generar el vídeo.")

        return FileResponse(
            path=output_mp4_path,
            media_type="video/mp4",
            filename="short.mp4",
            background=BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True),
        )

    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception as e:
        print("EXCEPTION:", str(e), flush=True)
        print(traceback.format_exc(), flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e) or "Error generando el vídeo.")
