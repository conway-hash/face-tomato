import asyncio
import os
import io
import cv2
import numpy as np
import discord
from discord.ext import commands
from discord import app_commands
from PIL import Image
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

TOMATO_GIF = "tomato.gif"
model = YOLO("yolov8n-pose.pt")

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


def detect_face(image_bgr):
    results = model(image_bgr, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints.xy[0].cpu().numpy()
    x, y = int(kps[0][0]), int(kps[0][1])  # nose = keypoint 0
    if x == 0 and y == 0:
        return None
    return x, y


def load_tomato_frames():
    gif = Image.open(TOMATO_GIF)
    duration = gif.info.get("duration", 100)
    frames = []
    try:
        while True:
            frames.append(gif.convert("RGBA"))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    # Keep every other frame to halve processing time, double duration to compensate
    return frames[::2], duration * 2


def build_gif(photo_bgr, point):
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(photo_rgb).convert("RGB")
    h, w = photo_bgr.shape[:2]

    tomato_frames, duration = load_tomato_frames()
    tw, th = int(w * 1.5), int(h * 1.5)

    # Pre-resize all tomato frames once
    resized_frames = [f.resize((tw, th), Image.BILINEAR) for f in tomato_frames]
    x = point[0] - tw // 2
    y = point[1] - th // 2

    out_frames = []
    for frame in resized_frames:
        canvas = base.copy().convert("RGBA")
        canvas.paste(frame, (x, y), frame)
        out_frames.append(canvas.convert("RGB"))

    buf = io.BytesIO()
    out_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=out_frames[1:],
        loop=0,
        duration=duration,
        optimize=False,
    )
    buf.seek(0)
    return buf


@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user} — slash commands synced")


@bot.tree.command(name="tomato", description="Throw a tomato at someone's face")
@app_commands.describe(image="The image to tomato")
async def tomato(interaction: discord.Interaction, image: discord.Attachment):
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.response.send_message("Please attach an image.", ephemeral=True)
        return

    await interaction.response.defer()

    img_bytes = await image.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    point = detect_face(image_bgr)
    if point is None:
        await interaction.followup.send("Couldn't detect a face in that image.")
        return

    loop = asyncio.get_event_loop()
    gif_buf = await loop.run_in_executor(None, build_gif, image_bgr, point)
    await interaction.followup.send(file=discord.File(gif_buf, filename="tomato.gif"))


bot.run(os.getenv("DISCORD_TOKEN"))
