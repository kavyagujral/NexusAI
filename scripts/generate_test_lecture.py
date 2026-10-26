"""
Script: generate_test_lecture.py
Generates a real 5-minute (300 seconds) educational video with:
1. Synthesized audio narration covering microservice architecture, API gateways, database clusters, and code snippets.
2. Synchronized graphical slides, architecture diagrams, code listings, and whiteboards.
3. Multiplexed into an MP4 container.
"""

import os
import subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

OUTPUT_DIR = Path("./data/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUT = OUTPUT_DIR / "educational_lecture_5min.mp4"
TEMP_AUDIO = Path("./data/temp_lecture_audio.wav")
TEMP_VIDEO = Path("./data/temp_lecture_visual.mp4")

FPS = 25
DURATION_SEC = 300  # Exactly 5 minutes (300 seconds)
TOTAL_FRAMES = FPS * DURATION_SEC
WIDTH, HEIGHT = 1280, 720

# Lecture Script Timings and Sections
SECTIONS = [
    {
        "start": 0,
        "end": 60,
        "type": "slide",
        "title": "Module 1: Introduction to Distributed Cloud Systems",
        "subtitle": "Computer Science 401 - Advanced System Architecture",
        "speech": "Welcome to Computer Science 401. Today we begin our in-depth study of distributed cloud computing and scalable system architectures. Over the next five minutes, we will examine core patterns including microservices, API gateways, service discovery, and database consistency models."
    },
    {
        "start": 60,
        "end": 150,
        "type": "architecture diagram",
        "title": "Core Architecture: Microservices & API Gateway",
        "subtitle": "Client -> API Gateway -> Auth / Inventory / Billing Services -> PostgreSQL Cluster",
        "speech": "Let us examine this microservice architecture diagram. The client sends all incoming requests to the unified API gateway. The gateway validates authentication tokens and routes requests to the appropriate downstream microservices such as user service, order service, and billing service. Notice how each service maintains its own decoupled datastore."
    },
    {
        "start": 150,
        "end": 210,
        "type": "code snippet",
        "title": "Implementation: Async Gateway Request Routing",
        "subtitle": "Python FastAPI + Redis Distributed Cache Route Handler",
        "speech": "Here is our implementation code snippet. Notice how the asynchronous route handler checks the Redis distributed cache before querying the primary database cluster. This caching layer reduces overall latency by sixty percent under high concurrent load."
    },
    {
        "start": 210,
        "end": 260,
        "type": "whiteboard",
        "title": "Whiteboard Derivation: CAP Theorem & Consistency",
        "subtitle": "Consistency vs Availability in Network Partition Scenarios",
        "speech": "On the whiteboard, let us analyze the trade-offs of the CAP theorem. During network partitions, our distributed database must choose between strict consistency and high availability. Here we use an eventual consistency model with vector clocks to resolve distributed merge conflicts."
    },
    {
        "start": 260,
        "end": 300,
        "type": "slide",
        "title": "Summary & Key Takeaways",
        "subtitle": "Decoupled Services, Resilient Gateways, & Distributed Consensus",
        "speech": "To summarize our lecture: decoupled microservices improve development velocity and fault tolerance. In next week's session, we will implement Raft distributed consensus. Thank you for attending."
    }
]


def render_frame(sec: float) -> np.ndarray:
    """Renders a 1280x720 educational slide / diagram frame."""
    # Determine active section
    active_sec = SECTIONS[-1]
    for s in SECTIONS:
        if s["start"] <= sec < s["end"]:
            active_sec = s
            break

    # Base background color
    if active_sec["type"] == "architecture diagram":
        bg_color = (15, 23, 42)      # Slate 900
    elif active_sec["type"] == "code snippet":
        bg_color = (10, 15, 26)      # Dark IDE black
    elif active_sec["type"] == "whiteboard":
        bg_color = (245, 247, 250)   # Clean whiteboard white
    else:
        bg_color = (18, 24, 38)      # Deep blue/gray

    img = Image.new("RGB", (WIDTH, HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(img)

    text_color = (15, 23, 42) if active_sec["type"] == "whiteboard" else (248, 250, 252)
    accent_color = (37, 99, 235) if active_sec["type"] == "whiteboard" else (59, 130, 246)
    sub_color = (100, 116, 139) if active_sec["type"] == "whiteboard" else (148, 163, 184)

    # Top Header Banner
    draw.rectangle([(0, 0), (WIDTH, 70)], fill=(30, 41, 59) if active_sec["type"] != "whiteboard" else (226, 232, 240))
    draw.text((40, 20), "NEXUS EDUCATIONAL LECTURE SERIES", fill=accent_color)
    draw.text((WIDTH - 200, 20), f"TIMESTAMP: {int(sec//60):02d}:{int(sec%60):02d} / 05:00", fill=text_color)

    # Category Tag
    draw.rectangle([(40, 95), (260, 125)], fill=accent_color)
    draw.text((50, 102), f"[{active_sec['type'].upper()}]", fill=(255, 255, 255))

    # Main Title & Subtitle
    draw.text((40, 140), active_sec["title"], fill=text_color)
    draw.text((40, 180), active_sec["subtitle"], fill=sub_color)

    # Content Area Rendering
    if active_sec["type"] == "architecture diagram":
        # Draw Architectural Box Diagram
        boxes = [
            ("Client Web / Mobile", (100, 260), (320, 360), (59, 130, 246)),
            ("API Gateway (Port 8000)", (420, 260), (680, 360), (16, 185, 129)),
            ("Auth Service", (780, 220), (1020, 300), (139, 92, 246)),
            ("Order Service", (780, 330), (1020, 410), (236, 72, 153)),
            ("Database Cluster (PostgreSQL)", (1080, 260), (1240, 380), (245, 158, 11))
        ]
        for name, (x1, y1), (x2, y2), col in boxes:
            draw.rectangle([(x1, y1), (x2, y2)], outline=col, width=3, fill=(30, 41, 59))
            draw.text((x1 + 15, y1 + 35), name, fill=(255, 255, 255))

        # Connector Arrows
        draw.line([(320, 310), (420, 310)], fill=(255, 255, 255), width=3)
        draw.line([(680, 260), (780, 260)], fill=(255, 255, 255), width=2)
        draw.line([(680, 360), (780, 360)], fill=(255, 255, 255), width=2)
        draw.line([(1020, 310), (1080, 310)], fill=(255, 255, 255), width=2)

    elif active_sec["type"] == "code snippet":
        # Draw Code Window
        draw.rectangle([(60, 230), (1220, 650)], fill=(15, 23, 42), outline=(51, 65, 85), width=2)
        code_lines = [
            "1 | @router.get('/api/services/{service_id}/route')",
            "2 | async def route_request(service_id: str, payload: RequestPayload):",
            "3 |     # Step 1: Check Redis distributed cache for pre-warmed route tokens",
            "4 |     cached_route = await redis_client.get(f'route:{service_id}')",
            "5 |     if cached_route:",
            "6 |         return json.loads(cached_route)",
            "7 |     # Step 2: Query PostgreSQL service discovery registry",
            "8 |     route_record = await db.fetch_one(query=select(ServiceNode).where(...))",
            "9 |     await redis_client.setex(f'route:{service_id}', 3600, json.dumps(route_record))",
            "10|     return route_record"
        ]
        for idx, line in enumerate(code_lines):
            draw.text((80, 260 + idx * 35), line, fill=(226, 232, 240) if not line.strip().startswith("#") else (100, 116, 139))

    elif active_sec["type"] == "whiteboard":
        # Draw Handwritten Whiteboard Sketch
        draw.text((80, 250), "CAP THEOREM TRADEOFF MATRIX:", fill=(30, 41, 59))
        draw.rectangle([(80, 300), (1200, 600)], outline=(148, 163, 184), width=2)
        draw.line([(450, 300), (450, 600)], fill=(148, 163, 184), width=2)
        draw.line([(820, 300), (820, 600)], fill=(148, 163, 184), width=2)
        draw.line([(80, 360), (1200, 360)], fill=(148, 163, 184), width=2)
        
        draw.text((100, 320), "CONSISTENCY (C)", fill=(37, 99, 235))
        draw.text((470, 320), "AVAILABILITY (A)", fill=(16, 185, 129))
        draw.text((840, 320), "PARTITION TOLERANCE (P)", fill=(236, 72, 153))

        draw.text((100, 400), "Every read receives the\nmost recent write or error.", fill=(51, 65, 85))
        draw.text((470, 400), "Every non-failing node\nreturns non-error response.", fill=(51, 65, 85))
        draw.text((840, 400), "System operates despite\narbitrary dropped messages.", fill=(51, 65, 85))
    else:
        # Standard Presentation Slide Bullet points
        bullets = [
            "• Distributed microservice architecture allows independent scaling per service domain.",
            "• API Gateways provide centralized rate-limiting, authentication, and telemetry.",
            "• In-memory caching layers (Redis) insulate core database clusters from peak read loads.",
            "• Event-driven messaging brokers enable asynchronous decoupled transactions."
        ]
        for idx, bullet in enumerate(bullets):
            draw.text((60, 260 + idx * 75), bullet, fill=text_color)

    # Bottom Progress Bar
    progress_w = int((sec / DURATION_SEC) * WIDTH)
    draw.rectangle([(0, HEIGHT - 10), (progress_w, HEIGHT)], fill=accent_color)

    return np.array(img)


def main():
    print("Generating 5-minute educational audio track via speech synthesis...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # 1. Synthesize audio files for each section
    aiff_files = []
    for idx, s in enumerate(SECTIONS):
        sec_aiff = Path(f"./data/sec_{idx}.aiff")
        subprocess.run(["say", "-v", "Samantha", "-o", str(sec_aiff), s["speech"]], check=True)
        aiff_files.append((sec_aiff, s["start"], s["end"]))

    # 2. Mix into continuous 300s 16kHz WAV
    # Create silence audio buffer of 300 seconds
    sample_rate = 16000
    full_audio = np.zeros(int(sample_rate * DURATION_SEC), dtype=np.float32)

    import soundfile as sf
    for aiff_path, start_sec, end_sec in aiff_files:
        wav_temp = aiff_path.with_suffix(".wav")
        # Convert aiff to 16kHz mono wav
        subprocess.run([
            ffmpeg_exe, "-y", "-i", str(aiff_path),
            "-ar", "16000", "-ac", "1", str(wav_temp)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        data, sr = sf.read(str(wav_temp))
        start_idx = int(start_sec * sample_rate)
        end_idx = min(start_idx + len(data), len(full_audio))
        full_audio[start_idx:end_idx] = data[:end_idx - start_idx]
        
        aiff_path.unlink(missing_ok=True)
        wav_temp.unlink(missing_ok=True)

    sf.write(str(TEMP_AUDIO), full_audio, sample_rate)
    print(f"Generated 5-minute synthesized audio at {TEMP_AUDIO}")

    # 3. Render 300-second video stream (1 frame every 0.1s for fast video generation: 10 FPS * 300 = 3000 frames)
    VIDEO_FPS = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(TEMP_VIDEO), fourcc, VIDEO_FPS, (WIDTH, HEIGHT))

    print(f"Rendering {DURATION_SEC * VIDEO_FPS} visual frames across 5 minutes...")
    for f in range(DURATION_SEC * VIDEO_FPS):
        current_sec = f / VIDEO_FPS
        frame_rgb = render_frame(current_sec)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        out_video.write(frame_bgr)
        if f % (VIDEO_FPS * 30) == 0:
            print(f"Rendered {current_sec:.0f}s / {DURATION_SEC}s ({current_sec/DURATION_SEC*100:.1f}%)")

    out_video.release()
    print("Visual rendering complete.")

    # 4. Multiplex audio and video into final educational lecture MP4
    print("Multiplexing audio and video into final educational lecture video...")
    subprocess.run([
        ffmpeg_exe, "-y",
        "-i", str(TEMP_VIDEO),
        "-i", str(TEMP_AUDIO),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(VIDEO_OUT)
    ], check=True)

    TEMP_AUDIO.unlink(missing_ok=True)
    TEMP_VIDEO.unlink(missing_ok=True)
    print(f"SUCCESS: Created 5-minute educational video at {VIDEO_OUT} ({VIDEO_OUT.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
