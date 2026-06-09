import cv2
import json
import time
import os

from pipeline.video_reader import VideoReader
from pipeline.inference_pipeline import InferencePipeline


# ─── CONFIG ──────────────────────────────────────────────────────────────────
video_path = r"C:\Users\abhij\Downloads\Telegram Desktop\video_2026-03-21_10-05-06.mp4"

# Process every Nth frame (0 = every frame, 1 = every 2nd, 2 = every 3rd)
# On a 3050 laptop, skip_frames=1 gives ~2x speed with almost no accuracy loss
# because temporal smoothing in the pipeline covers the skipped frames.
skip_frames = 1

# Optional: resize frame before processing (speeds up YOLO + ViT significantly)
# Set to None to use original resolution, or e.g. (1280, 720)
PROCESS_WIDTH  = 1280
PROCESS_HEIGHT = None   # auto-scale height to preserve aspect ratio

# ─────────────────────────────────────────────────────────────────────────────
out_path = "events.json"
reader   = VideoReader(video_path)
pipeline = InferencePipeline()

# Get source FPS for correct playback timing
cap_fps = reader.cap.get(cv2.CAP_PROP_FPS) or 30.0
delay_ms = max(1, int(1000 / cap_fps / (skip_frames + 1)))

frame_id = 0
t_start  = time.time()

print(f"\n[SEIVR++] Starting processing — skip_frames={skip_frames}")
print(f"[SEIVR++] Source FPS: {cap_fps:.1f}  |  Press Q to stop\n")

try:
    while True:
        ret, frame = reader.read()
        if not ret:
            break

        if frame_id % (skip_frames + 1) == 0:

            # Optional resize for faster inference
            if PROCESS_WIDTH is not None:
                orig_h, orig_w = frame.shape[:2]
                scale = PROCESS_WIDTH / orig_w
                new_h = int(orig_h * scale) if PROCESS_HEIGHT is None else PROCESS_HEIGHT
                frame = cv2.resize(frame, (PROCESS_WIDTH, new_h))

            frame = pipeline.process_frame(frame, frame_id)
            cv2.imshow("SEIVR++ | Surveillance Event Intelligence", frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                print("\n[SEIVR++] Stopped by user.")
                break

        frame_id += 1

except KeyboardInterrupt:
    print("\n[SEIVR++] Stopped by user.")
finally:
    cv2.destroyAllWindows()

# ─── REPORT ──────────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
events  = pipeline.event_engine.event_store.get_all()

print(f"\n{'='*52}")
print(f"  SEIVR++  —  Processing Complete")
print(f"{'='*52}")
print(f"  Frames processed : {frame_id}")
print(f"  Total time       : {elapsed:.1f}s  ({frame_id/max(elapsed,1):.1f} fps effective)")
print(f"  Events captured  : {len(events)}")
print(f"{'='*52}\n")

# Summary by type
from collections import Counter
event_types = Counter(e["event"] for e in events)
for etype, cnt in sorted(event_types.items()):
    print(f"  {etype:<45} x{cnt}")

# Save to JSON — overwrites previous run, always fresh for query engine
with open(out_path, "w") as f:
    json.dump(events, f, indent=4)

print(f"\n[SEIVR++] {out_path} saved ({len(events)} records).")