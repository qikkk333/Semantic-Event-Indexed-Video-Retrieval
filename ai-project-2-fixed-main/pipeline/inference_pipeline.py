import os
import cv2
import torch
import numpy as np
from torchvision import transforms
import yaml

from events.temporal_memory import PersonMemory
from models.tracking.bytetrack_tracker import SimpleTracker
from models.action.vit_action_model import ActionViT
from models.appearance.color_extractor import extract_color
from events.event_engine import EventEngine


# ---------- LOAD ZONES ----------
with open("configs/zones.yaml", "r") as f:
    zone_config = yaml.safe_load(f)

zones = {}
for name, data in zone_config["zones"].items():
    zones[name] = np.array(data["points"])


# ---------- CONSTANTS ----------
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# ---------- COLOUR PALETTE ----------
_PALETTE = [
    (255, 99,  71),   # tomato
    (50,  205, 50),   # lime green
    (30,  144, 255),  # dodger blue
    (255, 165,  0),   # orange
    (138,  43, 226),  # blue-violet
    (0,   206, 209),  # dark turquoise
    (255, 20,  147),  # deep pink
    (0,   191, 255),  # deep sky blue
    (124, 252,  0),   # lawn green
    (220,  20,  60),  # crimson
]

def _id_color(pid):
    return _PALETTE[pid % len(_PALETTE)]


# ---------- VIT TRANSFORM ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- ACTION ICON MAP ----------
ACTION_ICON = {
    "run":   ">>",
    "walk":  ">",
    "stand": "||",
}

# ---------- DISPLAY LABEL HELPER ----------
def _friendly_class(cls):
    if cls == "person_U":
        return "Person+Umbrella"
    return cls.capitalize()


def _draw_label(frame, x1, y1, x2, y2, pid, cls, extra, color):
    """Draw a professional pill-shaped label above the bounding box."""
    tag = f"#{pid}  {_friendly_class(cls)}  {extra}"
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.52
    thickness  = 1

    (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, thickness)

    pad_x, pad_y = 7, 4
    lx1 = max(x1, 0)
    ly2 = max(y1 - 1, th + pad_y * 2)
    lx2 = lx1 + tw + pad_x * 2
    ly1 = ly2 - th - pad_y * 2

    # Clip to frame width
    fh, fw = frame.shape[:2]
    if lx2 > fw:
        shift = lx2 - fw
        lx1 -= shift
        lx2 -= shift
    lx1 = max(lx1, 0)

    # Semi-transparent filled background
    overlay = frame.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Coloured left accent bar
    cv2.rectangle(frame, (lx1, ly1), (lx1 + 4, ly2), color, -1)

    # White text
    cv2.putText(frame, tag, (lx1 + 8, ly2 - pad_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_box(frame, x1, y1, x2, y2, color, thickness=2):
    """Draw a bounding box with corner brackets instead of a solid rectangle."""
    cl = 18  # corner length
    cv2.line(frame, (x1, y1), (x1 + cl, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + cl), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - cl, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + cl), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + cl, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - cl), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - cl, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - cl), color, thickness)


def _draw_hud(frame, frame_id, person_count, vehicle_count, fps_val):
    """Overlay a minimal HUD bar at the top of the frame."""
    fh, fw = frame.shape[:2]
    bar_h = 36

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    fs   = 0.52
    th   = 1

    cv2.putText(frame, "SEIVR++ | Surveillance Event Intelligence",
                (10, 24), font, fs, (180, 180, 180), th, cv2.LINE_AA)

    stats = f"Frame {frame_id:05d}   Persons {person_count}   Vehicles {vehicle_count}"
    if fps_val > 0:
        stats += f"   {fps_val:.1f} fps"
    (sw, _), _ = cv2.getTextSize(stats, font, fs, th)
    cv2.putText(frame, stats, (fw - sw - 10, 24),
                font, fs, (100, 220, 100), th, cv2.LINE_AA)

    cv2.line(frame, (0, bar_h), (fw, bar_h), (60, 60, 60), 1)


# ── UMBRELLA OVERLAP HELPER ──────────────────────────────────────────────────

def _umbrella_overlaps_person(px1, py1, px2, py2, ux1, uy1, ux2, uy2,
                               iou_thresh=0.10):
    """
    Return True when an umbrella box sufficiently overlaps a person box.

    FIX: the original code computed the *union* boundary (using max/min in the
    wrong order) which always gave overlap == 0 and therefore never triggered
    the person_U classification.

    Correct intersection:
        ix1 = max(px1, ux1)   ← rightmost left edge
        iy1 = max(py1, uy1)   ← bottommost top edge
        ix2 = min(px2, ux2)   ← leftmost right edge
        iy2 = min(py2, uy2)   ← topmost bottom edge

    We also expand the person box slightly upward (umbrellas are held above
    the head) before testing, and use overlap/umbrella_area as the metric so
    that a small umbrella still triggers the label.
    """
    # Expand person bbox slightly upward to catch umbrella above head
    expand_up = int((py2 - py1) * 0.30)
    epx1, epy1, epx2, epy2 = px1 - 8, py1 - expand_up, px2 + 8, py2 + 8

    # Correct intersection
    ix1 = max(epx1, ux1)
    iy1 = max(epy1, uy1)
    ix2 = min(epx2, ux2)
    iy2 = min(epy2, uy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return False

    u_area = max(1, (ux2 - ux1) * (uy2 - uy1))
    return (inter / u_area) >= iou_thresh


# ─────────────────────────────────────────────────────────────────────────────

class InferencePipeline:

    def __init__(self):
        self.tracker    = SimpleTracker()
        self.prev_zone  = {}
        self.entry_side = {}
        self.prev_position = {}
        self.speed_history = {}

        # Multi-frame buffer for ViT
        self.frame_buffer = {}
        self.buffer_size  = 8
        self.classes = ['run', 'stand', 'walk']

        # Action smoothing
        self.last_actions   = {}
        self.action_history = {}
        self.action_hold    = {}
        self.motion_state   = {}
        self.last_colors    = {}
        self.seen_ids       = set()

        # ── Sticky class memory ──────────────────────────────────────
        # Once a person is classified as person_U, keep that label even
        # if the umbrella disappears for a few frames (occlusion, etc.).
        # Key: track_id  Value: {"class": str, "frames_since_umbrella": int}
        self._sticky_class = {}
        # How many frames to keep person_U label after umbrella last seen
        self._umbrella_sticky_frames = 20

        # ViT inference throttle
        self._vit_interval   = 3
        self._vit_frame_cnt  = {}

        # FPS measurement
        self._fps_t0     = None
        self._fps_frames = 0
        self._fps_val    = 0.0

        checkpoint_path = "runs/vit_action_finetuned.pth"
        if os.path.exists(checkpoint_path):
            self.action_model = ActionViT(checkpoint=checkpoint_path, pretrained=False)
        else:
            self.action_model = ActionViT(pretrained=True)

        self.event_engine = EventEngine()
        self.memory       = PersonMemory()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.action_model.to(self.device)
        self.action_model.eval()

        print(f"[SEIVR++] Running on: {self.device.upper()}")


    # ------------------------------------------------------------------
    def _get_action(self, pid, crop):
        """Return smoothed action label for a person crop."""
        cnt = self._vit_frame_cnt.get(pid, 0)
        self._vit_frame_cnt[pid] = cnt + 1

        run_vit = (cnt % self._vit_interval == 0)

        if run_vit:
            crops = self.frame_buffer.get(pid, [crop])
            t_list = [transform(c) for c in crops if c is not None and c.size != 0]
            if not t_list:
                return self.last_actions.get(pid, "walk")

            t = torch.stack(t_list).to(self.device)
            raw_action = self.action_model.predict(t, conf_thresh=0.0, use_smoothing=False)
            action = raw_action if raw_action != "unknown" else self.last_actions.get(pid, "walk")
        else:
            action = self.last_actions.get(pid, "walk")

        # Temporal majority-vote smoothing (last 7 frames)
        N = 7
        if pid not in self.action_history:
            self.action_history[pid] = []
        self.action_history[pid].append(action)
        if len(self.action_history[pid]) > N:
            self.action_history[pid].pop(0)

        from collections import Counter
        top_action, top_count = Counter(self.action_history[pid]).most_common(1)[0]
        smoothed = top_action if top_count >= 4 else self.last_actions.get(pid, top_action)

        # Hold buffer: resist brief flips
        hold = self.action_hold.get(pid, {"action": smoothed, "count": 1})
        if hold["action"] == smoothed:
            hold["count"] += 1
        else:
            hold = {"action": smoothed, "count": 1}

        prev = self.last_actions.get(pid, smoothed)
        if prev in ("walk", "run") and smoothed == "stand" and hold["count"] < 6:
            smoothed = prev
        if prev == "run"  and smoothed == "walk" and hold["count"] < 5:
            smoothed = "run"
        if prev == "walk" and smoothed == "run"  and hold["count"] < 4:
            smoothed = "walk"

        hold["action"] = smoothed
        self.action_hold[pid]  = hold
        self.last_actions[pid] = smoothed
        return smoothed


    # ------------------------------------------------------------------
    def process_frame(self, frame, frame_id):
        import time

        # FPS counter
        if self._fps_t0 is None:
            self._fps_t0 = time.time()
        self._fps_frames += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._fps_val    = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t0     = time.time()

        tracks = self.tracker.update(frame)

        person_count  = 0
        vehicle_count = 0

        if len(tracks) == 0:
            _draw_hud(frame, frame_id, 0, 0, self._fps_val)
            return frame

        # ---------- PASS 1: collect umbrella boxes ----------
        umbrella_boxes = [t["bbox"] for t in tracks if t["class"] == "umbrella"]

        # ---------- PASS 2: process each track ----------
        for t in tracks:
            if t["class"] == "umbrella":
                continue

            pid       = t["id"]
            obj_class = t["class"]
            bbox      = t["bbox"]

            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()
            bbox = list(bbox)
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # ── Umbrella association (FIXED) ──────────────────────────
            # Also handles sticky classification: once person_U is set,
            # keep it for _umbrella_sticky_frames frames even if umbrella
            # disappears momentarily.
            if obj_class == "person":
                umbrella_seen_now = False
                for ub in umbrella_boxes:
                    ux1, uy1, ux2, uy2 = ub
                    if _umbrella_overlaps_person(x1, y1, x2, y2,
                                                  ux1, uy1, ux2, uy2):
                        umbrella_seen_now = True
                        break

                if umbrella_seen_now:
                    obj_class = "person_U"
                    self._sticky_class[pid] = {
                        "class": "person_U",
                        "frames_since_umbrella": 0
                    }
                else:
                    # Check sticky memory
                    sticky = self._sticky_class.get(pid)
                    if sticky and sticky["class"] == "person_U":
                        sticky["frames_since_umbrella"] += 1
                        if sticky["frames_since_umbrella"] <= self._umbrella_sticky_frames:
                            obj_class = "person_U"
                        else:
                            # Sticky expired — revert
                            del self._sticky_class[pid]

            # ── Zone detection (vehicles) ─────────────────────────────
            zone = "none"
            if obj_class in VEHICLE_CLASSES:
                for name, polygon in zones.items():
                    if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
                        zone = name
                        break

            # ── Entry side ────────────────────────────────────────────
            frame_width = frame.shape[1]
            if obj_class in VEHICLE_CLASSES and pid not in self.entry_side:
                if   center_x < frame_width * 0.25: self.entry_side[pid] = "LEFT_ENTRY"
                elif center_x > frame_width * 0.75: self.entry_side[pid] = "RIGHT_ENTRY"
                else:                                self.entry_side[pid] = "CENTER_ENTRY"

            # ── Route events ──────────────────────────────────────────
            route_event = None
            if obj_class in VEHICLE_CLASSES:
                previous = self.prev_zone.get(pid)
                if zone != "none" and previous != zone:
                    entry = self.entry_side.get(pid, "UNKNOWN")
                    if previous is None:
                        route_event = f"VEHICLE_FROM_{entry}_ENTERED_{zone.upper()}"
                    elif zone == "left_path":
                        route_event = f"VEHICLE_FROM_{entry}_TURNED_LEFT"
                    elif zone == "right_path":
                        route_event = f"VEHICLE_FROM_{entry}_TURNED_RIGHT"
                    elif zone == "straight_path":
                        route_event = f"VEHICLE_FROM_{entry}_WENT_STRAIGHT"
                    if route_event:
                        print(f"EVENT: {route_event} | ID {pid}")
                self.prev_zone[pid] = zone

            # ── Crop ──────────────────────────────────────────────────
            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

            # ── Frame buffer (persons) ────────────────────────────────
            if obj_class in ("person", "person_U") and crop is not None and crop.size != 0:
                if pid not in self.frame_buffer:
                    self.frame_buffer[pid] = []
                self.frame_buffer[pid].append(crop)
                if len(self.frame_buffer[pid]) > self.buffer_size:
                    self.frame_buffer[pid].pop(0)

            # ── Action recognition ────────────────────────────────────
            if obj_class in ("person", "person_U"):
                action = self._get_action(pid, crop) if crop is not None and crop.size != 0 \
                         else self.last_actions.get(pid, "walk")
                person_count += 1
            else:
                action = "vehicle"
                vehicle_count += 1

            # ── Vehicle colour ────────────────────────────────────────
            if obj_class in VEHICLE_CLASSES:
                if pid not in self.last_colors and crop is not None and crop.size != 0:
                    self.last_colors[pid] = extract_color(crop)
                color_name = self.last_colors.get(pid, "unknown")
            else:
                color_name = None

            # ── Temporal memory & event engine ────────────────────────
            self.memory.update(pid, action, frame_id)
            self.event_engine.process(
                pid, obj_class, action, color_name,
                [x1, y1, x2, y2], frame_id, route_event=route_event
            )

            # ── PROFESSIONAL VISUALISATION ────────────────────────────
            id_color = _id_color(pid)
            _draw_box(frame, x1, y1, x2, y2, id_color, thickness=2)

            if obj_class in ("person", "person_U"):
                icon  = ACTION_ICON.get(action, "")
                extra = f"{action.upper()}  {icon}"
            else:
                z_str = f"[{zone}]" if zone != "none" else ""
                extra = f"{color_name or ''}  {z_str}".strip()

            _draw_label(frame, x1, y1, x2, y2, pid, obj_class, extra, id_color)

        _draw_hud(frame, frame_id, person_count, vehicle_count, self._fps_val)
        return frame
