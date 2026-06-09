from ultralytics import YOLO


class SimpleTracker:

    def __init__(self, model="yolov8s.pt"):

        self.model = YOLO(model)

        self.class_map = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            25: "umbrella"
        }

        # ── ID-stability: remember last known bbox per track ID ──
        # When ByteTrack loses a track briefly and re-issues a NEW id for the
        # same object, we catch it by IoU-matching against the previous frame.
        self._prev_tracks = {}   # id → bbox (last frame)
        self._id_remap    = {}   # new_id → canonical_id
        self._iou_thresh  = 0.45 # minimum IoU to call it the same object

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (a[2]-a[0])*(a[3]-a[1]))
        areaB = max(1, (b[2]-b[0])*(b[3]-b[1]))
        return inter / (areaA + areaB - inter + 1e-6)

    # ------------------------------------------------------------------
    def update(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        raw_tracks = []

        for r in results:
            boxes = r.boxes
            if boxes.id is None:
                continue
            for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):
                x1, y1, x2, y2 = box.tolist()
                class_id = int(cls)
                if class_id not in self.class_map:
                    continue
                raw_tracks.append({
                    "id":    int(track_id),
                    "bbox":  [int(x1), int(y1), int(x2), int(y2)],
                    "class": self.class_map[class_id]
                })

        # ── ID-stability pass ───────────────────────────────────────
        # For every track returned this frame, check if a *different* id
        # from last frame has a high IoU with it — if so, remap to keep
        # the canonical (older) id.
        current_ids = {t["id"] for t in raw_tracks}
        new_prev    = {}

        for t in raw_tracks:
            tid  = t["id"]
            bbox = t["bbox"]

            # Resolve any earlier remap first
            canonical = self._id_remap.get(tid, tid)

            # If this id is *brand new* this frame, try to match against
            # a previously tracked bbox that has now disappeared from current_ids
            if tid not in self._prev_tracks:
                best_iou = self._iou_thresh
                best_old = None
                for old_id, old_bbox in self._prev_tracks.items():
                    if old_id not in current_ids:          # old track is absent
                        iou = self._iou(bbox, old_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_old = old_id
                if best_old is not None:
                    # Remap new id → old canonical id
                    old_canonical = self._id_remap.get(best_old, best_old)
                    self._id_remap[tid] = old_canonical
                    canonical = old_canonical

            t["id"] = canonical
            new_prev[canonical] = bbox

        # Update previous-frame snapshot
        self._prev_tracks = new_prev

        # ── NMS: merge duplicate boxes of same class with IoU > 0.5 ──
        merged_tracks = []
        used = set()
        for i, t1 in enumerate(raw_tracks):
            if i in used:
                continue
            for j, t2 in enumerate(raw_tracks):
                if i != j and t1["class"] == t2["class"] \
                        and self._iou(t1["bbox"], t2["bbox"]) > 0.5:
                    used.add(j)
            merged_tracks.append(t1)

        return merged_tracks
