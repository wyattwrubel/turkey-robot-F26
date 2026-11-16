"""
Vision/Confirm.py: k-of-n confirmation (replaces W26's pass-through filter).

W26's temporal filter was two versions pasted into one file and confirmed every
single-frame detection, so one false frame could start a chase. Here a turkey
is only reported once the same track was seen in at least K of its last N
processed frames.

Association is greedy nearest-centroid within a gate (a fraction of image
width), which holds up at low frame rates better than box overlap does.

    conf = Confirmer(k=3, n=5)
    tracks = conf.update(turkey_detections, image_width)   # every frame, even if empty
    confirmed = [t for t in tracks if t.confirmed]
"""

import time
from collections import deque
from dataclasses import dataclass, field

from Vision import config


@dataclass
class Track:
    id: int
    box: tuple
    conf: float
    hits: deque = field(default_factory=deque)   # 1 = seen, 0 = missed, last N frames
    misses: int = 0
    t_first: float = 0.0
    t_confirmed: float | None = None

    @property
    def confirmed(self):
        return self.t_confirmed is not None

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class Confirmer:
    def __init__(self, k=None, n=None, gate=None, drop_misses=None):
        self.k = k or config.CONFIRM_K
        self.n = n or config.CONFIRM_N
        self.gate = gate or config.TRACK_GATE
        self.drop = drop_misses or config.TRACK_DROP_MISSES
        self.tracks: list[Track] = []
        self._next_id = 0
        self.confirm_latencies = []      # seconds from first sighting to confirmation (F4)

    def update(self, detections, image_width, now=None):
        """detections: objects with .box and .conf (one class). Returns all live tracks."""
        now = time.monotonic() if now is None else now
        gate_px = self.gate * image_width
        unmatched = list(range(len(detections)))

        # greedy: each track takes its nearest unmatched detection inside the gate
        for tr in sorted(self.tracks, key=lambda t: -t.conf):
            best, best_d = None, gate_px
            cx, cy = tr.center
            for i in unmatched:
                x1, y1, x2, y2 = detections[i].box
                d = ((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2
                d = d ** 0.5
                if d <= best_d:
                    best, best_d = i, d
            if best is None:
                tr.hits.append(0)
                tr.misses += 1
            else:
                unmatched.remove(best)
                det = detections[best]
                tr.box, tr.conf = det.box, det.conf
                tr.hits.append(1)
                tr.misses = 0
            while len(tr.hits) > self.n:
                tr.hits.popleft()

        for i in unmatched:
            det = detections[i]
            self.tracks.append(Track(self._next_id, det.box, det.conf, deque([1]), 0, now))
            self._next_id += 1

        for tr in self.tracks:
            if tr.t_confirmed is None and sum(tr.hits) >= self.k:
                tr.t_confirmed = now
                self.confirm_latencies.append(now - tr.t_first)
            # a confirmed track that has gone quiet loses confirmation
            if tr.t_confirmed is not None and sum(tr.hits) == 0:
                tr.t_confirmed = None

        self.tracks = [t for t in self.tracks if t.misses < self.drop]
        return self.tracks

    def confirmed_tracks(self):
        """Confirmed tracks seen in the current frame (a missed frame drops them from output,
        the planner's 1.5 s lost-visual window covers short gaps)."""
        return [t for t in self.tracks if t.confirmed and t.hits and t.hits[-1] == 1]
