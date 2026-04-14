"""
Vision.py — Static image carousel (development fallback) + tkinter turkey controller

The image carousel (run/VisionRunner) is unchanged from the original.
faketurkeys() opens a small tkinter window for driving the turkey with
WASD/space and publishing the appropriate protocol payloads.

Usage in main.py (unchanged):
    vision = Vision(video_publisher, visionobject_publisher)
    await asyncio.gather(..., vision.VisionRunner())

Controls:
    WASD  — move turkey around the 0-100 coordinate space (1 unit = 1 m)
    SPACE — toggle cloak (publishes Turkeys: [])

Starts cloaked. Dot is orange when visible, blue when cloaked.
"""

import asyncio
import os
import threading
import tkinter as tk
import cv2


class FakeVision:

    # --- carousel config ---
    CAPTURE_RATE_HZ = 0.02
    CAPTURE_PERIOD  = 1.0 / CAPTURE_RATE_HZ
    JPEG_QUALITY    = 80
    IMAGE_DIR       = os.path.dirname(os.path.abspath(__file__))
    IMAGE_FILES     = ["turkey1.jpg", "turkey2.jpg", "turkey3.jpg"]

    # --- controller config ---
    GRID_SIZE       = 100       # coordinate space 0-100 (1 unit = 1 m)
    CELL_PX         = 6         # pixels per unit
    # TURKEY_SPEED    = 0.175       # m per tick (0.5 * 20 hz = 10 m/s)
    TURKEY_SPEED    = 0.6       # m per tick (0.5 * 20 hz = 10 m/s)
    
    PUBLISH_HZ      = 20

    COLOR_VISIBLE   = "#EF9F27"  # orange — turkey is visible to backend
    COLOR_CLOAKED   = "#378ADD"  # blue   — turkey is hidden (Turkeys: [])

    def __init__(self, video_publisher, visionobject_publisher):
        self._publisher        = video_publisher
        self._visionobject_pub = visionobject_publisher
        self._frames           = []
        self._index            = 0

        self._tx      = 20.0
        self._ty      = 20.0
        self._cloaked = True      # start cloaked
        self._keys    = {"w": False, "a": False, "s": False, "d": False}
        self._ui_done = False

    # ------------------------------------------------------------------
    # Image carousel (unchanged)
    # ------------------------------------------------------------------

    def _load_images(self) -> bool:
        for filename in self.IMAGE_FILES:
            path = os.path.join(self.IMAGE_DIR, filename)
            img  = cv2.imread(path)
            if img is None:
                print(f"[VISION] Could not load {path}")
                return False
            ok, buf = cv2.imencode(
                ".jpg", img,
                [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY]
            )
            if not ok:
                print(f"[VISION] Could not encode {filename}")
                return False
            self._frames.append(buf.tobytes())
            print(f"[VISION] Loaded {filename}")
        return True

    def _next_frame(self) -> bytes:
        frame       = self._frames[self._index]
        self._index = (self._index + 1) % len(self._frames)
        return frame

    async def run(self):
        return

        if not self._load_images():
            print("[VISION] Failed to load images — stopping")
            return
        print(f"[VISION] Carousel running at {self.CAPTURE_RATE_HZ} Hz")
        loop = asyncio.get_event_loop()
        while True:
            start = loop.time()
            await self._publisher.publish(self._next_frame())
            elapsed = loop.time() - start
            await asyncio.sleep(max(0, self.CAPTURE_PERIOD - elapsed))

    # ------------------------------------------------------------------
    # Tkinter controller UI (runs on its own thread)
    # ------------------------------------------------------------------

    def _run_ui(self):
        G  = self.GRID_SIZE
        PX = self.CELL_PX
        W  = G * PX

        root = tk.Tk()
        root.title("Turkey Controller")
        root.resizable(False, False)
        root.configure(bg="#111")

        canvas = tk.Canvas(root, width=W, height=W, bg="#1a1a18", highlightthickness=0)
        canvas.pack(padx=12, pady=(12, 4))

        for i in range(0, G + 1, 10):
            canvas.create_line(i * PX, 0, i * PX, W, fill="#2a2a28")
            canvas.create_line(0, i * PX, W, i * PX, fill="#2a2a28")

        status_var = tk.StringVar()
        tk.Label(root, textvariable=status_var, font=("Courier", 10),
                 fg="#aaa", bg="#111").pack(pady=(4, 8))

        turkey_dot = canvas.create_oval(0, 0, 10, 10,
                                        fill=self.COLOR_CLOAKED, outline="")

        def update():
            cx, cy = self._tx * PX, self._ty * PX
            canvas.coords(turkey_dot, cx - 5, cy - 5, cx + 5, cy + 5)
            color = self.COLOR_CLOAKED if self._cloaked else self.COLOR_VISIBLE
            canvas.itemconfig(turkey_dot, fill=color)
            cloak_str = "CLOAKED" if self._cloaked else "VISIBLE"
            status_var.set(f"x: {self._tx:5.1f}  y: {self._ty:5.1f}  [{cloak_str}]")

        def key_press(e):
            k = e.keysym.lower()
            if k in self._keys:
                self._keys[k] = True
            if k == "space":
                self._cloaked = not self._cloaked

        def key_release(e):
            k = e.keysym.lower()
            if k in self._keys:
                self._keys[k] = False

        root.bind("<KeyPress>",   key_press)
        root.bind("<KeyRelease>", key_release)
        root.focus_set()

        def tick():
            spd = self.TURKEY_SPEED
            if self._keys["w"]: self._ty = max(0.0,   self._ty - spd)
            if self._keys["s"]: self._ty = min(100.0, self._ty + spd)
            if self._keys["a"]: self._tx = max(0.0,   self._tx - spd)
            if self._keys["d"]: self._tx = min(100.0, self._tx + spd)
            update()
            root.after(int(1000 / self.PUBLISH_HZ), tick)

        tick()
        root.mainloop()
        self._ui_done = True

    # ------------------------------------------------------------------
    # Protocol publisher
    # ------------------------------------------------------------------

    def _build_payload(self) -> dict:
        if self._cloaked:
            return {"Id": "DynamicObjects", "Turkeys": [], "Humans": []}
        return {
            "Id": "DynamicObjects",
            "Turkeys": [[round(self._tx, 1), round(self._ty, 1), 0.0]],
            "Humans": [],
        }

    async def faketurkeys(self):
        self._ui_done = False
        threading.Thread(target=self._run_ui, daemon=True).start()
        print("[VISION] Turkey controller window opened — WASD=move  SPACE=cloak")

        period = 1.0 / self.PUBLISH_HZ
        last_payload = None

        while not self._ui_done:
            start = asyncio.get_event_loop().time()

            payload = self._build_payload()
            if payload != last_payload:
                await self._visionobject_pub.publish(payload)
                last_payload = payload

            elapsed = asyncio.get_event_loop().time() - start
            await asyncio.sleep(max(0, period - elapsed))

    # ------------------------------------------------------------------
    # Entry point (unchanged)
    # ------------------------------------------------------------------

    async def VisionRunner(self):
        await asyncio.gather(self.run(), self.faketurkeys())