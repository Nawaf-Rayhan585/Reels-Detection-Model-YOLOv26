import os, sys, time, base64, threading, json, io, queue, subprocess
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime

# ──────────────────────────────────────────────────────
#  AUTO-INSTALL HELPERS
# ──────────────────────────────────────────────────────
def pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *packages,
         "--break-system-packages", "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for pkg, imp in [("mss", "mss"), ("Pillow", "PIL")]:
    try: __import__(imp)
    except ImportError:
        print(f"  Installing {pkg}..."); pip_install(pkg)

import mss
from PIL import Image

# ──────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────
CONFIG = {
    # YOLOv26 model size: "yolo26n" (fastest) / "yolo26s" / "yolo26m" / "yolo26l" / "yolo26x" (most accurate)
    "yolo_model": "yolo26x",

    # Inference image size — higher = more accurate, slower
    "yolo_imgsz": 1280,

    # Minimum YOLO box confidence to even consider a detection
    "yolo_conf": 0.45,

    "scan_interval_seconds": 8,
    "alarm_beeps": 5,
    "warning_display_seconds": 12,
    "confidence_threshold": 0.55,
    "screenshot_quality": 80,
    "max_screenshot_width": 1280,
}

# YOLO classes that hint at distraction context (phone screens, TV, etc.)
YOLO_DISTRACTION_CLASSES = {
    "cell phone", "remote", "tv", "laptop", "tablet",
    "person",   # multiple people = likely video call or stream
}

# ──────────────────────────────────────────────────────
#  SCREEN CAPTURE
# ──────────────────────────────────────────────────────
def capture_screen_pil() -> Image.Image:
    """Return PIL Image of primary monitor."""
    with mss.mss() as sct:
        mon = sct.monitors[1]
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    max_w = CONFIG["max_screenshot_width"]
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    return img

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CONFIG["screenshot_quality"])
    return base64.b64encode(buf.getvalue()).decode()

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=CONFIG["screenshot_quality"])
    return buf.getvalue()

# ──────────────────────────────────────────────────────
#  YOLOv26 DETECTION ENGINE (100% local, offline, free)
# ──────────────────────────────────────────────────────
class YoloEngine:
    """
    Uses YOLOv26 for object detection + heuristic rules to decide distraction.
    No internet required after model download.
    """

    # Classes that strongly suggest distraction when detected on-screen
    STRONG_DISTRACTION = {"cell phone", "remote", "tv"}
    # Browser/UI keyword heuristics applied to the window title via wmctrl
    SOCIAL_KEYWORDS = [
        "instagram", "tiktok", "youtube shorts", "reels", "snapchat",
        "twitter", "reddit", "facebook", "netflix", "twitch", "discord",
        "whatsapp", "telegram", "9gag", "imgur",
    ]

    def __init__(self, model_name: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("  Installing ultralytics (YOLOv26)...")
            pip_install("ultralytics")
            from ultralytics import YOLO

        print(f"  Loading {model_name}.pt model (downloads on first run)...")
        self.model = YOLO(f"{model_name}.pt")
        self._active_title = ""
        # Try to get active window title for keyword heuristic
        self._title_thread = threading.Thread(
            target=self._poll_window_title, daemon=True)
        self._title_thread.start()

    def _poll_window_title(self):
        """Continuously poll active window title (Linux wmctrl / xdotool)."""
        while True:
            try:
                if sys.platform == "linux":
                    out = subprocess.check_output(
                        ["xdotool", "getactivewindow", "getwindowname"],
                        stderr=subprocess.DEVNULL).decode().strip().lower()
                    self._active_title = out
                elif sys.platform == "darwin":
                    script = 'tell application "System Events" to get name of first process whose frontmost is true'
                    out = subprocess.check_output(
                        ["osascript", "-e", script],
                        stderr=subprocess.DEVNULL).decode().strip().lower()
                    self._active_title = out
                elif sys.platform == "win32":
                    import ctypes
                    hwnd = ctypes.windll.user32.GetForegroundWindow()
                    buf = ctypes.create_unicode_buffer(512)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
                    self._active_title = buf.value.lower()
            except Exception:
                pass
            time.sleep(2)

    def _title_distraction(self) -> tuple[bool, str]:
        t = self._active_title
        for kw in self.SOCIAL_KEYWORDS:
            if kw in t:
                return True, f"Browser showing: {kw.title()}"
        return False, ""

    def analyze(self, img: Image.Image) -> dict:
        import numpy as np
        arr = np.array(img)
        results = self.model(
            arr,
            verbose=False,
            conf=CONFIG["yolo_conf"],
            imgsz=CONFIG["yolo_imgsz"],
        )[0]

        detected = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id].lower()
            conf_val = float(box.conf[0])
            detected.append((label, conf_val))

        # Check window title first (strong signal)
        title_dist, title_reason = self._title_distraction()

        # Check YOLO objects
        strong_hits = [(l, c) for l, c in detected if l in self.STRONG_DISTRACTION]
        # Multiple people on screen → likely watching a stream/video
        people = [(l, c) for l, c in detected if l == "person"]

        is_distracted = False
        confidence = 0.0
        category = None
        description = "Screen looks productive"

        if title_dist:
            is_distracted = True
            confidence = 0.90
            category = title_reason
            description = f"Window title contains social/entertainment keyword"

        elif strong_hits:
            best = max(strong_hits, key=lambda x: x[1])
            is_distracted = True
            confidence = min(0.85, best[1] + 0.15)
            category = {"cell phone": "Phone/Social Media",
                        "remote": "TV/Streaming",
                        "tv": "Streaming video (Netflix etc.)"}[best[0]]
            description = f"Detected {best[0]} on screen"

        elif len(people) >= 2:
            is_distracted = True
            confidence = 0.65
            category = "Streaming video / video call"
            description = f"Multiple people visible ({len(people)}), likely video content"

        else:
            # No strong signal — check for any object accumulation
            if detected:
                description = f"Detected: {', '.join(set(l for l, _ in detected[:4]))}"
            confidence = 0.1

        return {
            "is_distracted": is_distracted,
            "confidence": confidence,
            "category": category,
            "description": description,
            "productive": None if not is_distracted else "Close distracting app and focus",
            "yolo_objects": [l for l, _ in detected],
        }

# ──────────────────────────────────────────────────────
#  SOUND
# ──────────────────────────────────────────────────────
def play_alarm():
    try:
        if sys.platform == "win32":
            import winsound
            for _ in range(CONFIG["alarm_beeps"]):
                winsound.Beep(880, 300); time.sleep(0.15)
        elif sys.platform == "darwin":
            for _ in range(CONFIG["alarm_beeps"]):
                os.system("afplay /System/Library/Sounds/Funk.aiff &"); time.sleep(0.4)
        else:
            for _ in range(CONFIG["alarm_beeps"]):
                os.system("paplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || echo -e '\\a'")
                time.sleep(0.3)
    except Exception:
        for _ in range(CONFIG["alarm_beeps"]):
            print("\a", end="", flush=True); time.sleep(0.3)

# ──────────────────────────────────────────────────────
#  WARNING OVERLAY
# ──────────────────────────────────────────────────────
class WarningOverlay:
    def __init__(self):
        self.root = None
        self.active = False
        self._queue = queue.Queue()
        threading.Thread(target=self._run_tk, daemon=True).start()

    def _run_tk(self):
        self.root = tk.Tk(); self.root.withdraw()
        self.root.after(100, self._process_queue)
        self.root.mainloop()

    def _process_queue(self):
        try:
            while True: self._queue.get_nowait()()
        except queue.Empty: pass
        if self.root: self.root.after(100, self._process_queue)

    def show(self, info): self._queue.put(lambda: self._show(info))
    def hide(self): self._queue.put(self._hide)

    def _show(self, info):
        if self.active: self._hide()
        self.active = True
        w = tk.Toplevel(self.root)
        self.win = w
        w.title("⚠️ DISTRACTION DETECTED")
        w.attributes("-topmost", True); w.attributes("-alpha", 0.94)
        w.overrideredirect(True)
        sw, sh = w.winfo_screenwidth(), w.winfo_screenheight()
        w.geometry(f"{sw}x{sh}+0+0"); w.configure(bg="#0a0a0f")

        # Border
        tk.Frame(w, bg="#ff1a1a").place(relx=.5, rely=.5, anchor="center", width=692, height=492)
        frame = tk.Frame(w, bg="#0d0d1a"); frame.place(relx=.5, rely=.5, anchor="center", width=684, height=484)

        engine_label = info.get("_engine", "AI")
        tk.Label(frame, text="🚨", font=("Segoe UI Emoji", 60), bg="#0d0d1a", fg="#ff3333").pack(pady=(32,4))
        tk.Label(frame, text="DISTRACTION DETECTED",
                 font=tkfont.Font(family="Helvetica", size=26, weight="bold"),
                 bg="#0d0d1a", fg="#ff3333").pack()
        tk.Label(frame, text=f"Detected by {engine_label}",
                 font=tkfont.Font(family="Helvetica", size=10),
                 bg="#0d0d1a", fg="#555577").pack()

        cat = info.get("category","Unknown"); conf = info.get("confidence",0)
        tk.Label(frame, text=f"⚡  {cat}  ({int(conf*100)}% confidence)",
                 font=tkfont.Font(family="Helvetica", size=15),
                 bg="#0d0d1a", fg="#ffaa00").pack(pady=(10,2))

        desc = info.get("description","")
        if desc:
            tk.Label(frame, text=desc, font=tkfont.Font(family="Helvetica", size=12),
                     bg="#0d0d1a", fg="#cccccc", wraplength=580, justify="center").pack(pady=(2,6))

        # Show YOLO objects if available
        yolo_objs = info.get("yolo_objects")
        if yolo_objs:
            tk.Label(frame, text=f"Objects: {', '.join(yolo_objs[:6])}",
                     font=tkfont.Font(family="Helvetica", size=10),
                     bg="#0d0d1a", fg="#557755").pack()

        tk.Label(frame, text="🎯  Get back to work. Your future self will thank you.",
                 font=tkfont.Font(family="Helvetica", size=13, slant="italic"),
                 bg="#0d0d1a", fg="#44ff88").pack(pady=(8,0))

        bar_f = tk.Frame(frame, bg="#0d0d1a"); bar_f.pack(pady=(16,0), fill="x", padx=40)
        self._cv = tk.StringVar()
        tk.Label(bar_f, textvariable=self._cv,
                 font=tkfont.Font(family="Helvetica", size=11),
                 bg="#0d0d1a", fg="#888888").pack()
        self._pbar = tk.Canvas(bar_f, height=8, bg="#222233", highlightthickness=0)
        self._pbar.pack(fill="x", pady=(4,0))

        tk.Button(frame, text="✕  I'm back to work",
                  font=tkfont.Font(family="Helvetica", size=13, weight="bold"),
                  bg="#ff3333", fg="white", activebackground="#cc0000",
                  bd=0, padx=20, pady=10, cursor="hand2",
                  command=self._hide).pack(pady=(14,0))

        tk.Label(frame, text=f"Detected at {datetime.now():%H:%M:%S}",
                 font=tkfont.Font(family="Helvetica", size=9),
                 bg="#0d0d1a", fg="#444455").pack(pady=(6,0))

        self._total = CONFIG["warning_display_seconds"]
        self._rem = self._total
        self._tick()

    def _tick(self):
        if not self.active: return
        try:
            if self._rem <= 0: self._hide(); return
            self._cv.set(f"Auto-closing in {self._rem}s")
            w = self._pbar.winfo_width() or 580
            r = int(255 * self._rem / self._total)
            g = int(255 * (1 - self._rem / self._total))
            self._pbar.delete("all")
            self._pbar.create_rectangle(0, 0, int(w * self._rem / self._total), 8,
                                        fill=f"#{r:02x}{g:02x}33", outline="")
            self._rem -= 1
            self.win.after(1000, self._tick)
        except tk.TclError: pass

    def _hide(self):
        self.active = False
        try:
            if hasattr(self, "win"): self.win.destroy()
        except tk.TclError: pass

# ──────────────────────────────────────────────────────
#  TERMINAL DASHBOARD
# ──────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, engine_name):
        self.engine_name = engine_name
        self.scans = self.distractions = 0
        self.start = datetime.now(); self.last = None
        self.status = "Starting..."; self.lock = threading.Lock()

    def _up(self):
        d = datetime.now() - self.start
        h, r = divmod(int(d.total_seconds()), 3600); m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update(self, status, result=None):
        with self.lock:
            self.status = status
            if result:
                self.scans += 1
                if result.get("is_distracted"): self.distractions += 1; self.last = datetime.now()

    def render(self, result=None):
        with self.lock:
            os.system("clear" if os.name != "nt" else "cls")
            W = 62
            print(f"\033[1;36m{'─'*W}\033[0m")
            print(f"\033[1;36m  🔍 DISTRACTION DETECTOR  │  Engine: {self.engine_name}\033[0m")
            print(f"\033[1;36m{'─'*W}\033[0m")
            print(f"  ⏱  Uptime:        {self._up()}")
            print(f"  🔎 Scans:         {self.scans}")
            pct = self.distractions / self.scans * 100 if self.scans else 0
            col = "\033[31m" if pct > 30 else "\033[33m" if pct > 10 else "\033[32m"
            print(f"  🚨 Distractions:  {col}{self.distractions} ({pct:.1f}%)\033[0m")
            if self.last:
                print(f"  ⚡ Last caught:   {int((datetime.now()-self.last).total_seconds())}s ago")
            print(f"\n  📡 {self.status}")
            print(f"\033[1;36m{'─'*W}\033[0m")
            if result:
                if result.get("is_distracted"):
                    print(f"\n  \033[31m🚨 {result.get('category','?')}\033[0m")
                    print(f"  \033[33m   {result.get('description','')}\033[0m")
                    objs = result.get("yolo_objects")
                    if objs: print(f"  \033[90m   objects: {', '.join(objs[:5])}\033[0m")
                else:
                    print(f"\n  \033[32m✅ FOCUSED — {result.get('description','')}\033[0m")
            print(f"\n  Ctrl+C to stop\n")

# ──────────────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────────────
def main():
    print("\033[1;36m")
    print("╔══════════════════════════════════════════════╗")
    print("║        🔍 DISTRACTION DETECTOR               ║")
    print("║        Engine: YOLOv26 (local, offline)      ║")
    print("╚══════════════════════════════════════════════╝")
    print("\033[0m")

    engine = YoloEngine(CONFIG["yolo_model"])
    engine_name = f"YOLOv26 ({CONFIG['yolo_model']})"
    print(f"  ✅ {engine_name} engine ready")

    overlay = WarningOverlay()
    dash = Dashboard(engine_name)

    print(f"\n  📺 Scanning every {CONFIG['scan_interval_seconds']}s  |  threshold {CONFIG['confidence_threshold']}")
    print("  Press Ctrl+C to stop\n")
    time.sleep(1)

    last_result = None

    while True:
        try:
            dash.status = "📷 Capturing screen..."
            dash.render(last_result)

            img = capture_screen_pil()

            dash.status = f"🤖 Analyzing with {engine_name}..."
            dash.render(last_result)

            result = engine.analyze(img)
            result["_engine"] = engine_name   # tag for overlay
            last_result = result
            dash.update(result.get("category", "scan"), result)

            if result.get("is_distracted") and result.get("confidence", 0) >= CONFIG["confidence_threshold"]:
                dash.status = f"🚨 DISTRACTION: {result.get('category')}"
                dash.render(result)
                threading.Thread(target=play_alarm, daemon=True).start()
                overlay.show(result)
                def _auto_hide():
                    time.sleep(CONFIG["warning_display_seconds"]); overlay.hide()
                threading.Thread(target=_auto_hide, daemon=True).start()
            else:
                overlay.hide()
                dash.status = "✅ Focused"
                dash.render(result)

            for rem in range(CONFIG["scan_interval_seconds"], 0, -1):
                dash.status = f"⏳ Next scan in {rem}s..."
                dash.render(last_result)
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\033[1;33m  👋 Detector stopped. Stay focused!\033[0m\n")
            overlay.hide(); sys.exit(0)

        except json.JSONDecodeError:
            dash.status = "⚠️  Bad JSON from AI — retrying"
            time.sleep(CONFIG["scan_interval_seconds"])

        except Exception as e:
            dash.status = f"⚠️  {type(e).__name__}: {str(e)[:55]}"
            dash.render(last_result)
            time.sleep(CONFIG["scan_interval_seconds"])

if __name__ == "__main__":
    main()
