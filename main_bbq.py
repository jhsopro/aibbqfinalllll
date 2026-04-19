# main_bbq.py
import threading
import shared_state
from server import start_server
import numpy as np
import cv2
import time
from ultralytics import YOLO

from notifier import notify
from center_zone import poly_from_frac, bbox_triggers_zone, CENTER_POLY_FRAC
from temp_sensor import TempSensorSerial, get_mode, DemoTempController
from rules import (
    is_beef_label,
    beef_target_flips,
    beef_is_done,
    food_is_done_color,
    parse_label,
)
import math 

# config
WEBCAM_INDEX = 0
MODEL_PATH = "./meat_project/v2_multifood_doneness6/weights/best.pt"

CONF_THRESH = 0.60
IMG_SIZE = 640
IOU_THRESH = 0.35
MAX_DET = 30
CHECK_MODE = "overlap"


FLIP_INTERVAL_SEC = 15.0    
CAPTURE_AT_SEC = 14.7       
YOLO_FPS_LIMIT = 12       


WARNING_COOLDOWN_SEC = 1.0
WARNING_TEXT = "Kindly remove the chicken from the central area"


BEEF_PREF = "medium"      


COL_BG_BANNER = (20, 20, 20)      
COL_TEXT = (245, 245, 245)        
COL_BOX = (230, 230, 230)       
COL_ZONE = (160, 160, 160)       
COL_DONE = (60, 200, 120)         
COL_WARN = (40, 40, 230)     

TOP_H = 54
MARGIN = 14

COL_TOP = (18,18,18)
COL_CARD = (24,24,24)
COL_MUTED = (160,160,160)
COL_WHITE = (245,245,245)
COL_BAR_BG = (60,60,60)
COL_BAR_FG = (245,245,245)

FLIP_FLASH_SEC = 0.8


ZONE_FLASH_PERIOD_SEC = 2.0
ZONE_ALPHA_MIN = 0.06
ZONE_ALPHA_MAX = 0.24
ZONE_WARN_TEXT = "NO CHICKEN"

BEEF_STAGE_ORDER = ["raw", "medium", "cooked"]
BEEF_READY_WINDOW_SEC = 12.0

UI_SCALE = 1.25

START_TEMP_C = 300.0         
PREHEAT_HOLD_SEC = 2.0

READY_SPLASH_SEC = 2.0

SIDEBAR_MIN_CONF = 0.55

DEMO_BYPASS_HOTKEY = ord("b")  
DEMO_RESET_HOTKEY  = ord("r")  
PRESENTATION_START_HOTKEY = ord("p") 

MODE_HYST = 10.0

OUT_W, OUT_H = 1280, 720     
ROTATE = 0                  
FLIP_H = False              
FLIP_V = False

CHICKEN_COOKED_CONF_MIN = 0.80
CHICKEN_COOKED_STREAK   = 6
BEEF_MISSING_RESET_SEC = 1.5



class SimpleIoUTracker:
    """
    Dependency-free tracker.
    Assigns stable IDs using IoU matching between frames.
    Good enough for BBQ top-down shots and avoids ultralytics bytetrack/lap issues.
    """
    def __init__(self, iou_thresh=0.35, max_age=20):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.next_id = 1
        self.tracks = {} 

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else (inter / union)

    def update(self, boxes):
        """
        boxes: Nx4 float array
        returns: ids: N int array
        """
        boxes = np.asarray(boxes, dtype=np.float32)
        n = len(boxes)
        ids = np.full(n, -1, dtype=int)

   
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        if n == 0:
            return ids


        used_tracks = set()
        for i in range(n):
            best_tid = None
            best_iou = self.iou_thresh
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = self._iou(boxes[i], t["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is None:
                # new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": boxes[i].copy(), "age": 0}
                ids[i] = tid
                used_tracks.add(tid)
            else:
                # update existing
                self.tracks[best_tid]["box"] = boxes[i].copy()
                self.tracks[best_tid]["age"] = 0
                ids[i] = best_tid
                used_tracks.add(best_tid)

        return ids

def fit_to_size(frame, out_w, out_h):
    """Letterbox (no stretching) to exactly out_w x out_h."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame

    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = (0, 0, 0)  # force pure black background
    x = (out_w - nw) // 2
    y = (out_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def update_mode_hysteresis(prev_mode: str, temp_c: float) -> str:
    if temp_c is None:
        return prev_mode

    # boundaries: A < 500, B < 700, else C
    if prev_mode == "A":
        # only promote to B if clearly above 500
        return "B" if temp_c >= (500.0 + MODE_HYST) else "A"

    if prev_mode == "B":
        # demote to A only if clearly below 500
        if temp_c <= (500.0 - MODE_HYST):
            return "A"
        # promote to C only if clearly above 700
        if temp_c >= (700.0):
            return "C"
        return "B"

    if prev_mode == "C":
        # only demote to B if clearly below 700
        return "B" if temp_c <= (700.0 - MODE_HYST) else "C"

    # fallback
    return "B"

def normalize_beef_state(state: str) -> str:
    s = (state or "").lower().strip()

    # collapse model labels into your 3 UI buckets
    if s in ("raw",):
        return "raw"
    if s in ("mediumrare", "mediumnrare", "medium", "med", "rare"):
        return "medium"
    if s in ("cooked", "welldone", "well_done", "well-done"):
        return "cooked"

    return s

def beef_ui_text(norm_state: str) -> str:
    if norm_state == "raw":
        return "RAW"
    if norm_state == "medium":
        return "MEDIUM"
    if norm_state == "cooked":
        return "WELL DONE"
    return (norm_state or "UNKNOWN").upper()

def beef_stage_level(state: str) -> int:
    try:
        return BEEF_STAGE_ORDER.index(state)
    except ValueError:
        return -1

def beef_line_name(tid: int) -> str:
    return f"BEEF #{tid}"

def draw_pill(img, text, x, y, fg=COL_TEXT, bg=COL_BOX, scale=0.6, thickness=2, pad=8):
    scale *= UI_SCALE
    thickness = max(1, int(thickness * UI_SCALE))
    pad = int(pad * UI_SCALE)

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img,
                  (x, y - th - base - pad),
                  (x + tw + pad*2, y + pad),
                  bg, -1)
    cv2.putText(img, text,
                (x + pad, y - pad//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, fg, thickness, cv2.LINE_AA)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_panel(img, x, y, w, h, bg=(18,18,18), alpha=0.92, shadow=True):
    x, y, w, h = map(int, (x,y,w,h))
    if shadow:
        sh = img.copy()
        cv2.rectangle(sh, (x+3, y+3), (x+w+3, y+h+3), (0,0,0), -1)
        img[:] = cv2.addWeighted(sh, 0.25, img, 0.75, 0)

    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w,y+h), bg, -1)
    img[:] = cv2.addWeighted(overlay, float(alpha), img, 1.0-float(alpha), 0)

def draw_text(img, text, x, y, size=0.7, color=(245,245,245), thick=2):
    size *= UI_SCALE
    thick = max(1, int(thick * UI_SCALE))
    cv2.putText(img, text, (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                float(size), color, thick, cv2.LINE_AA)

def draw_chip(img, text, x, y, bg=(28,28,28), fg=(245,245,245)):
    scale = 0.6 * UI_SCALE
    thick = max(1, int(2 * UI_SCALE))

    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    w = tw + int(22 * UI_SCALE)
    h = th + base + int(14 * UI_SCALE)

    draw_panel(img, x, y, w, h, bg=bg, alpha=0.92, shadow=False)
    draw_text(img, text, x + int(11 * UI_SCALE), y + h - int(10 * UI_SCALE),
              size=0.6, color=fg, thick=2)
    return w

def draw_progress_bar(img, x, y, w, h, progress, fg=(245,245,245), bg=(60,60,60)):
    x, y, w, h = map(int, (x,y,w,h))
    progress = max(0.0, min(1.0, float(progress)))
    cv2.rectangle(img, (x,y), (x+w,y+h), bg, -1)
    fill = int(w * progress)
    if fill > 0:
        cv2.rectangle(img, (x,y), (x+fill,y+h), fg, -1)

def food_display_name(food: str) -> str:
    # sidebar beef keys may look like "beef#3" — always display "BEEF"
    if isinstance(food, str) and (food == "beef" or food.startswith("beef#") or food in ("beef1", "beef2")):
        return "BEEF"
    return food.upper()

def row_color(done: bool) -> tuple:
    return COL_DONE if done else COL_CARD

def arm_system(now):
    notify("ready", cooldown=1.0, beep=True)
    return True, now + READY_SPLASH_SEC, True

def draw_text_right(img, text, x_right, y, size=0.6, color=COL_MUTED, thick=2):
    size *= UI_SCALE
    thick = max(1, int(thick * UI_SCALE))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, float(size), thick)
    x = int(x_right - tw)
    cv2.putText(img, text, (x, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                float(size), color, thick, cv2.LINE_AA)

def draw_preheat_screen(frame, w, h, temp_c, target_c, force_progress_zero=False):
    # Fade background
    WARNING_HOLD_SEC = 0.8
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.28, frame, 0.72, 0)

    # Card (bigger height to guarantee spacing)
    card_w = int(560 * UI_SCALE)
    card_h = int(190 * UI_SCALE)
    x = MARGIN
    y = int(TOP_H + 18 * UI_SCALE)

    draw_panel(frame, x, y, card_w, card_h, bg=COL_CARD, alpha=0.94, shadow=True)

    # Values
    if temp_c is None:
        temp_str = "--"
        sub_str = "SENSOR NOT READY"
        progress = 0.0
    else:
        temp_str = f"{temp_c:0.0f}C"
        sub_str = f"TARGET {target_c:0.0f}C"
        progress = max(0.0, min(1.0, float(temp_c) / float(target_c)))

    if force_progress_zero:
        progress = 0.0

    draw_text(frame, "PREHEATING",
              x + int(20 * UI_SCALE), y + int(34 * UI_SCALE),
              size=0.62, color=COL_MUTED, thick=2)

    draw_text(frame, "TEMP",
              x + int(20 * UI_SCALE), y + int(70 * UI_SCALE),
              size=0.50, color=COL_MUTED, thick=2)

    draw_text(frame, temp_str,
              x + int(20 * UI_SCALE), y + int(122 * UI_SCALE),
              size=1.15, color=COL_WHITE, thick=3)

    draw_text(frame, sub_str,
              x + int(22 * UI_SCALE), y + int(152 * UI_SCALE),
              size=0.50, color=COL_MUTED, thick=2)

    # Progress bar (right side)
    bar_x = x + int(250 * UI_SCALE)
    bar_y = y + int(98 * UI_SCALE)
    bar_w = int(280 * UI_SCALE)
    bar_h = int(12 * UI_SCALE)

    draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h, progress,
                      fg=COL_BAR_FG, bg=COL_BAR_BG)

def draw_ready_splash(frame, w, h):
    # Small dim, not full blackout

    card_w = int(480 * UI_SCALE)
    card_h = int(140 * UI_SCALE)
    x = MARGIN
    y = int(TOP_H + 22 * UI_SCALE)

    draw_panel(frame, x, y, card_w, card_h, bg=COL_CARD, alpha=0.94, shadow=True)

    # Badge line
    draw_text(frame, "READY", x + int(20*UI_SCALE), y + int(44*UI_SCALE),
              size=0.95, color=COL_DONE, thick=3)

    draw_text(frame, "PLACE FOOD NOW", x + int(20*UI_SCALE), y + int(96*UI_SCALE),
              size=0.78, color=COL_WHITE, thick=2)

    draw_text(frame, "COOKING TIMER WILL START AUTOMATICALLY",
              x + int(20*UI_SCALE), y + int(126*UI_SCALE),
              size=0.55, color=COL_MUTED, thick=2)


def draw_sidebar(img, sidebar, w, h, flip_x, flip_y, flip_w, flip_h):
    if not sidebar:
        return

    panel_w = int(280 * UI_SCALE)
    panel_x = int(flip_x)
    panel_y = int(flip_y + flip_h + int(12 * UI_SCALE))
    panel_x = min(panel_x, w - panel_w - MARGIN)
    panel_x = max(MARGIN, panel_x)

    panel_h_max = h - panel_y - MARGIN
    panel_h = min(int(320 * UI_SCALE), panel_h_max)
    if panel_h < int(120 * UI_SCALE):
        return

    draw_panel(img, panel_x, panel_y, panel_w, panel_h, bg=COL_CARD, alpha=0.92, shadow=True)

    draw_text(img, "DETECTED",
              panel_x + int(16 * UI_SCALE),
              panel_y + int(28 * UI_SCALE),
              size=0.65, color=COL_MUTED, thick=2)

    def item_priority(food, info):
    # Beef rows (they have "status")
        if "status" in info:
            done_flag = 1 if info.get("done", False) else 0
            return (done_flag, 0, food)

    # Normal foods (count-based)
        done_n = int(info.get("done", 0))
        cook_n = int(info.get("cooking", 0))

        has_done = 1 if done_n > 0 else 0
        primary = done_n if has_done else cook_n

        return (has_done, primary, food)

    items = sorted(
        sidebar.items(),
        key=lambda kv: item_priority(kv[0], kv[1]),
        reverse=True
    )


    pad = int(16 * UI_SCALE)
    row_h = int(34 * UI_SCALE)
    y = panel_y + int(52 * UI_SCALE)

    for food, info in items[:8]:

        # ✅ SPECIAL CASE: beef uses status text, not x-counts
        if "status" in info:
            left = food_display_name(food)
            right = str(info["status"])

            is_done_row = (right.strip().upper() in ("WELL DONE", "COOKED"))


            cv2.line(img,
                     (panel_x + pad, y + int(8 * UI_SCALE)),
                     (panel_x + panel_w - pad, y + int(8 * UI_SCALE)),
                     (40, 40, 40), 1)

            dot_col = COL_DONE if is_done_row else (110, 110, 110)
            cv2.circle(img, (panel_x + pad, y), int(4 * UI_SCALE), dot_col, -1)

            draw_text(img, left,
                      panel_x + pad + int(10 * UI_SCALE),
                      y + int(6 * UI_SCALE),
                      size=0.62, color=COL_WHITE, thick=2)

            draw_text_right(img, right,
                            panel_x + panel_w - pad,
                            y + int(6 * UI_SCALE),
                            size=0.56, color=(COL_DONE if is_done_row else COL_MUTED), thick=2)

            y += row_h
            if y > panel_y + panel_h - int(14 * UI_SCALE):
                break
            continue

        # ----- normal foods -----
        cook_n = int(info.get("cooking", 0))
        done_n = int(info.get("done", 0))

        rows = []
        if cook_n > 0:
            rows.append(("COOKING", cook_n, False))
        if done_n > 0:
            rows.append(("COOKED", done_n, True))

        for state_word, n, is_done_row in rows:
            left = food_display_name(food)
            right = f"{state_word}  x{n}"

            cv2.line(img,
                     (panel_x + pad, y + int(8 * UI_SCALE)),
                     (panel_x + panel_w - pad, y + int(8 * UI_SCALE)),
                     (40, 40, 40), 1)

            dot_col = COL_DONE if is_done_row else (110, 110, 110)
            cv2.circle(img, (panel_x + pad, y), int(4 * UI_SCALE), dot_col, -1)

            draw_text(img, left,
                      panel_x + pad + int(10 * UI_SCALE),
                      y + int(6 * UI_SCALE),
                      size=0.62, color=COL_WHITE, thick=2)

            draw_text_right(img, right,
                            panel_x + panel_w - pad,
                            y + int(6 * UI_SCALE),
                            size=0.56, color=(COL_DONE if is_done_row else COL_MUTED), thick=2)

            y += row_h
            if y > panel_y + panel_h - int(14 * UI_SCALE):
                break



def display_food_name(food: str) -> str:
    if food in ("beef1", "beef2"):
        return "BEEF"
    return food.upper()


def main():
    print("✅ Loading YOLO:", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
    if not cap.isOpened():
        raise RuntimeError("Camera failed to open. Try WEBCAM_INDEX=1 or check permissions.")
    
    WIN = "AI BBQ SYSTEM"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)  




    temp_sensor = TempSensorSerial()
    flask_thread = threading.Thread(target=start_server, daemon=True)
    flask_thread.start()
    demo = DemoTempController(start=25.0)   
    demo_enabled = True
    demo.set_target(300.0)


    print("🔥 AI BBQ SYSTEM STARTED")

    system_armed = False
    frozen_temp_c = None
    preheat_started = False
    preheat_temp0 = None  
    preheat_start = None
    ready_splash_until = 0.0
    ready_splash_shown = False


    # State
    last_warning_time = -999.0
    last_flip_time = time.monotonic()
    last_cycle_start = time.monotonic()  
    prev_warning_active = False
    beef_flipped_once = False

    beef_flip_count = {}

    beef_best_stage = -1
    beef_final_done = False
    beef_stage_green_until = 0.0 



    # YOLO throttling
    last_yolo_time = 0.0
    last_results = None

    FLIP_FLASH_SEC = 0.8
    flip_flash_until = 0.0

    toast_text = ""
    toast_until = 0.0

    marsh_flip_count = 0
    marsh_seen = False
    MARSH_TARGET_FLIPS = 2

    mode_state = None
    mode = mode_state


    cooked_streak = {}
    prev_done = {}
    SIDEBAR_TTL_SEC = 1.2  

    last_seen_food = {}  
    last_seen_beef = {}   


    trackers = {}
    beef_stage_streak = {}
    beef_present_cycle = set()
    beef_present_streak = 0         
    beef_present_confirmed = False 


    while True:
        key = 255
        beef_seen = False
        marsh_seen = False
        
        ret, frame = cap.read()
        if not ret or frame is None:
            continue


        if ROTATE == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ROTATE == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif ROTATE == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        if FLIP_H:
            frame = cv2.flip(frame, 1)
        if FLIP_V:
            frame = cv2.flip(frame, 0)
        
        key = cv2.waitKey(1) & 0xFF


        

        if demo_enabled:
            if not preheat_started:
                    # Freeze temp before pressing P
                if frozen_temp_c is None:
                    frozen_temp_c = demo.current
                temp_c = frozen_temp_c
            else:
                    # Once P is pressed, resume real demo updates
                temp_c = demo.update()
        else:
            temp_c = temp_sensor.read_temp_c(debug=False)


            if temp_c is None:
                demo_enabled = True
                demo.current = 25.0
                demo.set_target(300.0)
                temp_c = demo.update()
                print("⚠️ Arduino temp missing → switched back to DEMO")


        if demo_enabled:
            if key == ord("1"):
                demo.set_target(300.0)
            elif key == ord("2"):
                demo.set_target(500.0)
            elif key == ord("3"):
                demo.set_target(700.0)



        now = time.monotonic()
        h, w = frame.shape[:2]

        if key in [DEMO_RESET_HOTKEY, ord("R")]:
            # System state
            system_armed = False
            preheat_started = False
            ready_splash_shown = False
            ready_splash_until = 0.0

            # Temperature freeze (like initial boot)
            demo.current = 25.0            
            demo.set_target(300.0)         
            frozen_temp_c = None
            preheat_temp0 = None
            preheat_start = None

            # Timers
            last_cycle_start = now
            flip_flash_until = 0.0
            toast_text = ""
            toast_until = 0.0
            last_warning_time = -999.0

            # Beef logic
            beef_flip_count.clear()
            beef_present_cycle.clear()
            beef_stage_streak.clear()
            beef_flipped_once = False
            beef_present_streak = 0
            beef_present_confirmed = False
            beef_stage_lock_until = {}

            # Other food logic
            cooked_streak.clear()
            prev_done.clear()
            marsh_flip_count = 0
            marsh_seen = False

            # Sidebar memory
            last_seen_food.clear()
            last_seen_beef.clear()

            # YOLO cache
            last_results = None
            last_yolo_time = 0.0

            notify("reset", cooldown=0.5, beep=True)
            continue


        if (key == PRESENTATION_START_HOTKEY or key == ord("P")) and (not preheat_started):
            preheat_started = True
            preheat_start = None         
            system_armed = False         
            preheat_temp0 = temp_c
            frozen_temp_c = None   


        WARNING_HOLD_SEC = 0.8
        warning_active = (now - last_warning_time) <= WARNING_HOLD_SEC



        zone_poly = poly_from_frac(w, h, CENTER_POLY_FRAC)
        
        # ---- MODE UPDATE (must be inside the loop) ----
        if mode_state is None and temp_c is not None:

            if temp_c < 500:
                mode_state = "A"
            elif temp_c < 700:
                mode_state = "B"
            else:
                mode_state = "C"

        mode_state = update_mode_hysteresis(mode_state, temp_c)
        mode = mode_state
        beef_target = beef_target_flips(preference=BEEF_PREF, mode=mode)


        # ---------------- PREHEAT GATE ----------------
        ARM_TEMP_C = START_TEMP_C - 0.5  # 298.5C counts as "at 300"

        if not preheat_started:
            target_c = demo.target if demo_enabled else START_TEMP_C

            draw_preheat_screen(frame, w, h, temp_c, target_c, force_progress_zero=True)
            shared_state.write_frame(frame)
            cv2.imshow(WIN, frame)
            if key in [27, ord("q"), ord("Q")]:
                break
            continue

        if not system_armed:
            # freeze cooking logic while preheating
            last_cycle_start = now
            flip_flash_until = 0.0
            toast_text = ""
            toast_until = 0.0

            beef_flip_count.clear()
            beef_present_cycle.clear()

            beef_best_stage = -1
            beef_stage_green_until = 0.0
            beef_final_done = False
            beef_last_flip_time = {}     
            beef_stage_lock_until = {}  


            target_c = demo.target if demo_enabled else START_TEMP_C
            draw_preheat_screen(frame, w, h, temp_c, target_c)
            shared_state.write_frame(frame)
            cv2.imshow(WIN, frame)


            if key in [DEMO_BYPASS_HOTKEY, ord("B")]:
                system_armed, ready_splash_until, ready_splash_shown = arm_system(now)
                last_cycle_start = now

            if key in [27, ord("q"), ord("Q")]:
                break

            continue
        
        if ready_splash_shown:
            if now <= ready_splash_until:
                draw_ready_splash(frame, w, h)
                shared_state.write_frame(frame)
                cv2.imshow(WIN, frame)


               
                if key == ord(" "): 
                    ready_splash_until = 0.0

                if key in [27, ord("q"), ord("Q")]:
                    break

                continue 
            else:
                ready_splash_shown = False







        elapsed = now - last_cycle_start
        remaining = FLIP_INTERVAL_SEC - elapsed
        
        if remaining <= 0:
            notify("flip", cooldown=0.3, beep=True)

            flip_flash_until = now + FLIP_FLASH_SEC
            toast_text = "FLIP NOW"
            toast_until = now + 1.2


            for tid in beef_present_cycle:
                beef_flip_count[tid] = beef_flip_count.get(tid, 0) + 1
                beef_flipped_once = True
                beef_last_flip_time[tid] = now
                beef_stage_lock_until[tid] = now + FLIP_INTERVAL_SEC


            beef_present_cycle.clear()
            last_cycle_start = now
            remaining = FLIP_INTERVAL_SEC

            if marsh_seen:
                marsh_flip_count += 1



        cycle_elapsed = now - last_cycle_start
        force_yolo = (cycle_elapsed >= CAPTURE_AT_SEC)

        min_yolo_interval = 1.0 / max(1, YOLO_FPS_LIMIT)
        do_yolo = force_yolo or ((now - last_yolo_time) >= min_yolo_interval)
        results_fresh = do_yolo

        if do_yolo:
            last_yolo_time = now
            results = model.predict(
                frame,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                max_det=MAX_DET,
                imgsz=IMG_SIZE,
                verbose=False
            )
            last_results = results
        else:
            results = last_results


        cv2.polylines(frame, [zone_poly], True, COL_ZONE, 2)


        if results is not None:
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss  = r.boxes.cls.cpu().numpy().astype(int)

                # build food_key list for each detection
                food_keys = []
                for cls_i in clss:
                    label_i = model.names[int(cls_i)]
                    f_i, _ = parse_label(label_i)
                    food_keys.append("beef" if f_i in ("beef1", "beef2") else f_i)

                ids = np.full(len(boxes), -1, dtype=int)


                for fk in set(food_keys):
                    idxs = [i for i, k in enumerate(food_keys) if k == fk]
                    if fk not in trackers:
                        trackers[fk] = SimpleIoUTracker(iou_thresh=0.35, max_age=18)
                    sub_boxes = boxes[idxs]
                    sub_ids = trackers[fk].update(sub_boxes)
                    for j, i in enumerate(idxs):
                        ids[i] = sub_ids[j]




                sidebar = {}
                best_det = {}
                beef_present = False
                beef_sidebar_text = None




                for i, (box, conf, cls, tid) in enumerate(zip(boxes, confs, clss, ids)):

                    label = model.names[cls]
                    food, state = parse_label(label)

                    # collapse beef variants into one logical food
                    if food in ("beef1", "beef2"):
                        food_key = "beef"
                    else:
                        food_key = food

                    sidebar_ok = float(conf) >= SIDEBAR_MIN_CONF


                    if (food_key not in best_det) or (float(conf) > best_det[food_key]["conf"]):
                        best_det[food_key] = {
                            "conf": float(conf),
                            "box": box,
                            "label": label,
                            "state": state
                        }
                    
                    if sidebar_ok:
                        if food_key == "beef":
                            last_seen_beef[int(tid)] = now
                        else:
                            last_seen_food[food_key] = now



                    if food == "marshmallow":
                        marsh_seen = True

                    x1, y1, x2, y2 = map(int, box)


                    if results_fresh and food == "chicken" and bbox_triggers_zone(box, zone_poly, CHECK_MODE):
                        last_warning_time = now
                        notify("warning", cooldown=WARNING_COOLDOWN_SEC, beep=True)





                    if is_beef_label(label):
                        beef_seen = True
                        beef_present_cycle.add(tid)



                        is_best_beef = (food_key == "beef") and ("beef" in best_det) and (best_det["beef"]["label"] == label)

                        state_n = normalize_beef_state(state)

                        if now < beef_stage_lock_until.get(tid, 0.0):
                            if state_n == "cooked":
                                state_n = "medium"  

                        status = beef_ui_text(state_n)



                        required_stage = "medium"
                        required_lvl = beef_stage_level(required_stage)
                        streak_key = ("beef", tid)
                        prev = beef_stage_streak.get(streak_key, 0)

                        if results_fresh:
                            if beef_stage_level(state_n) >= required_lvl:
                                beef_stage_streak[streak_key] = min(prev + 1, 6)
                            else:
                                beef_stage_streak[streak_key] = max(prev - 1, 0)

                        stage_confirmed_this = beef_stage_streak.get(streak_key, 0) >= 3




                        target_flips = beef_target_flips(preference=BEEF_PREF, mode=mode)
                        done = (
                            stage_confirmed_this and
                            beef_flipped_once and
                            beef_flip_count.get(tid, 0) >= target_flips
                        )



                        if done:
                            box_color = COL_DONE
                        elif stage_confirmed_this:
                            box_color = (120, 220, 170)
                        else:
                            box_color = COL_BOX

                    else:

                        is_cooked_frame = food_is_done_color(label, float(conf), mode)
                        if food_key == "chicken":
                            is_cooked_frame = is_cooked_frame and (float(conf) >= CHICKEN_COOKED_CONF_MIN)


                        streak_key = (food_key, int(tid))

                        if results_fresh:
                            cooked_streak[streak_key] = cooked_streak.get(streak_key, 0) + (1 if is_cooked_frame else -1)
                            cooked_streak[streak_key] = max(0, min(10, cooked_streak[streak_key]))

                        if food_key == "chicken":
                            done = cooked_streak.get(streak_key, 0) >= CHICKEN_COOKED_STREAK
                        else:
                            done = cooked_streak.get(streak_key, 0) >= 4

                        if food == "marshmallow":
                            done = (marsh_flip_count >= MARSH_TARGET_FLIPS)
                            status = f"READY ({marsh_flip_count}/{MARSH_TARGET_FLIPS})" if done else f"FLIP ({marsh_flip_count}/{MARSH_TARGET_FLIPS})"
                        else:
                            status = "READY" if done else "COOKING"

                        box_color = COL_DONE if done else COL_BOX

                        was_done = prev_done.get(streak_key, False)
                        if done and not was_done:
                            notify("done", cooldown=0.4, beep=True)
                        prev_done[streak_key] = done


                    if sidebar_ok:
                        if food_key == "beef":
        # beef: show per-piece status line (RAW/MEDIUM/WELL DONE)
                            beef_key = f"beef#{int(tid)}"
                            sidebar[beef_key] = {
                                "status": status,     
                                "done": bool(done),  
                            }
                        else:
        # other foods: keep count-based rows
                            if food_key not in sidebar:
                                sidebar[food_key] = {"cooking": 0, "done": 0}
                            if done:
                                sidebar[food_key]["done"] += 1
                            else:
                                sidebar[food_key]["cooking"] += 1



                    thickness = int((3 if done else 2) * UI_SCALE)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                    draw_pill(
                        frame,
                        display_food_name(food_key),
                        x1,
                        y1,
                        fg=(15, 15, 15),
                        bg=box_color,
                        scale=0.6,
                        thickness=2
                    )



                    status_col = COL_DONE if done else (200, 200, 200)
                    cv2.putText(frame, status, (x1, y2 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2, cv2.LINE_AA)
                    
                if beef_seen:
                    beef_present_streak = min(beef_present_streak + 1, 8)
                else:
                    beef_present_streak = max(beef_present_streak - 1, 0)
            

                beef_present_confirmed = (beef_present_streak >= 3)




        overlay = frame.copy()

        if warning_active:
            
            t = now % ZONE_FLASH_PERIOD_SEC
            phase = t / ZONE_FLASH_PERIOD_SEC 
            breath = 0.5 - 0.5 * math.cos(2 * math.pi * phase) 

            alpha = ZONE_ALPHA_MIN + (ZONE_ALPHA_MAX - ZONE_ALPHA_MIN) * breath
            zone_color = COL_WARN

            cv2.fillPoly(overlay, [zone_poly], zone_color)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.polylines(frame, [zone_poly], True, zone_color, 3)

            if breath > 0.55:
                px, py = int(zone_poly[0][0]), int(zone_poly[0][1])
                draw_pill(frame, ZONE_WARN_TEXT, px, py,
                          fg=(255, 255, 255), bg=zone_color, scale=0.75, thickness=2)
        else:
            alpha = 0.10
            zone_color = COL_ZONE
            cv2.fillPoly(overlay, [zone_poly], zone_color)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.polylines(frame, [zone_poly], True, zone_color, 2)

            px, py = int(zone_poly[0][0]), int(zone_poly[0][1])
            draw_pill(frame, "CENTER ZONE (NO CHICKEN)", px, py, fg=COL_TEXT, bg=(0, 0, 0), scale=0.41, thickness=2)

       # ---------------- TOP BAR (clean) ----------------
        draw_panel(frame, 0, 0, w, TOP_H, bg=COL_TOP, alpha=0.92, shadow=False)


        title = "AI BBQ SYSTEM"
        title_scale = 0.75 * UI_SCALE
        title_thick = max(1, int(2 * UI_SCALE))
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thick)

        draw_text(frame, title, (w - tw)//2, int(36 * UI_SCALE), size=0.75, color=COL_MUTED, thick=2)



        x0, y0 = MARGIN, 10
        mode_chip = f"MODE {mode}"
        temp_chip = "TEMP N/A" if temp_c is None else f"TEMP {temp_c:0.1f}C"
        x0 += draw_chip(frame, mode_chip, x0, y0, bg=(28,28,28)) + 10
        x0 += draw_chip(frame, temp_chip, x0, y0, bg=(28,28,28)) + 10


# ---------------- FLIP CARD (top-right) ----------------
        remaining = clamp(remaining, 0.0, FLIP_INTERVAL_SEC)
        progress = 1.0 - (remaining / FLIP_INTERVAL_SEC)

        panel_w = int(240 * UI_SCALE)
        panel_h = int(92 * UI_SCALE)

        px, py = w - panel_w - MARGIN, TOP_H + MARGIN

        if now <= flip_flash_until:
            draw_panel(frame, px, py, panel_w, panel_h, bg=(245,245,245), alpha=0.95, shadow=True)
            draw_text(frame, "FLIP NOW", px + 16, py + 44, size=0.95, color=(18,18,18), thick=3)
            draw_progress_bar(frame, px + 16, py + panel_h - 18, panel_w - 32, 8, 1.0, fg=(18,18,18), bg=(220,220,220))
        else:
            draw_panel(frame, px, py, panel_w, panel_h, bg=COL_CARD, alpha=0.92, shadow=True)
            draw_text(frame, "NEXT FLIP", px + 16, py + 30, size=0.65, color=COL_MUTED, thick=2)
            draw_text(frame, f"{remaining:0.1f}s", px + 16, py + 68, size=1.05, color=COL_WHITE, thick=3)
            draw_progress_bar(frame, px + 16, py + panel_h - 18, panel_w - 32, 8, progress, fg=COL_BAR_FG, bg=COL_BAR_BG)
        
        if 'sidebar' in locals():
            filtered = {}

            for key, info in sidebar.items():
                if isinstance(key, str) and key.startswith("beef#"):
                    tid = int(key.split("#")[1])
                    if (now - last_seen_beef.get(tid, -1e9)) <= SIDEBAR_TTL_SEC:
                        filtered[key] = info
                else:
                    if (now - last_seen_food.get(key, -1e9)) <= SIDEBAR_TTL_SEC:
                        filtered[key] = info

            draw_sidebar(frame, filtered, w, h, px, py, panel_w, panel_h)

            
        if warning_active:
            banner_h = int(62 * UI_SCALE)
            bx = MARGIN
            by = TOP_H + MARGIN  
            bw = w - (MARGIN * 2)

            draw_panel(frame, bx, by, bw, banner_h, bg=COL_WARN, shadow=True)

            draw_text(frame, "PLEASE REMOVE CHICKEN FROM CENTER AREA",
                      bx + int(16 * UI_SCALE),
                      by + int(40 * UI_SCALE),
                      size=0.78, color=(255,255,255), thick=3)


        if now <= toast_until and toast_text:
            tleft = toast_until - now
            alpha = 0.92 if tleft > 0.4 else max(0.0, tleft / 0.4) * 0.92

            tw, th = 360, 54
            tx = (w - tw) // 2
            ty = h - th - 18
            draw_panel(frame, tx, ty, tw, th, bg=(18,18,18), alpha=alpha, shadow=True)
            draw_text(frame, toast_text, tx + 18, ty + 36, size=0.75, color=(245,245,245), thick=2)

        recent_beef = any((now - t) <= BEEF_MISSING_RESET_SEC for t in last_seen_beef.values())

        if not recent_beef:
            beef_flip_count.clear()
            beef_flipped_once = False


        if not marsh_seen:
            marsh_flip_count = 0

        shared_state.write_frame(frame)
        cv2.imshow(WIN, frame)


        if key in [27, ord("q"), ord("Q")]:
            break



        

    cap.release()
    temp_sensor.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
