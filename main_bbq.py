# main_bbq.py
import cv2
import time
from ultralytics import YOLO

from rules import is_done
from center_zone import poly_from_frac, bbox_triggers_zone, CENTER_POLY_FRAC
from temp_sensor import MAX6675, get_mode

# ================= CONFIG =================
WEBCAM_INDEX = 0
MODEL_PATH = "./meat_project/v2_multifood_doneness4/weights/best.pt"

CONF_THRESH = 0.05
IMG_SIZE = 640
CHECK_MODE = "overlap"

WARNING_TEXT = "MOVE IT TO THE SIDE"
WARNING_COOLDOWN_SEC = 1.0

COLOR_WARN = (0, 0, 255)
COLOR_OK = (0, 255, 0)
COLOR_BOX = (255, 180, 0)
# =========================================

# INIT
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(WEBCAM_INDEX)

temp_sensor = MAX6675(bus=0, device=0)

last_capture = time.monotonic()
last_flip = time.monotonic()
last_warning_time = 0.0

print("🔥 AI BBQ SYSTEM STARTED")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    zone_poly = poly_from_frac(w, h, CENTER_POLY_FRAC)

    temp_c = temp_sensor.read_temp_c()
    mode = get_mode(temp_c)

    now = time.monotonic()
    warning_active = False

    # 9.8s → YOLO
    if now - last_capture >= 9.8:
        results = model.predict(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)
        r = results[0]

        if r.boxes is not None:
            for box, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int)
            ):
                label = model.names[cls]

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # CENTER SAFETY
                if label.startswith("chicken") and bbox_triggers_zone(box, zone_poly, CHECK_MODE):
                    last_warning_time = now
                    warning_active = True
                    continue

                # DONENESS LOGIC
                done = is_done(label, conf, mode)
                status = "DONE" if done else "COOKING"

                cv2.putText(frame, status,
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            COLOR_OK if done else COLOR_WARN, 2)

        last_capture = now

    # WARNING persistence
    if now - last_warning_time <= WARNING_COOLDOWN_SEC:
        warning_active = True

    if warning_active:
        cv2.rectangle(frame, (0, 0), (w, 80), COLOR_WARN, -1)
        cv2.putText(frame, "WARNING!", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        cv2.putText(frame, WARNING_TEXT, (240, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # FLIP TIMER
    if now - last_flip >= 10:
        print("🔄 FLIP NOW")
        last_flip = now

    # HUD
    temp_txt = "N/A" if temp_c is None else f"{temp_c:.1f}C"
    cv2.putText(frame, f"MODE {mode} | TEMP {temp_txt}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.polylines(frame, [zone_poly], True, (0, 255, 255), 3)

    cv2.imshow("AI BBQ SYSTEM", frame)
    if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
        break

cap.release()
temp_sensor.close()
cv2.destroyAllWindows()
