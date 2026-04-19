# temp_sensor.py
# Mac: read temperature from Arduino via Serial (recommended)
# Jetson later: you can replace this with SPI MAX6675 directly if you want.

import time
import random

class DemoTempController:
    """
    Fake temperature generator for demos.
    - ramps in mostly ~2°C jumps (with 1/3°C variation)
    - jitters near target
    - supports changing target via set_target()
    """
    def __init__(self, start=25.0, noise_std=0.8):
        self.current = float(start)
        self.target = float(start)
        self.noise_std = float(noise_std)
        self.last_t = time.monotonic()

    def set_target(self, target):
        self.target = float(target)

    def update(self):
        now = time.monotonic()
        dt = max(0.001, now - self.last_t)
        self.last_t = now

        err = self.target - self.current

        # step-based climb: mostly +2°C, sometimes +1 or +3 (more natural)
        if abs(err) > 6:
            base_step = 2.0
            step = base_step + random.choice([-1.0, 0.0, 0.0, 0.0, +1.0])  # mostly 2
            step = max(1.0, min(3.0, step))
            step = step if err > 0 else -step
        else:
            # near target: slow down and hover
            step = err * 0.25

        self.current += step

        # jitter stronger near target
        near = max(0.0, 1.0 - min(1.0, abs(err) / 25.0))
        self.current += random.gauss(0.0, self.noise_std) * (0.3 + 0.7 * near)

        # clamp
        self.current = max(0.0, min(999.0, self.current))

        # choose ONE:
        return float(int(round(self.current)))   # more "digital sensor"
        # return round(self.current, 1)          # smoother (1 decimal)

def get_mode(temp_c):
    if temp_c is None:
        return "B"
    if temp_c < 500:
        return "A"
    if temp_c < 700:
        return "B"
    return "C"


def demo_temperature(now):
    """
    Fake temperature curve for exhibition:
    - ramps to 300
    - then steps to 500
    - then steps to 700
    """
    t = now % 30.0  # 30s loop

    if t < 8:
        return 25 + (300 - 25) * (t / 8)      # ramp up
    elif t < 16:
        return 300 + (500 - 300) * ((t - 8) / 8)
    elif t < 24:
        return 500 + (700 - 500) * ((t - 16) / 8)
    else:
        return 700

class TempSensorSerial:
    """
    Reads temp from Arduino.
    Expected Arduino output examples:
      "TEMP: 187.25"
      "187.25"
    """
    def __init__(self, port=None, baud=115200, timeout=0.2, print_connect=True):
        import serial
        import serial.tools.list_ports

        self.serial = None
        self.last_temp = None
        self.last_read_time = 0.0

        if port is None:
            ports = list(serial.tools.list_ports.comports())
            # Prefer usbmodem / usbserial on mac
            candidates = [p.device for p in ports if ("usbmodem" in p.device.lower() or "usbserial" in p.device.lower())]
            if not candidates and ports:
                candidates = [ports[0].device]
            if not candidates:
                raise RuntimeError("No serial ports found. Plug in Arduino and try again.")
            port = candidates[0]

        self.serial = serial.Serial(port, baudrate=baud, timeout=timeout)
        # Give Arduino time to reset
        time.sleep(2.5)

        if print_connect:
            print(f"✅ Arduino serial connected: {port} @ {baud}")

    def read_temp_c(self, max_age_sec=5.0, debug=False):

        now = time.monotonic()
        try:
            line = self.serial.readline().decode(errors="ignore").strip()
            if line:
                if debug:
                    print("SERIAL:", repr(line))
                import re
                m = re.search(r"(-?\d+(?:\.\d+)?)", line)
                if m:
                    self.last_temp = float(m.group(1))
                    self.last_read_time = now
        except Exception as e:
            if debug:
                print("SERIAL ERR:", e)

        if self.last_temp is None:
            return None
        if (now - self.last_read_time) > max_age_sec:
            return None
        return self.last_temp




    def close(self):
        try:
            if self.serial:
                self.serial.close()
        except Exception:
            pass
