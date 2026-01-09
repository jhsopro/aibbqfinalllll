# temp_sensor.py  (Mac + Arduino USB serial)
import time
import serial
import serial.tools.list_ports


def _find_arduino_port():
    """Auto-find an Arduino-like serial port on macOS."""
    ports = list(serial.tools.list_ports.comports())

    # Prefer /dev/cu.usbmodem* or /dev/cu.usbserial*
    for p in ports:
        if "usbmodem" in p.device or "usbserial" in p.device:
            return p.device

    # Fallback: any /dev/cu.*
    for p in ports:
        if p.device.startswith("/dev/cu."):
            return p.device

    return None


class MAX6675:
    """
    Compatibility wrapper.

    main_bbq.py currently does:
        temp_sensor = MAX6675(bus=0, device=0)

    On macOS we DON'T have SPI, so we ignore bus/device and read temperature
    from Arduino over USB serial instead.

    Arduino should print lines like:
        TEMP_C:182.75
    (or just a number, we handle both)
    """

    def __init__(self, bus=0, device=0, port=None, baud=115200, timeout=1):
        _ = bus, device  # ignored on Mac

        if port is None:
            port = _find_arduino_port()

        if port is None:
            raise RuntimeError("找不到 Arduino serial port（/dev/cu.*）。請先插上 Arduino 再試。")

        self.port = port
        self.baud = baud
        self.ser = serial.Serial(self.port, self.baud, timeout=timeout)
        time.sleep(2)  # Arduino opens serial -> often resets

        self.last_temp_c = None
        print(f"✅ Arduino serial connected: {self.port} @ {self.baud}")

    def read_temp_c(self):
        """
        Try to read one valid temperature from serial.
        Return: float or None (if no valid line)
        """
        # Read a few lines quickly to find a valid one
        for _ in range(5):
            raw = self.ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Accept "TEMP_C:123.45" or just "123.45"
            if "TEMP_C" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    line = parts[-1].strip()

            try:
                temp = float(line)
                self.last_temp_c = temp
                return temp
            except ValueError:
                # ignore non-numeric lines
                pass

        return None

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


def get_mode(temp_c):
    """
    Your mode logic:
      A: 170-190
      B: 190-210
      C: 210-230 (and above)
    """
    if temp_c is None:
        return "B"  # fallback mode if no temp yet

    if 170 <= temp_c < 190:
        return "A"
    elif 190 <= temp_c < 210:
        return "B"
    else:
        return "C"
