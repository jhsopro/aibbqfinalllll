import serial
import time

PORT = "/dev/cu.usbmodemFX2348N1"   # ⚠️ 換成你現在看到的
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # 等 Arduino reset

print("📡 Listening to MAX6675...")

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()
    if line.startswith("TEMP_C:"):
        temp_c = float(line.replace("TEMP_C:", ""))
        print(f"🔥 Grill Temp: {temp_c:.2f} °C")
    time.sleep(0.1)
