import time
import serial
import serial.tools.list_ports

def pick_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "usbmodem" in p.device or "usbserial" in p.device:
            return p.device
    return ports[0].device if ports else None

port = pick_port()
if not port:
    raise SystemExit("No serial ports found")

baud = 115200
print("PORT =", port)
ser = serial.Serial(port, baud, timeout=1)
time.sleep(2)  # Arduino reset

print("Reading raw lines... (Ctrl+C stop)")
while True:
    raw = ser.readline()  # bytes
    if raw:
        print("RAW:", raw)                 # 看原始 bytes
        try:
            print("TXT:", raw.decode("utf-8", errors="replace").strip())
        except Exception as e:
            print("DECODE ERR:", e)
    else:
        print("...timeout (no data)")
    time.sleep(0.2)
