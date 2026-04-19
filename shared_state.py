# shared_state.py
import threading

lock = threading.Lock()
latest_frame = None

def write_frame(frame):
    global latest_frame
    with lock:
        latest_frame = frame.copy()

def read_frame():
    with lock:
        return latest_frame