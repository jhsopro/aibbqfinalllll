# server.py
import cv2
import time
from flask import Flask, Response
import shared_state

app = Flask(__name__)

def generate():
    while True:
        frame = shared_state.read_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>AI BBQ SYSTEM</title>
        <style>
            body { margin: 0; background: #111; display: flex; 
                   justify-content: center; align-items: center; 
                   height: 100vh; }
            img { width: 100%; max-width: 1280px; height: auto; }
        </style>
    </head>
    <body>
        <img src="/video">
    </body>
    </html>
    """

def start_server():
    app.run(host="0.0.0.0", port=8080, threaded=True, use_reloader=False)