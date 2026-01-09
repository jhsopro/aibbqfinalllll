# yolo_infer.py
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def detect(self, frame):
        results = self.model.predict(frame, conf=0.35, imgsz=640, verbose=False)
        detections = []

        r = results[0]
        if r.boxes is None:
            return detections

        for box, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int)
        ):
            label = self.model.names[cls]
            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": box
            })

        return detections
