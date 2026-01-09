from ultralytics import YOLO
import os

# =========================
# CONFIG
# =========================
DATA_YAML = "datasets/detect_food/data.yaml"   # ⚠️ 改成你的實際路徑
MODEL_BASE = "yolov8n.pt"                       # 起始模型
EPOCHS = 100
IMG_SIZE = 640
DEVICE = "mps"   # Mac 有 Apple Silicon 用 "mps"，沒有就改 "cpu"

PROJECT = "meat_project"
NAME = "v2_multifood_doneness"

# =========================

def main():
    # 1️⃣ 載入 YOLOv8 預訓練模型
    model = YOLO(MODEL_BASE)

    # 2️⃣ 訓練
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME
    )

    # 3️⃣ 驗證（用 val set）
    metrics = model.val()

    print("\n✅ 訓練完成！")
    print(f"📁 模型位置：{PROJECT}/{NAME}/weights/best.pt")
    print("📊 驗證指標：")
    print(metrics)


if __name__ == "__main__":
    main()
