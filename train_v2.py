from ultralytics import YOLO

DATA_YAML = "datasets/detect_food/data.yaml"
MODEL_BASE = "yolov8n.pt"
EPOCHS = 100
IMG_SIZE = 640
DEVICE = "mps"

PROJECT = "meat_project"
NAME = "v2_multifood_doneness"

def main():
    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,

        # ---- NMS / too-many-boxes fixes ----
        conf=0.25,        # try 0.25 first, then 0.35 if still spams
        max_det=200,      # cap boxes per image (try 100 if still slow)

        # ---- stability fixes (dense + previously messy labels) ----
        mosaic=0.0,
        mixup=0.0,

        # ---- speed: you already do model.val() afterwards ----
        val=False,
    )

    metrics = model.val(
        data=DATA_YAML,
        conf=0.25,
        max_det=200
    )

    print("\n✅ 訓練完成！")
    print(f"📁 模型位置：{PROJECT}/{NAME}/weights/best.pt")
    print("📊 驗證指標：")
    print(metrics)

if __name__ == "__main__":
    main()
