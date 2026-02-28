from ultralytics import YOLO

# 1. Load a model
# 'n' stands for nano (fastest), 's' for small, 'm' for medium
model = YOLO("yolo11n.pt") 

# 2. Train the model
results = model.train(
    data="data.yaml",     # path to your dataset config
    epochs=100,           # number of training passes
    imgsz=640,            # image size
    batch=16,             # number of images per batch
    device=0,             # use device=0 for GPU, device='cpu' for CPU
    name="robotic_arm_v1" # name for the result folder
)

# 3. Evaluate performance
metrics = model.val()

# 4. Export the model for use on your Raspberry Pi
# NCNN or ONNX formats are best for edge devices
model.export(format="ncnn")