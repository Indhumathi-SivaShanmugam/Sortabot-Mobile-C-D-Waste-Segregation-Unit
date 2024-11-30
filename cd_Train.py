from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with 'yolov8s.pt' or larger models if needed

# Train the model
model.train(
    data='/content/drive/MyDrive/C&D_extract/data.yaml',  # Path to YAML file
    epochs=50,  # Number of epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size
    project='/content/drive/MyDrive/C&D_extract/runs/train',  # Directory to save training results
    name='yolov8_D_waste_detection',  # Name of the training session
    exist_ok=True  # Allow overwriting if there's a previous training run
)
