from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model_path = '/content/drive/MyDrive/C&D_extract/runs/train/yolov8_D_waste_detection/weights/best.pt'
model = YOLO(model_path)

# Path to the test image
test_image_path = '/content/drive/MyDrive/C&D_extract/test/images/brick_7391617_jpg.rf.9d132b93c1c5f37771a0a92372b727c6.jpg'  # Replace with your test image path

# Run inference
results = model.predict(source=test_image_path, show=True)

# Display the results
plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
