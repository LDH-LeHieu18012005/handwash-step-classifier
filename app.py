import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import time

# Cấu hình
CLASSES = ['Step_1', 'Step_2', 'Step_3', 'Step_4', 'Step_5', 'Step_6']
IMG_WIDTH, IMG_HEIGHT = 270, 210
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mô hình
class HandWashResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights='DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))
    def forward(self, x):
        return self.model(x)

model = HandWashResNet().to(DEVICE)
model = torch.load("resnet18-predict-gen1.pt", map_location=DEVICE, weights_only=False)
model.eval()

# Sử dụng  ID webcam cụ thể 
CAMERA_ID = 0
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

expected_sequence = CLASSES.copy()
current_step_index = 0
stable_label = None
stable_start_time = None
stable_duration_required = 5.0
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if current_step_index >= len(expected_sequence):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    img = resized.astype(np.float32) / 255.0
    img = img[np.newaxis, np.newaxis, ...]
    img_tensor = torch.from_numpy(img).float().to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        class_idx = torch.argmax(outputs, dim=1).item()
        label = CLASSES[class_idx]

    curr_time = time.time()
    expected_label = expected_sequence[current_step_index]

    if label == expected_label:
        if stable_label == label:
            if stable_start_time and curr_time - stable_start_time >= stable_duration_required:
                print(label.replace("Step_", ""))
                current_step_index += 1
                stable_label = None
                stable_start_time = None
        else:
            stable_label = label
            stable_start_time = curr_time
    else:
        if label != stable_label:
            stable_label = None
            stable_start_time = None

    prev_time = curr_time

    cv2.putText(frame, f"Predicted: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Waiting: {expected_label}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)
    cv2.imshow('Startup Project', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
