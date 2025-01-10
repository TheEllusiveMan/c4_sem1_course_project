import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# # Загрузка модели ResNet18
# resnet18_weights = models.ResNet18_Weights.IMAGENET1K_V1
# model = models.resnet18(weights=resnet18_weights).to('cpu')
# model.eval()

model = YOLO('best_yolov9m_c4_practica.pt') # best 0.47
# model = YOLO('best_yolo11m_c4_practica.pt') # best 0.3
# model = YOLO('best_y11m.pt')
# model = YOLO('best_11m_v3.pt') # best 0.11

# Преобразования для модели
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model.predict(image)
    return output
