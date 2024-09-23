import torch
from torchvision import transforms
from PIL import Image

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_path, '..')  # 获取上一级目录作为基础路径
print(f'base_path={base_path}')  # 输出基础路径
sys.path.insert(0, base_path)  # 将基础路径添加到系统路径中

from models import model_changed_name as create_model

def load_model():
    model = create_model()
    model.load_state_dict(torch.load('./weight/tmp.pt'))
    model.eval()
    return model

def prepare_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度
    return image

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, 1)
        return predicted.item()

def main():
    model = load_model()
    image = prepare_image('data/own_val/photo/1/12-001.tiff')
    prediction = predict(model, image)
    print("Predicted class index:", prediction)

if __name__ == '__main__':
    main()