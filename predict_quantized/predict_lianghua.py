import torch
from torchvision import transforms
from PIL import Image

# import sys
# sys.path.insert(0, '/data/home/code/tmp_study_code/federated_learning/tst/federated_learning')
# from models import model_changed_name as create_model

import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(current_path, '..')  # 获取上一级目录作为基础路径
print(f'base_path={base_path}')  # 输出基础路径
sys.path.insert(0, base_path)  # 将基础路径添加到系统路径中

from models import model_changed_name as create_model

def load_quantized_model(model_path):
    # 创建模型实例
    model = create_model()
    # 应用动态量化
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    # 加载量化模型的状态字典
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        predicted = torch.max(output, 1)[1]
    return predicted.item()

def main():
    model_path = './predict_quantized/quantized_model_weights.pt'
    image_path = 'data/own_val/photo/1/12-001.tiff'  # Update this path to your image file

    model = load_quantized_model(model_path)
    image = prepare_image(image_path)
    prediction = predict(model, image)
    
    print("Predicted class index:", prediction)

if __name__ == '__main__':
    main()