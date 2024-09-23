import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

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

def apply_dynamic_quantization(model):
    from torch.quantization import quantize_dynamic
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},  # 通常量化全连接层
        dtype=torch.qint8
    )
    return quantized_model

def save_quantized_model(model, path='./predict_quantized/quantized_model_weights.pt'):
    torch.save(model.state_dict(), path)

def main():
    # 加载模型
    model = load_model()
    
    # 应用动态量化
    quantized_model = apply_dynamic_quantization(model)
    
    # 保存量化模型
    save_quantized_model(quantized_model)

    print('finished')

if __name__ == '__main__':
    main()