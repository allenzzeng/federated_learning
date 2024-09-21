import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

# 自定义的数据集类，继承自torchvision.datasets.ImageFolder
class ExtendedImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(ExtendedImageDataset, self).__init__(root, transform=transform)
    
    def __getitem__(self, index):
        # 调用父类的__getitem__来获取图像和标签
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        
        # 解析文件名以获取额外的四个参数
        filename = os.path.splitext(os.path.basename(path))[0]
        params = filename.split('-')[2:6]  # 从文件名获取参数
        params = torch.tensor([float(param) for param in params], dtype=torch.float32)
        
        # 返回图像、参数和标签
        return image, params, target

# 修改后的get_dataset函数，使用ExtendedImageDataset加载数据
def get_dataset():
    data_dir_train = r'./data/own_train'
    data_dir_val = r'./data'
    train_datasets = []
    eval_datasets = []

    # 获取训练数据文件夹
    folders = [f for f in os.listdir(data_dir_train) if os.path.isdir(os.path.join(data_dir_train, f))]

    # 为每个文件夹创建一个数据集
    for folder in folders:
        train_dataset = ExtendedImageDataset(os.path.join(data_dir_train, folder, 'photo'), transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        train_datasets.append(train_dataset)

    # 创建验证数据集
    eval_dataset = ExtendedImageDataset(os.path.join(data_dir_val, 'own_val', 'photo'), transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    return train_datasets, eval_dataset



# 使用示例
train_datasets, eval_dataset = get_dataset()
train_loader = torch.utils.data.DataLoader(train_datasets[0], batch_size=32, shuffle=True)  # 例子中只使用第一个数据集
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)

# 从加载器中获取一些数据以进行测试
for images, params, labels in train_loader:
    print("Images:", images.shape)
    print("Params:", params)
    print("Labels:", labels)
    break  # 只展示第一批数据