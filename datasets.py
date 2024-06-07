import os
import torch 
from torchvision import datasets, transforms

def get_dataset():


	data_dir_train=r'./data/own_train'
	data_dir_val=r'./data'
	train_datasets = []
	eval_datasets = []

	folders = [f for f in os.listdir(data_dir_train) if os.path.isdir(os.path.join(data_dir_train, f))]

	for folder in folders:
		train_dataset = datasets.ImageFolder(os.path.join(data_dir_train, folder, 'photo'), transform=transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		]))
		train_datasets.append(train_dataset)

		eval_dataset = datasets.ImageFolder(os.path.join(data_dir_val, 'own_val', 'photo'), transform=transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		]))
	

	return train_datasets, eval_dataset
