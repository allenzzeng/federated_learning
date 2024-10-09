[English](readme.md)
# 联邦学习项目

本项目实现了联邦学习和迁移学习，并支持自定义backbone网络。

## 代码参考
本项目的代码灵感来源于以下资源：
- [WZMIAOMIAO的图像处理深度学习仓库](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master)
- [DingXiaoH的RepVGG仓库](https://github.com/DingXiaoH/RepVGG)
- [FederatedAI的联邦学习实践仓库](https://github.com/FederatedAI/Practicing-Federated-Learning/tree/main/chapter03_Python_image_classification)

## 数据结构
- **客户端数据集**：位于 `./data/own_train/`。
- **验证数据集**：位于 `data/own_val/photo`。
- **迁移学习训练集**：位于 `data/own_val/train`。
- **迁移学习验证集**：位于 `data/own_val/val`。

## 配置参数
存储于 `utils` 文件夹中：
- `batch_size`：批次大小。
- `local_epochs`：每个客户端进行的训练次数。
- `global_epochs`：全局训练次数。
- `lr`：客户端学习率。
- `momentum`：动量参数，用于优化算法，改善梯度下降的收敛性能。（momentum 是优化算法中的一个超参数，主要用于改善梯度下降的收敛性能，特别是在处理非凸优化问题时。它在随机梯度下降（Stochastic Gradient Descent, SGD）和其变种中经常被使用，我用Adam）
- `lambda`：用于客户端更新的参数，对全局模型进行更新。（客户端返回diff，all客户端累加更新到weight_accumulator，server在乘以lambda，更新到global_model)  

## 运行项目
启动联邦学习训练，请执行：
```bash
python main.py -c ./utils/conf.json
```
或直接运行：
```bash
main.py
```

## 模型管理
- **模型命名**：记得在改变模型时 我喜欢把所需要引用的直接改名 model_changed_name ，就不用在其他文件中进行修改  
- **迁移学习权重**：进行迁移学习时，需要导入之前训练好的权重 `weight/tmp.pt`。

## 多模态输入参考分支 MML
