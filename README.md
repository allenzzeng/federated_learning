# Federated Learning

This project implements federated transfer learning with the ability to customize the backbone network.

The code is inspired by:
- [WZMIAOMIAO's deep-learning-for-image-processing repository](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master)
- [DingXiaoH's RepVGG repository](https://github.com/DingXiaoH/RepVGG)
- [FederatedAI's Practicing-Federated-Learning repository](https://github.com/FederatedAI/Practicing-Federated-Learning/tree/main/chapter03_Python_image_classification)

This code specifically focuses on applying federated learning to custom networks, custom datasets, federated transfer learning design, with the caveat that actual reproduction may encounter issues due to package versions, which may require bug fixes.

1. The folders under `./data/own_train/` contain all client datasets. `data/own_val/photo` is the validation dataset, `data/own_val/train` is the transfer learning training set, and `data/own_val/val` is the transfer learning validation set.

2. In the `weights` folder, parameters include:
   - `batch_size`: Batch size
   - `local_epochs`: Number of training epochs per client
   - `global_epochs`: Number of global training epochs
   - `lr`: Learning rate for clients
   - `momentum`: Momentum (used Adam)
   - `lambda`: Parameter used for client updates

3. To start federated learning training, run `main.py` or `python main.py -c ./utils/conf.json`.

4. Remember to rename directly the models you need to reference, e.g., `model_changed_name`, to avoid modifications in other files when changing models.

5. For transfer learning, import previously trained weights from `weight/tmp.pt`.


# federated_learning
联邦迁移学习代码，可以自定义backbone
本项目代码参考 https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master 和  https://github.com/DingXiaoH/RepVGG 和 https://github.com/FederatedAI/Practicing-Federated-Learning/tree/main/chapter03_Python_image_classification  
本代码仅提供联邦学习应用于自定义网络，自定义数据集，联邦迁移学习设计思路，实际复现可能因为各种包的版本出现问题，自行修复bug  
1 ./data/own_train/下面的的文件夹是所有的客户端数据集  data/own_val/photo是验证数据集   data/own_val/train是迁移学习训练集 data/own_val/val是迁移学习验证集  
2 weight文件夹中 batch_size 批次大小  local_epochs每个客户端训练的次数  global_epochs全局训练的次数 lr（客户端学习率），momentum（momentum 是优化算法中的一个超参数，主要用于改善梯度下降的收敛性能，特别是在处理非凸优化问题时。它在随机梯度下降（Stochastic Gradient Descent, SGD）和其变种中经常被使用，我用Adam），lambda（客户端返回diff，all客户端累加更新到weight_accumulator，server在乘以lambda，更新到global_model)  
3 运行main.py文件或者 python main.py -c ./utils/conf.json ，开始联邦学习训练，  
4 记得在改变模型时 我喜欢把所需要引用的直接改名 model_changed_name ，就不用在其他文件中进行修改  
5 迁移学习需要导入之前 weight/tmp.pt 训练好的权重  
