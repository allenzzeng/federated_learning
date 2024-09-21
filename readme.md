[中文](readme_zh.md)
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

MultiModal Learning Input in Branch 'MML'
