
from models import  model_changed_name as model  
import torch

from tqdm import tqdm
import sys
import torch.nn as nn
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.manifold import TSNE

class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf 
		
		self.global_model =  model().cuda() if torch.cuda.is_available() else model()
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		print(self.global_model)
		print(len(eval_dataset))
		self.eval_dataset=eval_dataset
		self.len_val=len(eval_dataset)

	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				

	def model_eval(self,epoch,best_acc,valtimelist,Valid_Accuracy_list):

		self.global_model.eval()
		with torch.no_grad():   #不加的话内存不够
			t2=time.perf_counter()
			val_bar = tqdm(self.eval_loader, file=sys.stdout)
			XX_list = [] #含XX YY都是下载去获得tsne
			YY_list = []
			acc,val_loss,count=0,0,0
			for stp, (val_images, val_params, val_labels) in enumerate(val_bar):
				if torch.cuda.is_available():
					al_images = val_images.cuda()
					val_params = val_params.cuda()
					val_labels = val_labels.cuda()
                
				outputs = self.global_model(val_images, val_params)
				predict_y = torch.max(outputs, dim=1)[1]
				acc += torch.eq(predict_y, val_labels).sum().item()
				loss_function = nn.CrossEntropyLoss()
				loss = loss_function(outputs, val_labels)
				val_loss += loss.item()
				count += 1
				if count == 1:
					X = outputs
					Y = val_labels
				else:
					X = torch.cat((X, outputs), dim=0)
					Y = torch.cat((Y, val_labels), dim=0)
                
				XX_list.append(outputs.cpu().detach().numpy())
				YY_list.append(val_labels.cpu().numpy())
			valtime=time.perf_counter()-t2
			XX = np.concatenate(XX_list, axis=0)  # Concatenate the X arrays along the rows
			YY = np.concatenate(YY_list, axis=0)  # Concatenate the Y arrays along the rows
	
			np.save('X.npy', XX)
			np.save('Y.npy', YY)
			os.rename('X.npy', f'X_{epoch}.npy')
			os.rename('Y.npy', f'Y_{epoch}.npy')
	
			print(f"valtime={valtime}")
			valtimelist.append(valtime)
			val_accurate = acc / self.len_val
			print(f"val_accurate={val_accurate}")
			Valid_Accuracy_list.append(val_accurate)
			# Val_loss = val_loss /stp
			# Val_loss_list.append(Val_loss)
	
	
	
	
	
			# # print(f'done->> batch_size {batch_size} lr {lr} p {p} weight_decay {weight_decay} step_size {step_size} gamma {gamma} best_epoch {best_epoch} best_acc {best_acc}')
			# # with open('logger.txt','a') as f:
			# #     f.write(f'one section done! delete batch_size={batch_size} lr={lr} p={p} weight_decay={weight_decay} step_size{step_size}gamma={gamma} best_epoch={best_epoch} best_acc={best_acc} \n')
		# # # #------------------------------------------------------------------------------------------------------------------------
	
		# # #------------------------------------------------------------------------------------------------------------------------
	
	
			def plot_t_sne(preds_result, labels):
				import matplotlib.pyplot as plt
				# 绘制t-sne聚类图结果
				# using t-SNE to show the result
				print("start t-sne! epoch:{}".format(epoch))
				tsne = manifold.TSNE(n_components=2, init="pca")  # random_state=501
				#   TSNE是一种降维算法，用于将高维数据映射到低维空间。本例中，n_components表示将高维数据降维到二维，init="pca"表示使用PCA作为初始化方法。
				best_preds = preds_result.cpu().detach().numpy()
				# X_tsne = tsne.fit_transform(best_preds)
	
				tsne = TSNE(perplexity=1)  #perplexity must be less than n_samples
				X_tsne = tsne.fit_transform(best_preds)
	
				x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
				encoder_result1 = ((X_tsne - x_min) / (x_max - x_min))
				#对tsne降维后的数据X_tsne进行归一化，先求出X_tsne每一列的最大值和最小值，
				# 然后计算每个值与最小值之差除以最大值和最小值之差，得到归一化以后的结果。
	
				fig = plt.figure(2)
				idx_1 = (labels == 0)
				p1 = plt.scatter(encoder_result1[idx_1, 0], encoder_result1[idx_1, 1], marker='x', color='m',
								label='NM', s=50)
				# 这段代码的功能是绘制一个散点图，标签为OM，图形颜色为紫色，x轴坐标为encoder_result1第一列，y轴坐标为encoder_result1第二列，每个散点的大小为50。其中labels == 0代表只绘制标签为0的点，idx_1为布尔型数组，表示标签为0的点的索引，
				# 因此encoder_result1[idx_1, 0]表示只绘制标签为0的点的x坐标，encoder_result1[idx_1, 1]表示只绘制标签为0的点的y坐标。
				idx_2 = (labels == 1)
				p2 = plt.scatter(encoder_result1[idx_2, 0], encoder_result1[idx_2, 1], marker='o', color='r',
								label='HM', s=50)
				idx_3 = (labels == 2)
				p2 = plt.scatter(encoder_result1[idx_3, 0], encoder_result1[idx_3, 1], marker='+', color='c',
								label='LM', s=50)
	
				plt.legend(loc='upper right')
				plt.xlabel("First component", fontsize=10)
				plt.ylabel("Second component",  fontsize=10)
				plt.grid(ls='--')
				# plt.savefig(
				#     './picture/t-SNE_of_AlexNet.svg')
				plt.show()
			try:
				plot_t_sne(X, Y)
			except Exception as e:
				print("An error occurred:", e)
	
	
	
	        # 保存全局模型的权重
			save_path = os.path.join(os.getcwd(), 'weight', 'tmp.pt')
			torch.save(self.global_model.state_dict(), save_path)
	        
		 	# # # --------------------------------------------------------------------------------------------------------------------------------
			import itertools
			from sklearn.metrics import confusion_matrix 
			import matplotlib.pyplot as plt
			def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
				if normalize:
					cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
					print("Normalized confusion matrix")
				else:
					print('Confusion matrix, without normalization')
				print(cm)
				plt.imshow(cm, interpolation='nearest', cmap=cmap)
				plt.title(title)
				plt.colorbar()
				tick_marks = np.arange(len(classes))
				plt.xticks(tick_marks, classes, rotation=45)
				plt.yticks(tick_marks, classes)
			
				fmt = '.2f' if normalize else 'd'
				thresh = cm.max() / 2.
				for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
					plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
			
			
				plt.tight_layout()
				plt.ylabel('True label')
				plt.xlabel('Predicted label')

			def get_all_preds(model, loader):
				all_preds = torch.tensor([])  # Empty tensor to accumulate predictions
				with torch.no_grad():  # No gradient needed for inference mode
					for batch in loader:
						images, params, labels = batch  # Unpack the batch; ensure loader provides images, params, and labels
						if torch.cuda.is_available():
							images = images.cuda()
							params = params.cuda()  # Move parameters to GPU if available
							labels = labels.cuda()
						
						preds = model(images, params).cpu()  # Make predictions using both images and params
						all_preds = torch.cat((all_preds, preds), dim=0)  # Concatenate the predictions
						
				return all_preds
			
			# #验证标签和验证预测的图------------------------------------------------------------------------------
			self.global_model.load_state_dict(torch.load('./weight/tmp.pt'))   #默认应该本次训练的net
			val_preds=get_all_preds(self.global_model,self.eval_loader)
			valcm=confusion_matrix(self.eval_dataset.targets,val_preds.argmax(dim=1))
			names=('0?','1?','2?')
			plt.figure(figsize=(10,10))
			plot_confusion_matrix(valcm,names)  #true
			print('done_matrix')#不知道为什么最后显示一个矩阵，不影响
