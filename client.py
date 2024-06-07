from models import model_changed_name as model
import models, torch, copy
class Client(object):
		
	def __init__(self, conf, global_model, train_dataset, client_id):
		self.conf = conf
		self.client_id = client_id
		self.local_model = model().cuda() if torch.cuda.is_available() else model()
		self.local_model.load_state_dict(global_model.state_dict())
		self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf["batch_size"], shuffle=True)
			
		
	def local_train(self, model):

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		#print(id(model))
		# optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
		# 							momentum=self.conf['momentum'])
		optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])
		#print(id(self.local_model))
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
			
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
			
				optimizer.step()
			# print("Epoch %d done."  % e)	
			print("client_id: {} Epoch {} done.".format(self.client_id, e))
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			#print(diff[name])
			
		return diff
		