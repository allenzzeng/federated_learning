from models import model_changed_name as model
import models, torch, copy

class Client(object):
    def __init__(self, conf, global_model, train_dataset, client_id):
        self.conf = conf
        self.client_id = client_id
        self.local_model = model().cuda() if torch.cuda.is_available() else model()
        self.local_model.load_state_dict(global_model.state_dict())
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=conf["batch_size"], 
            shuffle=True
        )
        
    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        
        optimizer = torch.optim.Adam(self.local_model.parameters(), lr=self.conf['lr'])
        self.local_model.train()
        
        for e in range(self.conf["local_epochs"]):
            for batch_id, (images, params, labels) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    params = params.cuda()
                    labels = labels.cuda()
                
                optimizer.zero_grad()
                # Ensure the model accepts both images and params as inputs
                output = self.local_model(images, params)
                loss = torch.nn.functional.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
            
            print(f"client_id: {self.client_id} Epoch {e} done.")
        
        diff = {name: (data - model.state_dict()[name]) for name, data in self.local_model.state_dict().items()}
        
        return diff