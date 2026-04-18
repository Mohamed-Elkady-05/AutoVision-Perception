import torch as th
import torch.optim as opt
from torch.optim import lr_scheduler

class TrainerOpt:
    def __init__(self,model , optimizer='adam',lr=0.001,scheduler_name=None):
        self.model = model
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if optimizer == 'adam':
            self.optimizer = opt.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = opt.SGD(self.model.parameters(), lr=lr , momentum=0.9)
        elif optimizer == 'rmsprop':
            self.optimizer = opt.RMSprop(self.model.parameters(), lr=lr)
        elif optimizer == 'adagrad':
            self.optimizer = opt.Adagrad(self.model.parameters(), lr=lr)
        
        self.scheduler = None
        if scheduler_name == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
    
    def train_epoch(self,X_train,criterion):
        self.model.train()
        total_loss = 0

        for img, lab in X_train:
            img , lab = img.to(self.device), lab.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(img)
            loss = criterion(output, lab)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = (total_loss / len(X_train))
        return avg_loss
        
    def evaluate(self, loader, criterion):
      self.model.eval()
      val_loss = 0
      correct = 0
      total = 0
      with th.no_grad():
          for img, lab in loader:
              img, lab = img.to(self.device), lab.to(self.device)
              output = self.model(img)
              loss = criterion(output, lab)
              val_loss += loss.item()
              _, pred = th.max(output, 1)
              total += lab.size(0)
              correct += (pred == lab).sum().item()
      
      avg_loss = val_loss / len(loader)
      acc = 100 * correct / total
      
      if self.scheduler:
          self.scheduler.step(avg_loss)
          
      return avg_loss, acc