import torch, os
from torch import nn

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(8288, 1024),
          nn.ReLU(),
          nn.Linear(1024, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 32),
          nn.ReLU(),
          nn.Linear(32, 2)
        )


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class Trainer():
    def __init__(self, model, optimizer, loss_func):
        torch.manual_seed(32)

        self.train_losses = []
        self.eval_losses = []
        
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_func
        self.epoch = 0

    def train(self, train_dataloader, test_dataloader, n_epochs, epoch_log, epoch_eval):
        print("Training...")

        self.model.train()
        for epoch in range(0, n_epochs):
            epoch_losses = []
            for i, batch in enumerate(train_dataloader):
                inputs, targets = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                batch_loss = loss.item()
                epoch_losses.append(batch_loss)
            
            self.epoch += 1
            epoch_loss = torch.mean(torch.tensor(epoch_losses))

            if epoch % epoch_log == 0:
                print("train loss at epoch", epoch, " is ", epoch_loss.item())

            if epoch % epoch_eval == 0:
                eval_loss = self.evaluate(test_dataloader)
                self.eval_losses.append(eval_loss)

            self.train_losses.append(epoch_loss)
                
    def evaluate(self, dataloader):
        print("Evaluating...")

        self.model.eval()
        losses = [] 
        n_correct = 0.0
        n_total = 0.0
        for i, batch in enumerate(dataloader):
            inputs, targets = batch
            
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
            
            losses.append(loss.item()) 
            
            _, predicted_classes = torch.max(outputs, 1)
            n_correct += (predicted_classes == targets).sum().item()
            n_total += targets.size(0)

        accuracy = 100 * (n_correct/n_total)
        
        loss = torch.mean(torch.tensor(losses))

        print("eval loss", loss.item())
        print("eval accuracy", accuracy) 

        return loss
