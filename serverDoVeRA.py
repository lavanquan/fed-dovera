from serverbase import ServerBase
import torch

class ServerMLP(ServerBase):
    def __init__(self, model, test_loader):
        super().__init__(model, test_loader)

    def model_eval(self):
        model = self.model
        model = model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.view(-1, 28*28).to(torch.device('cuda'))
                targets = targets.to(torch.device('cuda'))

                # targets = targets.reshape(-1, 1)
                # inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        avg_test_loss = test_loss / len(self.test_loader)
        print(f'val_loss: {avg_test_loss}')
        return avg_test_loss
    
    def compute_accuracy(self):
        model = self.model
        model = model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0.0

        correct_pred, num_examples = 0, 0

        with torch.no_grad():
            for features, targets in self.test_loader:
                features = features.view(-1, 28*28).to(torch.device('cuda'))
                targets = targets.to(torch.device('cuda'))

                logits = model(features)
                _, predicted_labels = torch.max(logits, 1)
                num_examples += targets.size(0)
                correct_pred += (predicted_labels==targets).sum()

            return correct_pred.float()/num_examples*100

class ServerMLPMulOutput(ServerBase):
    def __init__(self, model, test_loader):
        super().__init__(model, test_loader)

    def model_eval(self):
        model = self.model
        model = model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
        avg_test_loss = test_loss / len(self.test_loader)
        print(f'val_loss: {avg_test_loss}')
        return avg_test_loss