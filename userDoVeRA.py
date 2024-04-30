from userbase import UserBase
import torch

class UserMLP(UserBase):
    def __init__(self, train_loader, model, user_id, local_epochs):
        super().__init__(train_loader, model, user_id, local_epochs)

    def user_train(self):
        # Get model
        model = self.model

        # Use all available gpus for training
        num_gpus = torch.cuda.device_count()
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)

        # Define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train model
        for epoch in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.view(-1, 28*28).to(torch.device('cuda'))
                targets = targets.to(torch.device('cuda'))
                logits = model(inputs)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # optimizer.zero_grad()
                # targets = targets.reshape(-1, 1)
                # inputs, targets = inputs.cuda(), targets.cuda()
                # outputs = model(inputs)
                # loss = criterion(outputs, targets)
                # loss.backward()
                # optimizer.step()
        loss = loss.item()
        # print(f'User {self.id} Train Loss: {loss:.3f}')
        self.model = model
        return loss

class UserMLPMulOutput(UserBase):
    def __init__(self, train_loader, model, user_id, local_epochs):
        super().__init__(train_loader, model, user_id, local_epochs)

    def user_train(self):
        # Get model
        model = self.model

        # Use all available gpus for training
        num_gpus = torch.cuda.device_count()
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)), dim=0)

        # Define the loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Train model
        for epoch in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                optimizer.zero_grad()
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        loss = loss.item()
        # print(f'User {self.id} Train Loss: {loss:.3f}')
        self.model = model
        return loss