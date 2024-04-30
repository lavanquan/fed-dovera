from typing import List 
from abc import ABC, abstractmethod
import copy
import torch 

class ServerBase(ABC):
    def __init__(self, model, test_loader):
        # super().__init__()
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_loader = test_loader
        
    @abstractmethod
    def model_eval(self):
        # model = self.model
        # model = model.to(self.device)

        # criterion = torch.nn.MSELoss()
        # test_loss = 0.0
        
        # with torch.no_grad():
        #     for inputs, targets in test_loader:
        #         targets = targets.reshape(-1, 1)
        #         inputs, targets = inputs.cuda(), targets.cuda()
        #         outputs = model(inputs)
        #         test_loss += criterion(outputs, targets).item()
        # avg_test_loss = test_loss / len(test_loader)
        # print(f'val_loss: {avg_test_loss}')
        # return avg_test_loss
        pass
    
    def zero_grad(self, model: torch.nn.Module) -> float:
        """
        zero gradient of a given model
        input: 
            + model (torch.nn.Module): the given model
        output:
            + check_sum (float): sum of all gradient values which should be 0
        """
        check_sum = 0.0
        # check_sum = model.to(model)
        for param in model.parameters():
            param.grad = torch.zeros_like(param.data)
            # check_sum += param.grad.sum()
        return check_sum

    def zero_param(self, model: torch.nn.Module, layer_require_updated=None) -> float:
        """
        zero weight of server before aggregate
        inputs:
            + model (torch.nn.Module): input model
        outputs:
            + check_sum (float): sum of all weight values which should be 0
        """
        check_sum = 0.0
        for name, param in model.named_parameters():
            if layer_require_updated is None:
                param.data = torch.zeros_like(param.data)
                check_sum += param.data.sum()
            else:
                if any(item in name for item in layer_require_updated):
                    # print("CUU TUI SET ZERO: ", name)
                    param.data = torch.zeros_like(param.data)
                    check_sum += param.data.sum()
        return check_sum

    def distribute_model(self, user_list: List, distributed_layers=None):
        # Make sure that model is on the same device
        self.model = self.model.to(self.device)

        for user in user_list:
            if distributed_layers is None:
                user.model = copy.deepcopy(self.model)
            else:
                for server_param, user_param in zip(self.model.named_parameters(), user.model.named_parameters()):
                    if any(item in server_param[0] for item in distributed_layers):
                        # print("CUU TUI DISTRIBUTE: ", server_param[0], user_param[0])
                        user_param[1].data = torch.zeros_like(user_param[1].data).to(self.device)
                        user_param[1].data += server_param[1].data
            
            # check_sum_grad = self.zero_grad(user.model)
            # assert check_sum_grad == 0, "check sum should be 0"
    
    def aggregate_weights(self, user_list: List, layer_require_updated=None):
        """
        Aggregate model using weights from user, updated weights is stored in the server's model
        Inputs:
            + user_list (list): list of users
        Outputs:
            + updated weights for self.model
        """
        # Make sure that model of all users is on the same device of server
        self.model = self.model.to(self.device)
        for user in user_list:
            user.model = user.model.to(self.device)
        
        # Aggregate gradient (FedAvg)
        ratio = 1.0/len(user_list)

        # Zero weight of model server before aggregation
        self.zero_param(self.model, layer_require_updated=layer_require_updated)

        # Aggregate weight with ratio
        for user in user_list:
            with torch.no_grad():
                for server_param, user_param in zip(self.model.named_parameters(), user.model.named_parameters()):
                    # print(server_param[0])
                    if layer_require_updated is None:
                        server_param[1].data += user_param[1].data * ratio
                    else:
                        if any(item in server_param[0] for item in layer_require_updated):
                            # print("CUU TUI AGGREGATE: ", server_param[0], user_param[0])
                            server_param[1].data += user_param[1].data * ratio