# torch import
import torch
import torch.nn as nn


# Adversarial attack
class Attack(object):
    def __init__(self, old_model, new_model, alpha, loader, protos, device, epochs, 
                 x_min, x_max, target):
        self.old_model = old_model
        self.new_model = new_model
        self.alpha = alpha
        self.loader = loader
        self.device = device
        self.target = target
        self.epochs = epochs
        self.proto = protos[target]
        self.protos = protos
        self.protos_ = torch.stack(self.protos).cpu()
        self.x_min = x_min
        self.x_max = x_max

    def perturb(self, x, alpha, grad, x_min, x_max):
        x_prime = x - (alpha * grad / torch.norm(grad, keepdim=True))
        x_prime = torch.clamp(x_prime, x_min.to(self.device), x_max.to(self.device))
        return x_prime
    
    def run(self):
        p_data, p_label = [], []
    
        for data, label in self.loader:
            data, label = data.to(self.device), label.to(self.device)
            target = torch.tensor(self.target).expand(len(label)).to(self.device)
            
            for i in range(self.epochs):
                # Adversarial attack requires gradients w.r.t. the data
                data.requires_grad = True
                feats = self.old_model(data)["features"]
                L = nn.MSELoss()

                # in a targeted attack, we take the loss w.r.t. the target prototype
                loss = L(feats, self.proto.expand(len(label), self.proto.shape[0]).to(self.device))
            
                # zero out all existing gradients
                self.old_model.zero_grad()
                # calculate gradients
                loss.backward()
                data_grad = data.grad
            
                perturbed_data = self.perturb(data, self.alpha, data_grad, self.x_min, self.x_max)
                data = perturbed_data.clone().detach()

                # to select only those adversarial samples which are successfully misclassified as the target class
                if i == (self.epochs-1):
                    adv_output = self.old_model(perturbed_data)["features"].cpu()
                    d = torch.cdist(adv_output, self.protos_)
                    adv_pred = torch.argmin(d, dim=1)  # NCM classification
                    mask = adv_pred == target.cpu()
                    success = mask.float().sum()

                    if success > 0:
                        print('successful attacks for class ',self.target ,' -> ',success.item())
                        
                        p_data.append(perturbed_data[mask])
                        p_label.append(target[mask])
                    else:
                        return [], []

        return torch.cat(p_data, 0), torch.cat(p_label, 0)