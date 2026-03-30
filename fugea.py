import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
from .utils import wrap_model, EnsembleModel

class FUGEA:
    def __init__(self, 
                 model_names, 
                 epsilon=16/255, 
                 alpha=1.6/255, 
                 T_prg=10, 
                 T_rce=5, 
                 beta=10, 
                 threshold=-0.3, 
                 s=10, 
                 num_neighbor=20, 
                 gamma=0.5, 
                 decay=1.0, 
                 targeted=False):
        
        self.epsilon, self.alpha = epsilon, alpha
        self.T_prg, self.T_rce = T_prg, T_rce
        self.beta, self.threshold = beta, threshold
        self.s, self.num_neighbor, self.gamma = s, num_neighbor, gamma
        self.decay, self.targeted = decay, targeted
        self.zeta = 3.0 * epsilon
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        loaded_models = []
        for name in model_names:
            m = models.__dict__[name](weights="DEFAULT") if name in models.__dict__ else timm.create_model(name, pretrained=True)
            loaded_models.append(wrap_model(m.eval().to(self.device)))
        
        self.model = EnsembleModel(loaded_models)
        self.num_model = len(loaded_models)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, data, label):
        data, label = data.to(self.device), label.to(self.device)
        if self.targeted: label = label[1]
        
        delta = torch.zeros_like(data).to(self.device)
        delta = (delta + 0.001 * torch.randn_like(data)).detach().requires_grad_(True)
        
        momentum_G = 0.

        for _ in range(self.T_rce):
            outputs = [m(delta + data) for m in self.model.models]
            losses = [self.loss_func(out, label) for out in outputs]
            grads = [torch.autograd.grad(l, delta, retain_graph=True)[0] for l in losses]
            
            weights = self._calculate_uw(losses, data, delta, grads, label)
            cos_mask = (self._calculate_gam(grads, data.size()) >= self.threshold).float()
            
            agg_out = (torch.stack(outputs) * weights.view(self.num_model, 1, 1)).sum(0)
            grad = torch.autograd.grad(self.loss_func(agg_out, label).sum(), delta)[0] * cos_mask
            momentum_G = self.decay * momentum_G + grad
            delta = self._update(delta, data, momentum_G, self.alpha * self.s)

        for _ in range(self.T_prg):
            snpg_grads = [self._snpg(data, delta, label, i) for i in range(self.num_model)]
            weights = torch.softmax(torch.stack([self.loss_func(m(delta + data), label) for m in self.model.models]), dim=0)
            
            agg_grad = sum(weights[i].item() * snpg_grads[i] for i in range(self.num_model))
            momentum_G = self.decay * momentum_G + agg_grad
            delta = self._update(delta, data, momentum_G, self.alpha)

        return delta.detach()

    def _snpg(self, data, delta, label, idx):
        avg_grad = 0
        for _ in range(self.num_neighbor):
            p = torch.zeros_like(delta).uniform_(-self.zeta, self.zeta).to(self.device)
            x_near = torch.clamp(data + delta + p, 0, 1)
            g1 = torch.autograd.grad(self.loss_func(self.model.models[idx](x_near), label).sum(), delta, retain_graph=True)[0]
            x_next = torch.clamp(x_near - self.alpha * g1.sign(), 0, 1)
            g2 = torch.autograd.grad(self.loss_func(self.model.models[idx](x_next), label).sum(), delta, retain_graph=True)[0]
            avg_grad += (1 - self.gamma) * g1 + self.gamma * g2
        return avg_grad / self.num_neighbor

    def _calculate_uw(self, losses, ori, delta, grads, label):
        u_w = torch.softmax(torch.stack(losses), dim=0)
        w = torch.zeros(self.num_model, device=self.device)
        for j in range(self.num_model):
            adv_j = torch.clamp(ori + torch.clamp(delta + grads[j].sign() * self.alpha, -self.epsilon, self.epsilon), 0, 1)
            loss_self_j = self.loss_func(self.model.models[j](adv_j), label)
            for i in range(self.num_model):
                if i != j: w[j] += (self.loss_func(self.model.models[i](adv_j), label) / loss_self_j) * self.beta
        return torch.softmax(torch.softmax(w, dim=0) * u_w, dim=0)

    def _calculate_gam(self, grads, sz):
        sim = nn.CosineSimilarity(dim=1)
        res = torch.zeros(self.num_model, sz[0], sz[2], sz[3], device=self.device)
        for i in range(self.num_model):
            temp_sim = sum(sim(F.normalize(grads[i], dim=1), F.normalize(grads[j], dim=1)) for j in range(self.num_model) if i != j)
            res[i] = temp_sim / (self.num_model - 1)
        return res.mean(0).view(sz[0], 1, sz[2], sz[3])

    def _update(self, delta, data, grad, alpha):
        delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        return (torch.clamp(data + delta, 0, 1) - data).detach().requires_grad_(True)

    def __call__(self, *args):
        return self.forward(*args)
