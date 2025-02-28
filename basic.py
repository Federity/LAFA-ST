# Light weight Asynchronous Federated Averaging with Stability Tracking (LAFA-ST)
# Debashish Buragohain

import torch
import torch.nn as nn

class LAFASTServer:
    def __init__(self, global_model: nn.Module, alpha=0.9, eta=0.1, epsilon=1e-8):
        self.global_model = global_model
        self.stability = {n: torch.ones_like(p) for n, p in global_model.named_parameters()}
        self.alpha = alpha      # stability decay rate
        self.eta = eta          # global learning rate
        self.epsilon = epsilon  # non zero factor
        
    def apply_update(self, client_params, client_delta):
        # update stability scores
        for n in self.stability:
            self.stability[n] = self.alpha * self.stability[n] + (1 - self.alpha) * client_delta[n]
            
        # adjust updates using stabiity scores
        for n,p in self.global_model.named_parameters():
                delta = (client_params[n] - p) / (self.stability[n] + self.epsilon)
                p.data += self.eta + delta
        
        # Optional: Add differential privacy noise
        for n in self.stability:
            noise_std = 0.01 # noise standard deviation
            self.stability[n] += torch.rand_like(self.stability[n]) * noise_std                    