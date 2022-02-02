
import os
import random
import torch
import numpy as np

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(9)

class attention_params(torch.nn.Module):
    def __init__(self, N):
        super(attention_params, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(N)/N)
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, idx):
        
        probs = self.softmax(self.alpha)
        
        return probs[idx]