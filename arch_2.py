import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        self.end = nn.Sequential(
            nn.LeakyReLU(inplace = True),
            nn.LazyLinear(32),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace = True),
            nn.LazyLinear(10))
        
    #same block being concatenated
    def forward(self, X, num_blocks = 1):
        
        h = X.view(X.shape[0], -1)
        h = self.end(h)
        h = nn.functional.log_softmax(h, dim=1)
        return h