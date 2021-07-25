import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.start = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1), # 32-2 = 30
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)) # 32/2 = 16
        
        self.middle = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1), # 16
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace = True))
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        
        self.end = nn.Sequential(
            nn.LeakyReLU(inplace = True),
            nn.LazyLinear(32),
            nn.LeakyReLU(inplace = True),
            nn.LazyLinear(10))
        
    #same block being concatenated
    def forward(self, X, num_blocks = 0):
        h = self.start(X)
        for i in range(num_blocks):
            h= self.middle(h)
        h = X.view(X.shape[0], -1)
        h = self.end(h)
        h = nn.functional.log_softmax(h, dim=1)
        return h