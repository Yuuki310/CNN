import torch.nn as nn
from torchsummary import summary


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(4)

        self.N = kwargs.get("N") if kwargs.get("N") else 128 
        self.L = kwargs.get("L") if kwargs.get("L") else 32 # kernel_size
        
        self.network = nn.Sequential(
            self.encoder(256,32),
            # self.one_d_block(N=128),
            nn.AdaptiveAvgPool2d(1)
        )
        

        # self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 10)

        # self.ave = nn.AdaptiveAvgPool2d(1)
    def tcn(self):
        network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1,kernel_size), (1,1), padding=0, 
                           dilation=1, groups=1, bias=True, padding_mode='zeros')            
        )       
    def encoder(self,N,L):
        encoder = nn.Sequential(
            nn.Conv2d(1, N, (1,L),(1,L//2)),
            nn.ReLU(),
            nn.BatchNorm2d(N),            
        ) 
        return encoder
    
    def forward(self, x):
        x = self.network(x)
        return x
    
if __name__ == "__main__":
    model = Net()
    summary(model,(1,1,24000))