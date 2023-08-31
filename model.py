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
            self.encoder(64,4),
            self.temporalblock(64,32,d=2),
            self.temporalblock(32,16,d=4),
            self.temporalblock(16,8,d=8),
            nn.AdaptiveAvgPool3d(1),
        )

    def temporalblock(sekf, N, B, d=1, dropout=0.3):
        block = nn.Sequential(
            nn.Conv2d(N, B, (1,16), stride=(1,8), bias=False, dilation=d),
            nn.BatchNorm2d(B),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        return block

    def one_d_block(self, N=256, L=8, B=128, d=1):
        block = nn.Sequential(
            # M : batchsize
            # bottleneck
            # [M, N, K] -> [M, B, K]
            nn.Conv2d(N, B, (1,1), stride=(1,1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(B),
            #[M, B, K] -> [M, B, K]
            nn.Conv2d(B, B, (1,L), stride=(1,1), dilation=d),
            nn.ReLU(),
            nn.BatchNorm2d(B),    
            #[M, B, K] -> [M, N, K] 
            nn.Conv2d(B, N, (1,1), stride=(1,1), bias=False),

        )
        return block
    
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