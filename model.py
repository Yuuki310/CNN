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
            self.encoder(),
            self.one_d_block(N=128),
            nn.AdaptiveAvgPool2d(1)
        )
        

        # self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 10)

        # self.ave = nn.AdaptiveAvgPool2d(1)

    def one_d_block(self, N=128, L=32, B=128):
        block = nn.Sequential(
            # M : batchsize
            # bottleneck
            # [M, N, K] -> [M, B, K]
            nn.Conv1d(N, B, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(B),
            #[M, B, K] -> [M, B, K]
            nn.Conv1d(B, B, L, stride=L//2),
            nn.ReLU(),
            nn.BatchNorm1d(B),    
            #[M, B, K] -> [M, N, K] 
            nn.Conv1d(B, N, 1, bias=False),

        )
        return block
    
    def encoder(self):
        encoder = nn.Sequential(
            nn.Conv1d(1, self.N, kernel_size=self.L),
            nn.ReLU(),
            #nn.BatchNorm2d(self.N),            
        ) 
        return encoder
    
    def forward(self, x):
        x = self.network(x)
        return x
    
if __name__ == "__main__":
    model = Net()
    summary(model,(1,16000))