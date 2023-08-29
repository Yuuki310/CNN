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
            self.one_d_block(128, self.L),
#            nn.Conv1d(128, 256, 32),
            nn.AdaptiveAvgPool2d(1)
        )
        # self.bottleneck1 = nn.Conv1d(1, 8,  kernel_size=1, stride=1, bias=False)
        # self.conv1 = nn.Conv1d(1, 16, kernel_size=L, stride=L // 2, bias=False)
        # self.conv2 = nn.Conv1d(16, 1, kernel_size=L, stride=L // 2 ,bias=False)
        # # self.conv3 = nn.Conv1d(32, 16, kernel_size=L, stride=L // 2, bias=False)
        # # self.conv4 = nn.Conv1d(16, 1, kernel_size=L, stride=L // 2 ,bias=False)
        

        # self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 10)

        # self.ave = nn.AdaptiveAvgPool2d(1)

    def one_d_block(self, d, L, ):
        block = nn.Sequential(
            nn.Conv1d(d, d//2, kernel_size=L, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(d//2),
            nn.Conv1d(d//2, d, L, stride=L//2),
            nn.ReLU(),
            nn.BatchNorm2d(d),    
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
        # x = self.bottleneck1(x)
        # x = self.relu(x)
        # x = self.pool(x)
        
        x = self.network(x)
        # x = self.conv3(x)
        # x = self.relu(x)
        # x = self.pool(x)
        
        # x = self.conv4(x)
        # x = self.relu(x)
        # x = self.pool(x)

        # x = self.conv3(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = x.view(x.size()[0], -1)
        return x
    
if __name__ == "__main__":
    model = Net()
    summary(model,(1,16000))