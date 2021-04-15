import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#import pretty_errors


class PointNet(nn.Module):
  def __init__(self):
    super(PointNet, self).__init__()
    self.conv1 = nn.Conv1d(3, 64, 1)
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 1024, 1)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 40)

    self.bn1 = nn.BatchNorm1d(64)  # 归一化层
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)
    self.bn4 = nn.BatchNorm1d(512)
    self.bn5 = nn.BatchNorm1d(256)

    self.relu = nn.ReLU(inplace=True) # 激活函数
    self.dropout = nn.Dropout(p=0.1)  #防止过拟合，丢掉一些神经元

  def forward(self, x):
    # TODO: use functions in __init__ to build network
    x = self.relu(self.bn1(self.conv1(x)))  #64
    x = self.relu(self.bn2(self.conv2(x)))  #128
    x = self.relu(self.bn3(self.conv3(x)))  #1024

    x, index = torch.max(x, 2, keepdim=True)  
    x= x.view(-1,1024)

    x = self.relu(self.bn4(self.fc1(x))) #512
    x = self.relu(self.bn5(self.fc2(x))) #256
    x = self.fc3(x) #40

    return x


if __name__ == "__main__":
  net = PointNet()
  sim_data = torch.rand(3, 3, 10000)
  out = net(sim_data)
  print('gfn', out.size())