import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(51153)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16*5*5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = self.conv2(x)
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(x), 2)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def train_fn(model, inp):
    output = model(inp)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss: ", loss.item())


if __name__ == "__main__":
    net = Net()
    inp = torch.randn(1, 1, 32, 32)
    epoch = 2
    for e in range(epoch):
        train_fn(net, inp)



