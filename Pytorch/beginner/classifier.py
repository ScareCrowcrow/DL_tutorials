import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as tud
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
CIFAR10 datasets: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
size: 3 * 32 * 32

steps:
1、Load and normalizing the CIFAR10 training and test datasets using torchvision
2、Define a Convolutional Neural Network
3、Define a loss function
4、Train the network on the training data
5、Test the network on the test data
"""
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 2
MODEL_SAVE_PATH = "./cifar_net.pth"
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# step1:
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = tud.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = tud.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# step2:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


net = Net()

# step3
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


# train
def train_fn(model, train_loader):
    # model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.
        for i, (images, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, total_loss/2000))
                total_loss = 0.
    print("Finished Trainning")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


def test_fn(model, test_loader):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    images, labels = next(iter(test_loader))
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % CLASSES[labels[j]] for j in range(4)))
    preds = model(images)
    _, preds = torch.max(preds, dim=1)
    print('Predicted: ', ' '.join('%5s' % CLASSES[preds[j]] for j in range(4)))
    correct = 0.
    total = 0.
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (CLASSES[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    train_loader, test_loader = load_data()
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    # print(labels)
    # imshow(torchvision.utils.make_grid(images))
    # print(" ".join("%5s" % CLASSES[labels[j]] for j in range(4)))
    # train_fn(net, train_loader)
    test_fn(net, test_loader)