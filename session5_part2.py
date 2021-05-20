import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def imshow(img: torch.Tensor):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = F.relu(out)
        out = self.pool(out)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = self.pool(out)
        out = F.relu(out)
        out = out.view(-1, 16*5*5)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        # print(out.shape)
        out = self.sig(self.fc2(out))
        return out
        

def main():
    epochs = 10

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4)

    valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    data_iter = iter(trainloader)
    images, labels = data_iter.next()

    # print(images.shape)
    # print(labels.shape)

    # imshow(torchvision.utils.make_grid(images))
    net = MyCNN()
    # print(net(images))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(0, epochs):
        net.train()
        running_loss = 0.0
        for batch_num, data in enumerate(trainloader):
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_num % 500 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch+1, batch_num+1, running_loss/500))
                running_loss = 0.0

        net.eval()
        with torch.no_grad():
            labels = []
            preds = []
            for batch_num, data in enumerate(valloader):
                inputs, batch_labels = data[0], data[1]
                labels += batch_labels
                output = net(inputs)
                _, batch_preds = torch.max(output, 1)
                preds += batch_preds.cpu()
            acc = metrics.accuracy_score(labels, preds)
            print('validation acc: %.2f' % (acc))

if __name__ == "__main__":
    main()
