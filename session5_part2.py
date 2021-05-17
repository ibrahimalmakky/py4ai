import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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
        self.fc1 = nn.Linear(16*10*10, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv1(x)
        print(out.shape)
        out = F.relu(out)
        out = self.pool(out)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        out = self.pool(out)
        print(out.shape)
        out = F.relu(out)
        out = out.view(-1, 16*10*10)
        print(out.shape)
        out = F.relu(self.fc1(out))
        print(out.shape)
        out = self.sig(self.fc2(out))
        return out
        

def main():
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    data_iter = iter(trainloader)
    images, labels = data_iter.next()

    # print(images.shape)
    # print(labels.shape)

    # imshow(torchvision.utils.make_grid(images))
    net = MyCNN()
    print(net(images))


if __name__ == "__main__":
    main()
