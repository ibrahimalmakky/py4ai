import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def imshow(img: torch.Tensor):
    # img = img / 2 + 0.5     # unnormalize
    img = img.numpy()
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers for the network
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        # print(out.shape)
        out = self.pool(out)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = out.view(-1, 16*10*10)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        # print(out.shape)
        out = F.relu(self.fc2(out))
        # print(out.shape)
        out = F.relu(self.fc3(out))
        # print(out.shape)
        return out

def main():
    epochs = 10
    val_freq = 1

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    summar_writer = tb.SummaryWriter("./runs/")
    summar_writer.add_text("Test", "Hello World!")
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)

    validation_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    data_iter = iter(trainloader)
    images, labels = data_iter.next()

    print(images.shape, labels.shape)

    # show images
    # imshow(torchvision.utils.make_grid(images))

    net = Network()
    net.to(device)
    output = net(images)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # get the weights from a certain layer
    # print(net.fc1.weight.detach().shape)
    # print(output)
    for epoch in range(0, epochs):
        net.train()
        running_loss = 0.0
        for batch_num, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(loss.item())

            running_loss += loss.item()
            if batch_num % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_num + 1, running_loss / 2000))
                running_loss = 0.0

        net.eval()
        with torch.no_grad():
            labels = []
            preds = []
            for _, val_data in enumerate(val_loader, 0):
                inputs, batch_labels = val_data
                labels += batch_labels

                output = net(inputs)

                _, batch_pred = torch.max(output, 1)

                preds += batch_pred.cpu()
        
        acc = metrics.accuracy_score(labels, preds)
        summar_writer.add_scalar("Accuracy", acc, epoch)
        print(acc)

if __name__ == "__main__":
    main()
