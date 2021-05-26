from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from mnist import MNIST

transform = transforms.Compose([transforms.ToTensor()])

mnist_dt = MNIST("C:\\Users\\ibrahim.almakky\\OneDrive - Mohamed Bin Zayed University of Artificial Intelligence\\Documents\\Courses\\PythonForAI_MAR21\\Code Examples\\data\\MNIST", 
                    train=True, 
                    transforms=transform)

weights = [1/x for x in range(2, 113, 10)]
print(weights)
sample_weights = [weights[x] for x in mnist_dt.dataset["targets"]]
print(sample_weights[0:10])

train_loader = DataLoader(mnist_dt, 
                          batch_size=10,  
                          sampler=WeightedRandomSampler(weights=sample_weights, num_samples=10, replacement=True))

for i, data in enumerate(train_loader):
    print(data[1])
    exit()
