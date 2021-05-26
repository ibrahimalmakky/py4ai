import os
import glob
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

class MNIST(Dataset):

    IMG_EXT = ".png"

    TRAIN_DIR = "training"
    TEST_DIR = "testing"

    def __init__(self, path, train=True, transform=None) -> None:
        super().__init__()

        classes = [str(x) for x in range(0,10)]
        self.transform = transform

        if train:
            data_path = os.path.join(path, self.TRAIN_DIR)
        else:
            data_path = os.path.join(path, self.TEST_DIR)

        self.dataset = {"inputs":[], "targets":[]}

        for dt_class in classes:
            class_path = os.path.join(data_path, dt_class)
            class_files = glob.glob(os.path.join(class_path, "*"+self.IMG_EXT))
            
            for class_file in class_files:
                self.dataset["inputs"].append(class_file)
                self.dataset["targets"].append(int(dt_class))

        print(len(self.dataset["inputs"]))
        print(len(self.dataset["targets"]))


    def __len__(self):
        return len(self.dataset["targets"])

    def __getitem__(self, index):
        img_path = self.dataset["inputs"][index]
        target = self.dataset["targets"][index]

        img = Image.open(img_path)
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

if __name__ == "__main__":
    path = "/path/to/dataset"
    train_transforms = transforms.Compose([transforms.RandomRotation((0,20)), 
                                           transforms.RandomHorizontalFlip(p=0.5)])
    mnist = MNIST(path, transform=train_transforms)

    weights = [1/x for x in range(2, 112, 10)]
    print(weights)
    sample_weights = [weights[x] for x in mnist.dataset["targets"]]
    sampler = WeightedRandomSampler(sample_weights, num_samples=10)
    
    trainloader = torch.utils.data.DataLoader(mnist, batch_size=10, num_workers=4, sampler=sampler)
    for batch_num, data in enumerate(trainloader):
        print(data[1])
