from PIL import Image
import copy
import os
from pathlib import Path
from matplotlib import pyplot as plt
import math
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision.transforms import transforms
from torchvision import transforms, datasets, models
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm


class MyVGG19_BN(nn.Module):
    def __init__(self, num_classes=10, dropout: float = 0.5):
        super(MyVGG19_BN, self).__init__()
        pretrain_model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        pretrain_model.classifier = nn.Sequential()  # remove last layer
        self.features = pretrain_model
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def harmonic(train_acc, test_acc):
    harmonic = 2 * train_acc * test_acc / (train_acc + test_acc)
    return harmonic

def main(num_epoches=50):
    num_epoches = 50

    # Create dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 16

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Runing on: ", device)

    model = MyVGG19_BN(num_classes=10)
    summary(model, (3, 32, 32))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    training_acc = []
    training_loss = []
    testing_acc = []
    testing_loss = []
    H = 0.0  # harmonic
    best_model = copy.deepcopy(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train model
    for epoch in range(num_epoches):
        model.train()
        print("\n", "*" * 25, "epoch {}".format(epoch + 1), "*" * 25)
        running_loss = 0.0
        num_correct = 0.0

        train_loop = tqdm(train_loader)
        for data in train_loop:
            img, label = data
            img, label = img.to(device), label.to(device)

            out = model(img)

            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate acc & loss
            running_loss += loss.item() * label.size(0)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(probs, dim=1)
            num_correct += (pred == label).sum().item()

            train_loop.set_description(f"Train Epoch [{epoch + 1}/{num_epoches}]")

        train_acc = num_correct / len(train_dataset)
        train_loss = running_loss / len(train_dataset)
        training_acc.append(train_acc)
        training_loss.append(train_loss)
        print("Train --> Loss: {:.6f}, Acc: {:.6f}".format(train_loss, train_acc))

        # 用 testing dataset 來評估 model
        model.eval()
        eval_loss = 0
        num_correct = 0

        test_loop = tqdm(test_loader)
        for data in test_loop:
            img, label = data
            img, label = img.to(device).detach(), label.to(device).detach()

            out = model(img)
            loss = loss_func(out, label)
            eval_loss += loss.item() * label.size(0)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(probs, dim=1)
            num_correct += (pred == label).sum().item()

            test_loop.set_description(f"Test Epoch [{epoch + 1}/{num_epoches}]")

        test_acc = num_correct / len(test_dataset)
        test_loss = eval_loss / len(test_dataset)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        print("Test -->  Loss: {:.6f}, Acc: {:.6f}".format(test_loss, test_acc))

        # 紀錄 Harmonic 最高的 model 為 best model

        current_H = harmonic(train_acc, test_acc)
        if current_H > H:
            best_model = copy.deepcopy(model)
            H = current_H
        print("Current Harmonic : {:.6f}, Best Harmonic : {:.6f}".format(current_H, H))


    # save best model
    if os.path.exists("./models") == False:
        os.mkdir("./models")
    torch.save(best_model.state_dict(), "./models/VGG19_bn_cifar10_state_dict.pth")
    torch.save(best_model, "./models/VGG19_bn_cifar10.pth")



if __name__ == "__main__":
    main()