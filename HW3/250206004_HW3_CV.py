import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class ResBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                            padding=((kernel_size-1)//2))
        
        #self.bn1 = nn.BatchNorm2d(nb_channels)
        #self.do1 = nn.Dropout2d(p=0.25)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                            padding=((kernel_size-1)//2))

        #self.bn2 = nn.BatchNorm2d(nb_channels)
        #self.do2 = nn.Dropout2d(p=0.25)

    def forward(self, x):
        y = self.conv1(x)
        #y = self.bn1(y)
        #y = self.do1(y)
        y = F.relu(y)
        y = self.conv2(y)
        #y = self.bn2(y)
        #y = self.do2(y)
        y = y + x
        y = F.relu(y)

        return y

class ResNet(nn.Module):
    def __init__(self, nb_channels, kernel_size, nb_blocks):
        super().__init__()

        self.conv0 = nn.Conv2d(3, nb_channels, kernel_size=1)

        self.resblocks = nn.Sequential(
            *(ResBlock(nb_channels, kernel_size) for _ in range(nb_blocks))
        )
        
        self.avg = nn.AvgPool2d(kernel_size=28)
        self.fc = nn.Linear(nb_channels, 10)
    
    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.resblocks(x)
        x = F.relu(self.avg(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def Batch_GD(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        model.train()
        train_loss = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f},Test Loss: {test_loss:.4f}')

    return train_losses, test_losses

train_acc_save=[]
test_acc_save=[]

train_dataset = torchvision.datasets.SVHN(
    root = '.',
    split = 'train',
    transform = transforms.ToTensor(),
    download = True
)

test_dataset = torchvision.datasets.SVHN(
    root = '.',
    split = 'test',
    transform = transforms.ToTensor(),
    download = True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",device)

k_folds=5

kfold=KFold(n_splits=k_folds,shuffle=True)

nb_blocks=7

kernel_size=5

nb_channels=12


for fold, (train_ids,val_ids) in enumerate(kfold.split(train_dataset)):

    print('--------------------------------')
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10, sampler=val_subsampler)


    model = ResNet(nb_channels=nb_channels, kernel_size=kernel_size, nb_blocks=nb_blocks)

    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses, test_losses = Batch_GD(model, criterion, optimizer, train_loader, val_loader, epochs=10)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Val Loss')
    plt.legend()

    model.eval()

    n_correct = 0.
    n_total = 0.
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    train_acc = n_correct / n_total

    n_correct = 0.
    n_total = 0.

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    test_acc = n_correct / n_total

    print('\n')
    print('-------------------------------------------------------')
    print(f"Train acc: {train_acc:.4f}, Val acc: {test_acc:.4f}")
    print('-------------------------------------------------------')
    print('\n')
    f=open("train_val_acc_do.txt","a+")
    f.writelines(str(nb_channels)+","+str(kernel_size)+","+str(nb_blocks)+","+str(train_acc)+","+str(test_acc)+"\n")
    f.close()

    f = open("train_val_loss_do.txt", "a+")
    f.writelines(
        str(nb_channels) + "," + str(kernel_size) + "," + str(nb_blocks) + "," + str(train_losses) + "," + str(
            test_losses) + "\n")
    f.close()

    plt.savefig(str(nb_channels) + "_" + str(kernel_size) + "_" + str(nb_blocks) + "_fold"+str(fold)+"train_val_loss.png")