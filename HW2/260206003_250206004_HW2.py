import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

num_classes = 10
batch_size = 32
batch_sizes=[16,32,64]

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

#Initial CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        #Defining the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes) #Automatic Softmax
    
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

#Number of channels in second convolutional layer is increased
class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()

        # Defining the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Automatic Softmax

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

#Pooling size is increased at the second pooling layer
class CNN3(nn.Module):
    def __init__(self, num_classes):
        super(CNN3, self).__init__()

        # Defining the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Automatic Softmax

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

#Pooling size is decreased at the first pooling layer
class CNN4(nn.Module):
    def __init__(self, num_classes):
        super(CNN4, self).__init__()

        # Defining the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(2304, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Automatic Softmax

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

#An additional convolutional layer is added.
class CNN5(nn.Module):
    def __init__(self, num_classes):
        super(CNN5, self).__init__()

        # Defining the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Automatic Softmax

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = CNN(num_classes)
model2= CNN2(num_classes)
model3= CNN3(num_classes)
model4= CNN4(num_classes)
model5= CNN5(num_classes)

models=[model1,model2,model3,model4,model5]

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
names=["CNN","CNN2","CNN3","CNN4","CNN5"]

#Nested loops for trying 15 different configurations
for i in batch_sizes:
    for model in models:
        print("batch size:",i,"\nmodel:",names[models.index(model)])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size = i,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size = i,
                                                   shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device:",device)

        model.to(device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        train_losses, test_losses = Batch_GD(model, criterion, optimizer, train_loader, test_loader, epochs=10)

        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.legend()
        plt.show()

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
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

        test_acc = n_correct / n_total
        print('\n')
        print('-------------------------------------------------------')
        print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
        print('-------------------------------------------------------')
        print('\n')

        train_acc_save.append(str(train_acc)+"\n")
        test_acc_save.append(str(test_acc)+"\n")


f=open("train_acc.txt","w")
f.writelines(train_acc_save)
f.close()

f=open("test_acc.txt","w")
f.writelines(test_acc_save)
f.close()

model=CNN4(num_classes)

print("batch size:",64,"\nmodel:","CNN2")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size = 64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size = 64,
                                           shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:",device)

model.to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses, test_losses = Batch_GD(model, criterion, optimizer, train_loader, test_loader, epochs=10)

model.eval()

def plot_confusion_matrix(cm, classes):
  print("\nConfusion Matrix\n")
  fmt = 'd'
  print(cm)
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Confusion Matrix')
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  thresh = cm.max() / 2

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i,j], fmt), horizontalalignment="center", color= 'white' if cm[i,j] > thresh else 'black')

  plt.tight_layout()
  plt.xlabel('True label')
  plt.ylabel('Predicted label')
  plt.show()

x_test = test_dataset.data
y_test = np.array(test_dataset.labels)
pred_test = np.array([])

for inputs, targets in test_loader:
  inputs, targets = inputs.to(device), targets.to(device)
  outputs = model(inputs)
  _, predictions = torch.max(outputs, 1)
  pred_test = np.concatenate((pred_test, predictions.cpu().numpy()))

cm = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(cm, list(range(num_classes)))