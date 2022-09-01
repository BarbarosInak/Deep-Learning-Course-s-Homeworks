import matplotlib.pyplot as plt
import numpy as np

f=open("train_val_acc.txt","r")
train_val_acc=f.readlines()
f.close()

f=open("train_val_acc_1.txt","r")
train_val_acc_n=f.readlines()
f.close()

[train_val_acc.append(each) for each in train_val_acc_n]

f=open("train_val_acc_2.txt","r")
train_val_acc_n=f.readlines()
f.close()

[train_val_acc.append(each) for each in train_val_acc_n]

train_val_acc=[each[:-1] for each in train_val_acc]

train_val_acc.sort()


for i in range(len(train_val_acc)):
    train_val_acc[i]=train_val_acc[i].split(",")
    train_val_acc[i]=[float(each) for each in train_val_acc[i]]

print(train_val_acc)

train_acc=[]
val_acc=[]
names=[]
for i in range(0,len(train_val_acc),5):
    sum=0
    sum2=0
    for k in range(5):

        sum+=train_val_acc[i+k][3]
        sum2+=train_val_acc[i+k][4]
    train_acc.append(sum/5)
    val_acc.append(sum2/5)
    names.append(str(int(train_val_acc[i + k][0])) + "_" + str(int(train_val_acc[i + k][1])) + "_" + str(int(train_val_acc[i + k][2])))

print(names)
print(train_acc)
print(val_acc)

plt.bar(names,train_acc)
plt.xlabel("nb_channels,kernel_size,nb_blocks")
plt.ylabel("Accuracy")
plt.title("Train Accuracy Scores")
plt.ylim(0.78,1.0)
plt.show()

plt.bar(names,val_acc)
plt.xlabel("nb_channels,kernel_size,nb_blocks")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Scores")
plt.ylim(0.78,1.0)
plt.show()

f=open("train_test_loss.txt","r")
train_test_loss=f.readlines()
f.close()

train_test_loss=[each.split(",") for each in train_test_loss]

a=train_test_loss[1:3]

b=[]
b.append(a[0][1][1:-1])
b.append(a[1][0][:-2])

b=[each.split(" ") for each in b]

print(b)

c=[]

for i in range(2):
    for j in range(len(b[i])):
        if i==1 and j==0:
            pass
        else:
            c.append(float(b[i][j]))

print(c)

plt.plot(range(1,11),c)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Test Loss Graph")
plt.show()