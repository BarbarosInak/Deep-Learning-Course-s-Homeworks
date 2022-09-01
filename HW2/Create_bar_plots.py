import numpy as np
import matplotlib.pyplot as plt


batch_sizes=[16,32,64]
names=["CNN","CNN2","CNN3","CNN4","CNN5"]

f = open("train_acc.txt","r")
train_acc=f.readlines()
f.close()

f = open("test_acc.txt","r")
test_acc=f.readlines()
f.close()

test_acc=[float(each[:-1]) for each in test_acc]
train_acc=[float(each[:-1]) for each in train_acc]

print(test_acc)
print(train_acc)

epoch16_train=train_acc[:5]
epoch32_train=train_acc[5:10]
epoch64_train=train_acc[10:]

epoch16_test=test_acc[:5]
epoch32_test=test_acc[5:10]
epoch64_test=test_acc[10:]

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar


# Set position of bar on X axis
br1 = np.arange(len(epoch16_train))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, epoch16_train, color='bisque', width=barWidth,
        edgecolor='grey', label='16')
plt.bar(br2, epoch32_train, color='skyblue', width=barWidth,
        edgecolor='grey', label='32')
plt.bar(br3, epoch64_train, color='mediumvioletred', width=barWidth,
        edgecolor='grey', label='64')

# Adding Xticks
plt.xlabel('Models', fontweight='bold', fontsize=15)
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(epoch16_train))],
           names)
plt.title("Train Accuracy")
plt.ylim(0.85,1.0)
plt.legend()
plt.savefig("train_acc.png")



# Set position of bar on X axis
br1 = np.arange(len(epoch16_test))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, epoch16_test, color='bisque', width=barWidth,
        edgecolor='grey')
plt.bar(br2, epoch32_test, color='skyblue', width=barWidth,
        edgecolor='grey')
plt.bar(br3, epoch64_test, color='mediumvioletred', width=barWidth,
        edgecolor='grey')
plt.ylim(0.85,1.0)
# Adding Xticks
plt.xlabel('Models', fontweight='bold', fontsize=15)
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(epoch16_test))],
           names)
plt.title("Test Accuracy")

plt.legend(["16","32","64"])
plt.savefig("test_acc.png")
