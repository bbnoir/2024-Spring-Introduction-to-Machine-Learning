# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
curve_data1 = np.genfromtxt('../results/dis_train_4.csv', delimiter=',', skip_header=1)
curve_data2 = np.genfromtxt('../results/dis_train_3.csv', delimiter=',', skip_header=1)

# %%
plt.figure(figsize=(6,6))
plt.title("Training curve")
plt.plot(curve_data1[:, 0], curve_data1[:, 1], label='4 classes', marker='o')
plt.plot(curve_data2[:, 0], curve_data2[:, 1], label='3 classes', marker='o')

# Add number on each dot
for i, (x, y) in enumerate(zip(curve_data1[:, 0], curve_data1[:, 1])):
    plt.text(x, y, f"{y:.3f}", ha='center', va='bottom')
for i, (x, y) in enumerate(zip(curve_data2[:, 0], curve_data2[:, 1])):
    plt.text(x, y, f"{y:.3f}", ha='center', va='bottom')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
