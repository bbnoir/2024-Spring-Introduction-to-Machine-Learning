# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
data_train = np.genfromtxt('../HW2_training.csv', delimiter=',', skip_header=1)
print(data_train.shape)

# %%
plt.figure(figsize=(6,6))
plt.title("Data distribution")
cs = plt.scatter(data_train[:, 1], data_train[:, 2], c=data_train[:, 0], cmap='viridis', alpha=0.5)
plt.xlabel("Offensive")
plt.ylabel("Defensive")
plt.grid()
plt.legend(*cs.legend_elements(), loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
# trurn class 3 to class 0
data_train[data_train[:, 0] == 3.0, 0] = 0.0

# %%
y = np.genfromtxt('../results/gen_model.csv', delimiter=',', skip_header=1)

# %%
# plot coutourf
x1 = np.linspace(0, 100, 1000)
x2 = np.linspace(0, 100, 1000)
plt.figure(figsize=(6,6))
plt.title("Boundary plot for generative model")
cs = plt.contourf(x1, x2, y.reshape(1000, 1000).transpose(), levels=[-1, 0, 1, 2, 3], alpha=0.5)
plt.xlabel("Offensive")
plt.ylabel("Defensive")
plt.grid()
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
plt.legend(proxy, ["Team 0", "Team 1", "Team 2", "Team 3"], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
y = np.genfromtxt('../results/dis_model.csv', delimiter=',', skip_header=1)

# %%
# plot coutourf
x1 = np.linspace(0, 100, 1000)
x2 = np.linspace(0, 100, 1000)
plt.figure(figsize=(6,6))
plt.title("Boundary plot for discriminative model")
cs = plt.contourf(x1, x2, y.reshape(1000, 1000).transpose(), levels=[-1, 0, 1, 2, 3], alpha=0.5)
plt.xlabel("Offensive")
plt.ylabel("Defensive")
plt.grid()
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
plt.legend(proxy, ["Team 0", "Team 1", "Team 2", "Team 3"], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
y = np.genfromtxt('../results/gen_model2.csv', delimiter=',', skip_header=1)

# %%
# plot coutourf
x1 = np.linspace(0, 100, 1000)
x2 = np.linspace(0, 100, 1000)
plt.figure(figsize=(6,6))
plt.title("Boundary plot for generative model")
cs = plt.contourf(x1, x2, y.reshape(1000, 1000).transpose(), levels=[-1, 0, 1, 2, 3], alpha=0.5)
plt.xlabel("Offensive")
plt.ylabel("Defensive")
plt.grid()
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
plt.legend(proxy, ["Team 0", "Team 1", "Team 2"], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
y = np.genfromtxt('../results/dis_model2.csv', delimiter=',', skip_header=1)

# %%
# plot coutourf
x1 = np.linspace(0, 100, 1000)
x2 = np.linspace(0, 100, 1000)
plt.figure(figsize=(6,6))
plt.title("Boundary plot for discriminative model")
cs = plt.contourf(x1, x2, y.reshape(1000, 1000).transpose(), levels=[-1, 0, 1, 2, 3], alpha=0.5)
plt.xlabel("Offensive")
plt.ylabel("Defensive")
plt.grid()
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
plt.legend(proxy, ["Team 0", "Team 1", "Team 2"], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# %%
