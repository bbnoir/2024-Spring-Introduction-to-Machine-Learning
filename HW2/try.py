# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
seed = 1224
classes = [0, 1, 2, 3]

df = pd.read_csv('HW2_training.csv')
data = df.to_numpy()

print(data.shape)

# %%
t = data[:, 0]
x = data[:, 1:3]
#t = np.where(t == 4, 1, t)
#data[:, 3] = t

plt.figure(figsize=(6,6))
plt.title("Data plot")
for c in classes:
    x_c = np.squeeze(np.take(x, np.where(t == c), axis=0))
    plt.scatter(x_c[:, 0], x_c[:, 1], label = c)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Generative model

# %%
def mean_matrix(data):
    x = data[:, 1:3]
    t = data[:, 0]
    mean_matrix = np.zeros((0, 2))
    for c in classes:
        mean_matrix = np.vstack((mean_matrix, np.mean(np.squeeze(np.take(x, np.where(t == c), axis=0)), axis=0)))
    return mean_matrix

def weight(data):
    x = data[:, 1:3]
    return np.linalg.inv(np.cov(x.T)) @ mean_matrix(data).T

def bias(data):
    x = data[:, 1:3]
    t = data[:, 0]
    b =  -0.5 * np.sum(mean_matrix(data) @ np.linalg.inv(np.cov(x.T)) * mean_matrix(data), axis=1)
    for c in classes:
        b[c] += np.log(np.sum(t == c) / len(t))
    return b

def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

def predict_generative(x, w, b):
    return softmax(np.dot(x, w) + b)

def classify(y):
    return np.argmax(y, axis=1)

# %%
w = weight(data)
b = bias(data)

# %%
y = predict_generative(x, w, b)
result = classify(y)

acc = np.mean(result == t)
print(f"accuracy:{acc}")

# %%
x1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
x2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)

xx1, xx2 = np.meshgrid(x1, x2)
xxx = np.asarray([xx1.ravel(), xx2.ravel()]).T

y = predict_generative(xxx, w, b)
result = classify(y)

plt.figure(figsize=(6,6))
plt.title(f"Generative model decision boundary, acc={acc}")
for c in classes:
    xxx_c = np.squeeze(np.take(xxx, np.where(result == c), axis=0))
    plt.scatter(xxx_c[:, 0], xxx_c[:, 1], label = c)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Discriminative model

# %%
B = 100
s = 100

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def basic_function(x, j):
    return np.tanh((x - 3*j/B) / s)

def design_matrix(x):
    design_matrix = np.zeros((x.shape[0], 0))
    for j in range(B):
        design_matrix = np.hstack((design_matrix, basic_function(x, j)))
    return design_matrix

def diagonal_matrix(y):
    diagonal_matrix = np.zeros((y.shape[0], y.shape[0]))
    np.fill_diagonal(diagonal_matrix, y * (1 - y))
    return diagonal_matrix

def predict_discriminative(x, w):
    return softmax(x @ w.T)

def one_hot(t):
    target = np.zeros((t.shape[0], len(classes)))
    for i in range(target.shape[0]):
        target[i][int(t[i] - 1)] = 1
    return target

def update_weight(w, x, t):
    N = x.shape[0]
    x = design_matrix(x)
    y = predict_discriminative(x, w)
    w_new = np.zeros_like(w)
    for i in range(len(classes)):
        R = diagonal_matrix(y[:, i])
        w_new[i, :] = w[i, :] - np.linalg.pinv(x.T @ R @ x) @ x.T @ (y[:, i] - t[:, i])
    return w_new

# %%
def train(data, iters):
    x = data[:, 1:3]
    x = np.insert(x, 0, 1.0, axis=1)
    t_ = one_hot(data[:, 0])
    
    N = x.shape[0]
    F = x.shape[1] * B
    C = y.shape[1]
    w = np.zeros((C, F))
    
    for it in range(iters):
        w = update_weight(w, x, t_)
        
        y_ = predict_discriminative(design_matrix(x), w)
        result = classify(y_)

        acc = np.mean(result == t)
        print(f"accuracy:{acc}")
        
        x1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
        x2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)

        xx1, xx2 = np.meshgrid(x1, x2)
        xxx = np.asarray([xx1.ravel(), xx2.ravel()]).T
        xxx = np.insert(xxx, 0, 1.0, axis=1)
        xxx_dm = design_matrix(xxx)

        y_ = predict_discriminative(xxx_dm, w)
        result = classify(y_)

        plt.figure(figsize=(10,10))
        plt.title(f"Discriminative model decision boundary, acc = {acc}")
        for c in classes:
            xxx_c = np.squeeze(np.take(xxx, np.where(result == c), axis=0))
            plt.scatter(xxx_c[:, 1], xxx_c[:, 2], label = c)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid()
        plt.show()
    return w

# %%
w = train(data, 4)
y = predict_discriminative(design_matrix(np.insert(x, 0, 1.0, axis=1)), w)
result = classify(y)

print(f"accuracy:{np.mean(result == t)}")

# %%
x1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
x2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)

xx1, xx2 = np.meshgrid(x1, x2)
xxx = np.asarray([xx1.ravel(), xx2.ravel()]).T
xxx = np.insert(xxx, 0, 1.0, axis=1)
xxx_dm = design_matrix(xxx)

y = predict_discriminative(xxx_dm, w)
result = classify(y)

plt.figure(figsize=(6,6))
plt.title("Discriminative model decision boundary")
for c in classes:
    xxx_c = np.squeeze(np.take(xxx, np.where(result == c), axis=0))
    plt.scatter(xxx_c[:, 1], xxx_c[:, 2], label = c)
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Part. 2

# %%
seed = 1224
classes = [0, 1, 2]
new_classes = ['A', 'B', 'C']

df = pd.read_csv('HW2_training.csv')
data = df.to_numpy()

print(data.shape)

# %%
x = data[:, 1:3]
t = data[:, 0]
t = np.where(t == 3, 0, t)
data[:, 0] = t

plt.figure(figsize=(6,6))
plt.title("Data plot")
for c in classes:
    x_c = np.squeeze(np.take(x, np.where(t == c), axis=0))
    plt.scatter(x_c[:, 0], x_c[:, 1], label = new_classes[c - 1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Generative Model

# %%
w = weight(data)
b = bias(data)

# %%
y = predict_generative(x, w, b)
result = classify(y)

acc = np.mean(result == t)
print(f"accuracy:{np.mean(result == t)}")

# %%
x1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
x2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)

xx1, xx2 = np.meshgrid(x1, x2)
xxx = np.asarray([xx1.ravel(), xx2.ravel()]).T

y = predict_generative(xxx, w, b)
result = classify(y)

plt.figure(figsize=(6,6))
plt.title(f"Generative model decision boundary, acc={acc}")
for c in classes:
    xxx_c = np.squeeze(np.take(xxx, np.where(result == c), axis=0))
    plt.scatter(xxx_c[:, 0], xxx_c[:, 1], label = new_classes[c - 1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Discriminative model

# %%
def train_2(data, iters):
    x = data[:, 1:3]
    x = np.insert(x, 0, 1.0, axis=1)
    t_ = one_hot(data[:, 0])
    
    N = x.shape[0]
    F = x.shape[1] * B
    C = y.shape[1]
    w = np.zeros((C, F))
    
    for it in range(iters):
        w = update_weight(w, x, t_)
        
        y_ = predict_discriminative(design_matrix(x), w)
        result = classify(y_)

        acc = np.mean(result == t)
        print(f"accuracy:{acc}")
        
        x1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
        x2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)

        xx1, xx2 = np.meshgrid(x1, x2)
        xxx = np.asarray([xx1.ravel(), xx2.ravel()]).T
        xxx = np.insert(xxx, 0, 1.0, axis=1)
        xxx_dm = design_matrix(xxx)

        y_ = predict_discriminative(xxx_dm, w)
        result = classify(y_)

        plt.figure(figsize=(6,6))
        plt.title(f"Discriminative model decision boundary, acc = {acc}")
        for c in classes:
            xxx_c = np.squeeze(np.take(xxx, np.where(result == c), axis=0))
            plt.scatter(xxx_c[:, 1], xxx_c[:, 2], label = new_classes[c - 1])
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid()
        plt.show()
    return w

# %%
B = 100
s = 100

# %%
w = train_2(data, 5)
y = predict_discriminative(design_matrix(np.insert(x, 0, 1.0, axis=1)), w)
result = classify(y)

print(f"accuracy:{np.mean(result == t)}")

# %%
x1 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
x2 = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)

xx1, xx2 = np.meshgrid(x1, x2)
xxx = np.asarray([xx1.ravel(), xx2.ravel()]).T
xxx = np.insert(xxx, 0, 1.0, axis=1)
xxx_dm = design_matrix(xxx)

y = predict_discriminative(xxx_dm, w)
result = classify(y)

plt.figure(figsize=(6,6))
plt.title("Discriminative model decision boundary")
for c in classes:
    xxx_c = np.squeeze(np.take(xxx, np.where(result == c), axis=0))
    plt.scatter(xxx_c[:, 1], xxx_c[:, 2], label = new_classes[c - 1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()