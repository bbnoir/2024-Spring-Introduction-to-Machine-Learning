import matplotlib.pyplot as plt
import numpy as np
import os

# Parse
HW1_data = np.genfromtxt('HW1.csv', delimiter=',', skip_header=1)
output_data = np.genfromtxt('output.csv', delimiter=',', skip_header=1)
pred_data = np.genfromtxt('pred.csv', delimiter=',', skip_header=1)

M_list = [5, 10, 15, 20, 25, 30]
train_t = HW1_data[:10000, 0]
train_x = HW1_data[:10000, 1:]
train_pred = pred_data[:10000, :]
test_t = HW1_data[10000:, 0]
test_x = HW1_data[10000:, 1:]
test_pred = pred_data[10000:, :]

train_mse = output_data[:6, 1]
train_acc = output_data[:6, 2]
test_mse = output_data[:6, 3]
test_acc = output_data[:6, 4]

# Plot

pic_path = './pic2/'
if not os.path.exists(pic_path):
    os.mkdir(pic_path)

## Scatter plot of train and test data
for i in range(11):
    plt.figure(figsize=(20, 10))
    for m in range(6):
        plt.subplot(2, 3, m+1)
        plt.scatter(train_x[:, i], train_t, label='train_t', alpha=0.5)
        plt.scatter(train_x[:, i], train_pred[:, m], label='train_pred', alpha=0.5)
        plt.legend()
        plt.title('M={}'.format(M_list[m]))
        plt.xlabel('x')
        plt.ylabel('y/t')
    if not os.path.exists(pic_path + 'scatter/train/'):
        os.makedirs(pic_path + 'scatter/train/')
    plt.savefig(pic_path + 'scatter/train/fitting_curve_f{}.png'.format(i+1))
    plt.close()
    plt.figure(figsize=(20, 10))
    for m in range(6):
        plt.subplot(2, 3, m+1)
        plt.scatter(test_x[:, i], test_t, label='test_t', alpha=0.5)
        plt.scatter(test_x[:, i], test_pred[:, m], label='test_pred', alpha=0.5)
        plt.legend()
        plt.title('M={}'.format(M_list[m]))
        plt.xlabel('x')
        plt.ylabel('y/t')
    if not os.path.exists(pic_path + 'scatter/test/'):
        os.makedirs(pic_path + 'scatter/test/')
    plt.savefig(pic_path + 'scatter/test/fitting_curve_f{}.png'.format(i+1))
    plt.close()

## Part 2 - MSE and Accuracy
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(M_list, train_mse, label='train_mse', marker='o')
plt.plot(M_list, test_mse, label='test_mse', marker='o')
plt.legend()
plt.title('MSE')
plt.xlabel('M')
plt.ylabel('MSE')
plt.subplot(1, 2, 2)
plt.plot(M_list, train_acc, label='train_acc', marker='o')
plt.plot(M_list, test_acc, label='test_acc', marker='o')
plt.legend()
plt.title('Accuracy')
plt.xlabel('M')
plt.ylabel('Accuracy')
plt.savefig(pic_path + 'mse_acc.png')
plt.close()

