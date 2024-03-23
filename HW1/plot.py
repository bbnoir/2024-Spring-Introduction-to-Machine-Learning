import matplotlib.pyplot as plt
import numpy as np
import os

M_list = [5, 10, 15, 20, 25, 30]

# Parse
HW1_data = np.genfromtxt('HW1.csv', delimiter=',', skip_header=1)
output_data = np.genfromtxt('./output/new_test/accuracy.csv', delimiter=',', skip_header=1)
train_pred = np.genfromtxt('./output/new_test/train_prediction.csv', delimiter=',', skip_header=1)
test_pred = np.genfromtxt('./output/new_test/test_prediction.csv', delimiter=',', skip_header=1)
redge_result = np.genfromtxt('./output/new_test/ridge_accuracy.csv', delimiter=',', skip_header=1)
redge_train_pred = np.genfromtxt('./output/new_test/ridge_train_prediction.csv', delimiter=',', skip_header=1)
redge_test_pred = np.genfromtxt('./output/new_test/ridge_test_prediction.csv', delimiter=',', skip_header=1)

train_t = HW1_data[:10000, 0]
train_x = HW1_data[:10000, 1:]
test_t = HW1_data[10000:, 0]
test_x = HW1_data[10000:, 1:]

# Plot

pic_path = './output/pic6/'
if not os.path.exists(pic_path):
    os.mkdir(pic_path)
    os.mkdir(pic_path + 'normal/')
    os.mkdir(pic_path + 'normal/train/')
    os.mkdir(pic_path + 'normal/test/')
    os.mkdir(pic_path + 'ridge/')
    os.mkdir(pic_path + 'ridge/train/')
    os.mkdir(pic_path + 'ridge/test/')


def scatter_plot(x, t, y, label, title, pic_name):
    plt.figure(figsize=(20, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.scatter(x, t, label='t', alpha=0.5)
        plt.scatter(x, y[:, i], label=label, alpha=0.5)
        plt.legend()
        plt.title('M={}'.format(M_list[i]))
        plt.xlabel('x')
        plt.ylabel('y/t')
    plt.savefig(pic_path + pic_name)
    plt.close()

## Scatter plot of train and test data
for i in range(11):
    scatter_plot(train_x[:, i], train_t, train_pred, 'train', 'Train Data', 'normal/train/{}.png'.format(i))
    scatter_plot(test_x[:, i], test_t, test_pred, 'test', 'Test Data', 'normal/test/{}.png'.format(i))
    scatter_plot(train_x[:, i], train_t, redge_train_pred, 'train', 'Train Data', 'ridge/train/{}.png'.format(i))
    scatter_plot(test_x[:, i], test_t, redge_test_pred, 'test', 'Test Data', 'ridge/test/{}.png'.format(i))

## Part 2 - MSE and Accuracy

def plot_acc(train_mse, train_acc, test_mse, test_acc, pic_name):
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
    plt.savefig(pic_path + pic_name)
    plt.close()

plot_acc(output_data[:6, 1], output_data[:6, 2], output_data[:6, 3], output_data[:6, 4], 'normal/acc_mse.png')
plot_acc(redge_result[:6, 1], redge_result[:6, 2], redge_result[:6, 3], redge_result[:6, 4], 'ridge/acc_mse.png')
