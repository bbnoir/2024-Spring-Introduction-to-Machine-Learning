import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig('./pic/scatter/train/fitting_curve_f{}.png'.format(i+1))
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
    plt.savefig('./pic/scatter/test/fitting_curve_f{}.png'.format(i+1))
