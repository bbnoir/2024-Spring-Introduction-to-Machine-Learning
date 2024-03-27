import matplotlib.pyplot as plt
import numpy as np
import os

M_list = [5, 10, 15, 20, 25, 30]

# Parse
HW1_data = np.genfromtxt('HW1.csv', delimiter=',', skip_header=1)
output_data = np.genfromtxt('./accuracy.csv', delimiter=',', skip_header=1)
train_pred = np.genfromtxt('./train_prediction.csv', delimiter=',', skip_header=1)
test_pred = np.genfromtxt('./test_prediction.csv', delimiter=',', skip_header=1)
redge_result = np.genfromtxt('./ridge_accuracy.csv', delimiter=',', skip_header=1)
redge_train_pred = np.genfromtxt('./ridge_train_prediction.csv', delimiter=',', skip_header=1)
redge_test_pred = np.genfromtxt('./ridge_test_prediction.csv', delimiter=',', skip_header=1)

train_t = HW1_data[:10000, 0]
train_x = HW1_data[:10000, 1:]
test_t = HW1_data[10000:, 0]
test_x = HW1_data[10000:, 1:]

# Plot

pic_path = './pic/'
if not os.path.exists(pic_path):
    os.mkdir(pic_path)
    os.mkdir(pic_path + 'normal/')
    os.mkdir(pic_path + 'normal/train/')
    os.mkdir(pic_path + 'normal/test/')
    os.mkdir(pic_path + 'ridge/')
    os.mkdir(pic_path + 'ridge/train/')
    os.mkdir(pic_path + 'ridge/test/')


def scatter_plot(x, t, y, label, title, pic_name):
    plt.figure()
    for i in range(6):
        plt.subplot(3, 2, i+1)
        plt.scatter(x, t, label=label+'_t', alpha=0.5, marker='o', s=0.7)
        plt.scatter(x, y[:, i], label=label+'_y', alpha=0.5, marker='o', s=0.7)
        plt.legend()
        plt.title('M={}'.format(M_list[i]))
        plt.xlabel('x3')
        plt.ylabel('y/t')
    # plt.savefig(pic_path + pic_name)
    plt.show()
    plt.close()

## Scatter plot of train and test data
# for i in range(11):
i = 2
scatter_plot(train_x[:, i], train_t, train_pred, 'train', 'Train Data', 'normal/train/{}.png'.format(i))
scatter_plot(test_x[:, i], test_t, test_pred, 'test', 'Test Data', 'normal/test/{}.png'.format(i))
scatter_plot(train_x[:, i], train_t, redge_train_pred, 'train', 'Train Data', 'ridge/train/{}.png'.format(i))
scatter_plot(test_x[:, i], test_t, redge_test_pred, 'test', 'Test Data', 'ridge/test/{}.png'.format(i))

## Part 2 - MSE and Accuracy

def plot_mse(train_mse, test_mse, pic_name):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(M_list, train_mse, label='train_mse', marker='o')
    plt.legend()
    plt.title('Train MSE')
    plt.xlabel('M')
    plt.ylabel('MSE')
    plt.subplot(1, 2, 2)
    plt.plot(M_list, test_mse, label='test_mse', marker='o')
    plt.legend()
    plt.title('Test MSE')
    plt.xlabel('M')
    plt.ylabel('MSE')
    plt.show()
    # plt.savefig(pic_path + pic_name)
    print('Save to', pic_path + pic_name)
    plt.close()

def plot_acc(train_acc, test_acc, pic_name):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(M_list, train_acc, label='train_acc', marker='o')
    plt.legend()
    plt.title('Train Accuracy')
    plt.xlabel('M')
    plt.ylabel('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(M_list, test_acc, label='test_acc', marker='o')
    plt.legend()
    plt.title('Test Accuracy')
    plt.xlabel('M')
    plt.ylabel('Accuracy')
    plt.show()
    # plt.savefig(pic_path + pic_name)
    print('Save to', pic_path + pic_name)
    plt.close()

plot_acc(output_data[:6, 1], output_data[:6, 2], output_data[:6, 3], output_data[:6, 4], 'normal/acc_mse.png')
plot_acc(redge_result[:6, 1], redge_result[:6, 2], redge_result[:6, 3], redge_result[:6, 4], 'ridge/acc_mse.png')
plot_mse(output_data[:6, 1], output_data[:6, 3], 'normal/mse.png')
plot_acc(output_data[:6, 2], output_data[:6, 4], 'normal/acc.png')
plot_mse(redge_result[:6, 1], redge_result[:6, 3], 'ridge/mse.png')
plot_acc(redge_result[:6, 2], redge_result[:6, 4], 'ridge/acc.png')

x = train_x[:, 2]
t = train_t
y = train_pred[:, 2]
plt.figure()
plt.scatter(x, t, label='train_t', alpha=0.5, marker='o', s=0.7)
plt.scatter(x, y, label='train_y', alpha=0.5, marker='o', s=0.7)
plt.legend()
plt.title('M=5')
plt.xlabel('x3')
plt.ylabel('y/t')
# plt.savefig(pic_path + pic_name)
plt.show()
plt.close()

# plot cross validation accuracy
cv_data = np.genfromtxt('./cv_accuracy_sum.csv', delimiter=',', skip_header=1)
plt.figure()
plt.plot(M_list, cv_data[:, 1], label='val_acc_sum', marker='o')
# add value label
for i in range(6):
    plt.text(M_list[i], cv_data[i, 1], str(cv_data[i, 1]), size=12, ha='center', va='bottom')
plt.legend()
plt.title('Cross Validation Accuracy Sum')
plt.xlabel('M')
plt.ylabel('Accuracy Sum')
plt.show()
plt.close()

# correlation between train_pred and each feature
corr_data = np.zeros((11, 6))
for m in range(6):
    for i in range(11):
        corr_data[i, m] = np.corrcoef(train_x[:, i], train_pred[:, m])[0, 1]
plt.figure()
for m in range(6):
    plt.plot(range(11), corr_data[:, m], label='M={}'.format(M_list[m]), marker='o')
    plt.legend()
    plt.title('Correlation between Train Prediction and Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Correlation')
    plt.show()
    plt.close()
