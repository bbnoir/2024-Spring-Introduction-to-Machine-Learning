# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def plot_confusion(filename, title):
    conf_data = np.genfromtxt(filename, delimiter=',', skip_header=0)
    plt.figure(figsize=(6,6))
    plt.title(title)
    cs = plt.imshow(conf_data, cmap='Purples', alpha=0.95)
    for i in range(conf_data.shape[0]):
        for j in range(conf_data.shape[1]):
            plt.text(j, i, str(int(conf_data[i, j])), ha='center', va='center', fontsize=12)  # Increase font size to 12
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(conf_data.shape[1]), np.arange(conf_data.shape[1]).astype(int), fontsize=12)  # Increase font size to 12
    plt.yticks(np.arange(conf_data.shape[0]), np.arange(conf_data.shape[0]).astype(int), fontsize=12)  # Increase font size to 12
    plt.colorbar(cs)
    plt.savefig(filename.replace('.csv', '.png').replace('results', 'plots'))
    plt.show()

    
# %%
plot_confusion('../results/gen_train_confusion_4.csv', "Confusion matrix on training data")
plot_confusion('../results/gen_test_confusion_4.csv', "Confusion matrix on testing data")

# %%
plot_confusion('../results/dis_train_confusion_4.csv', "Confusion matrix on training data")
plot_confusion('../results/dis_test_confusion_4.csv', "Confusion matrix on testing data")

# %%
plot_confusion('../results/gen_train_confusion_3.csv', "Confusion matrix on training data")
plot_confusion('../results/gen_test_confusion_3.csv', "Confusion matrix on testing data")

# %%
plot_confusion('../results/dis_train_confusion_3.csv', "Confusion matrix on training data")
plot_confusion('../results/dis_test_confusion_3.csv', "Confusion matrix on testing data")

# %%
