import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from model import DNN
from data import HW2_Dataset

def plot_conf(confusion, save_path, fig_title):
    plt.figure(figsize=(6,6))
    cs = plt.imshow(confusion, cmap='Purples', alpha=0.95)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, str(int(confusion[i, j])), ha='center', va='center', fontsize=12)
    plt.title(fig_title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(confusion.shape[1]), np.arange(confusion.shape[1]).astype(int), fontsize=12)
    plt.yticks(np.arange(confusion.shape[0]), np.arange(confusion.shape[0]).astype(int), fontsize=12)
    plt.colorbar(cs)
    plt.savefig(save_path)

if __name__ == "__main__":
    data_train = np.genfromtxt('./dataset/HW2_training.csv', delimiter=',', skip_header=1)
    num_hidden_layers_list = [1, 2, 3, 4, 5, 6]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = HW2_Dataset("./dataset/HW2_training.csv")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
    testset = HW2_Dataset("./dataset/HW2_testing.csv")
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    for num_hidden_layers in num_hidden_layers_list:
        print(f"Plotting boundary for model with {num_hidden_layers} hidden layers")
        model_path = f"./models/hw2_num_hidden_layers_{num_hidden_layers}.pth"
        model = DNN(2, 4, 100, num_hidden_layers, True).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        pred = torch.zeros(0, dtype=torch.int64).to(device)
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred_batch = torch.max(outputs, 1)
            pred = torch.cat((pred, pred_batch))
        pred = pred.cpu().numpy()
        confusion = np.zeros((4, 4))
        for i in range(len(pred)):
            confusion[trainset.label[i], pred[i]] += 1
        plot_conf(confusion, f"./plots/part4_4_{num_hidden_layers}_1.png", "Confusion Matrix for Training Data")
        
        pred = torch.zeros(0, dtype=torch.int64).to(device)
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred_batch = torch.max(outputs, 1)
            pred = torch.cat((pred, pred_batch))
        pred = pred.cpu().numpy()
        confusion = np.zeros((4, 4))
        for i in range(len(pred)):
            confusion[testset.label[i], pred[i]] += 1
        plot_conf(confusion, f"./plots/part4_4_{num_hidden_layers}_2.png", "Confusion Matrix for Testing Data")
