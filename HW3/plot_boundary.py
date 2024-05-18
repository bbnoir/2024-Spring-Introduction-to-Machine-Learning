import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from model import DNN
from data import HW2_Sample_Dataset

if __name__ == "__main__":
    data_train = np.genfromtxt('./dataset/HW2_training.csv', delimiter=',', skip_header=1)
    num_hidden_layers_list = [1, 2, 3, 4, 5, 6]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HW2_Sample_Dataset("./data/hw2_sample.csv")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
    for num_hidden_layers in num_hidden_layers_list:
        print(f"Plotting boundary for model with {num_hidden_layers} hidden layers")
        model_path = f"./models/hw2_num_hidden_layers_{num_hidden_layers}.pth"
        model = DNN(2, 4, 100, num_hidden_layers).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        pred = torch.zeros(0, dtype=torch.int64).to(device)
        for data in dataloader:
            data = data.to(device)
            output = model(data)
            pred = torch.cat((pred, torch.argmax(output, dim=1)))
        pred = pred.cpu().numpy()
        plt.figure(figsize=(6, 6))
        x1 = np.linspace(0, 100, 1000)
        x2 = np.linspace(0, 100, 1000)
        plt.title(f"Boundary plot for HW2 Dataset with {num_hidden_layers} hidden layers")
        cs = plt.contourf(x1, x2, pred.reshape(1000, 1000).transpose(), levels=[-1, 0, 1, 2, 3], alpha=0.5)
        plt.scatter(data_train[:, 1], data_train[:, 2], c=data_train[:, 0], cmap='viridis', alpha=0.5)
        plt.xlabel("Offensive")
        plt.ylabel("Defensive")
        plt.grid()
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
        # plt.legend(proxy, ["Team 0", "Team 1", "Team 2", "Team 3"], loc='upper left', bbox_to_anchor=(1, 1))
        plt.legend(proxy, ["Team 0", "Team 1", "Team 2", "Team 3"])
        plt.savefig(f"./plots/part4_3_{num_hidden_layers}.png")
        plt.close()