import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MaxNLocator

def plot_accuracy(history):
    train_accuracy = [x["train_accuracy"] for x in history]
    test_accuracy = [x["test_accuracy"] for x in history]
    plt.plot(train_accuracy, '-o', label="train_accuracy")
    plt.plot(test_accuracy, '-o', label="test_accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy")
    plt.show()

def plot_part1():
    hidden_dim_list = [5, 10, 20, 50, 75, 100]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, hidden_dim in enumerate(hidden_dim_list):
        history_path = f"./history/hiddendim_{hidden_dim}_layer_1.pkl"
        history = torch.load(history_path)
        train_accuracy = [x["train_accuracy"] for x in history]
        test_accuracy = [x["test_accuracy"] for x in history]
        axs[i//3, i%3].plot(train_accuracy, '-o', label="Training accuracy")
        axs[i//3, i%3].plot(test_accuracy, '-o', label="Testing accuracy")
        axs[i//3, i%3].set_title(f"# Neurons = {hidden_dim}")
        axs[i//3, i%3].legend(loc="lower right")
        axs[i//3, i%3].set_xlabel("Epoch")
        axs[i//3, i%3].set_ylabel("Accuracy")
    plt.suptitle("Accuracy for Different Number of Neurons")
    plt.savefig("./plots/part1_1.png")

    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.subplots_adjust(wspace=0.3)
    for i, hidden_dim in enumerate(hidden_dim_list):
        history_path = f"./history/hiddendim_{hidden_dim}_layer_1.pkl"
        history = torch.load(history_path)
        train_accuracy = [x["train_accuracy"] for x in history]
        test_accuracy = [x["test_accuracy"] for x in history]
        axs2[0].plot(train_accuracy, label=f"# Neurons = {hidden_dim}")
        axs2[1].plot(test_accuracy, label=f"# Neurons = {hidden_dim}")
    axs2[0].set_title("Training Accuracy for Different Number of Neurons")
    axs2[0].legend()
    axs2[0].set_xlabel("Epoch")
    axs2[0].set_ylabel("Accuracy")
    axs2[1].set_title("Testing Accuracy for Different Number of Neurons")
    axs2[1].legend()
    axs2[1].set_xlabel("Epoch")
    axs2[1].set_ylabel("Accuracy")
    plt.savefig("./plots/part1_2.png")

def plot_part2():
    train_data_size_list = [10000, 20000, 30000, 60000]
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, train_data_size in enumerate(train_data_size_list):
        history_path = f"./history/train_data_size_{train_data_size}.pkl"
        history = torch.load(history_path)
        train_accuracy = [x["train_accuracy"] for x in history]
        test_accuracy = [x["test_accuracy"] for x in history]
        axs[i//2, i%2].plot(train_accuracy, '-o', label="Training accuracy")
        axs[i//2, i%2].plot(test_accuracy, '-o', label="Testing accuracy")
        axs[i//2, i%2].set_title(f"# Training Data = {train_data_size}")
        axs[i//2, i%2].legend(loc="lower right")
        axs[i//2, i%2].set_xlabel("Epoch")
        axs[i//2, i%2].set_ylabel("Accuracy")
    plt.suptitle("Accuracy for Different Number of Training Data")
    plt.savefig("./plots/part2_1.png")
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.subplots_adjust(wspace=0.3)
    for i, train_data_size in enumerate(train_data_size_list):
        history_path = f"./history/train_data_size_{train_data_size}.pkl"
        history = torch.load(history_path)
        train_accuracy = [x["train_accuracy"] for x in history]
        test_accuracy = [x["test_accuracy"] for x in history]
        axs2[0].plot(train_accuracy, label=f"# Training Data = {train_data_size}")
        axs2[1].plot(test_accuracy, label=f"# Training Data = {train_data_size}")
    axs2[0].set_title("Training Accuracy for Different Number of Training Data")
    axs2[0].legend()
    axs2[0].set_xlabel("Epoch")
    axs2[0].set_ylabel("Accuracy")
    axs2[1].set_title("Testing Accuracy for Different Number of Training Data")
    axs2[1].legend()
    axs2[1].set_xlabel("Epoch")
    axs2[1].set_ylabel("Accuracy")
    plt.savefig("./plots/part2_2.png")
    
def plot_part3():
    num_hidden_layers_list = [1, 2, 3, 4, 5]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, num_hidden_layers in enumerate(num_hidden_layers_list):
        history_path = f"./history/num_hidden_layers_{num_hidden_layers}.pkl"
        history = torch.load(history_path)
        train_accuracy = [x["train_accuracy"] for x in history]
        test_accuracy = [x["test_accuracy"] for x in history]
        axs[i//3, i%3].plot(train_accuracy, '-o', label="Training accuracy")
        axs[i//3, i%3].plot(test_accuracy, '-o', label="Testing accuracy")
        axs[i//3, i%3].set_title(f"# Hidden Layers = {num_hidden_layers}")
        axs[i//3, i%3].legend(loc="lower right")
        axs[i//3, i%3].set_xlabel("Epoch")
        axs[i//3, i%3].set_ylabel("Accuracy")
    plt.suptitle("Accuracy for Different Number of Hidden Layers")
    plt.savefig("./plots/part3_1.png")
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.subplots_adjust(wspace=0.3)
    for i, num_hidden_layers in enumerate(num_hidden_layers_list):
        history_path = f"./history/num_hidden_layers_{num_hidden_layers}.pkl"
        history = torch.load(history_path)
        train_accuracy = [x["train_accuracy"] for x in history]
        test_accuracy = [x["test_accuracy"] for x in history]
        axs2[0].plot(train_accuracy, label=f"# Hidden Layers = {num_hidden_layers}")
        axs2[1].plot(test_accuracy, label=f"# Hidden Layers = {num_hidden_layers}")
    axs2[0].set_title("Training Accuracy for Different Number of Hidden Layers")
    axs2[0].legend()
    axs2[0].set_xlabel("Epoch")
    axs2[0].set_ylabel("Accuracy")
    axs2[1].set_title("Testing Accuracy for Different Number of Hidden Layers")
    axs2[1].legend()
    axs2[1].set_xlabel("Epoch")
    axs2[1].set_ylabel("Accuracy")
    plt.savefig("./plots/part3_2.png")

def plot_part4():
    pass
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--part", dest="part", required=True, help="part number")
    parser.add_argument("-f", "--file", dest="history", help="history file")
    args = parser.parse_args()
    
    if args.part == "1":
        plot_part1()
    elif args.part == "2":
        plot_part2()
    elif args.part == "3":
        plot_part3()
    elif args.part == "4":
        plot_part4()
    elif args.history is None:
        print("Please specify a history file")
    else:
        history = torch.load(args.history)
        plot_accuracy(history)

if __name__ == "__main__":
    main()