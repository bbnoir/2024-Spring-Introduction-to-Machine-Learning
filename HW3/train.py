import torch
import torchvision
import torchvision.transforms as transforms
from model import DNN
from tqdm import tqdm
import argparse
from data import HW2_Dataset

def avoid_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, trainloader, testloader, criterion, optimizer, n_epochs, device, n_class, save_path=None):
    history = []
    best_accuracy = 0
    for epoch in range(n_epochs):
        result = {}
        model.train()
        total = trainloader.dataset.__len__()
        correct = 0
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels_onehot = torch.zeros(labels.size(0), n_class).to(device)
            labels_onehot.scatter_(1, labels.view(-1, 1), 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
        result["train_accuracy"] = correct / total
        
        model.eval()
        total = testloader.dataset.__len__()
        correct = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
        result["test_accuracy"] = correct / total
        history.append(result)
        # Save the model if the test accuracy is the best
        if result["test_accuracy"] > best_accuracy:
            best_accuracy = result["test_accuracy"]
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch+1}/{n_epochs}: train_accuracy={result['train_accuracy']}, test_accuracy={result['test_accuracy']}, best_accuracy={best_accuracy}")

    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--part", dest="part", required=True, help="part number")
    args = parser.parse_args()

    train_batch_size = 64

    if args.part != "4":
        # Load MNIST dataset
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    else:
        # Load HW2 dataset
        trainset = HW2_Dataset("./dataset/HW2_training.csv")
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
        testset = HW2_Dataset("./dataset/HW2_testing.csv")
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        

    # Choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # device = torch.device("cpu")

    # Shared hyperparameters
    n_epochs = 20
    random_seed = 67
    criterion = torch.nn.CrossEntropyLoss()

    if args.part == "1":
        # Part 1-1: Different number of neurons with 1 hidden layer
        print("Part 1-1: Different number of neurons with 1 hidden layer")
        hidden_dim_list = [5, 10, 20, 50, 75, 100]
        for hidden_dim in hidden_dim_list:
            avoid_randomness(random_seed)
            print(f"Start training with hidden_dim={hidden_dim}")
            model = DNN(784, 10, hidden_dim, 1, False).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            history = train(model, trainloader, testloader, criterion, optimizer, n_epochs, device, 10)
            history_path = f"./history/hiddendim_{hidden_dim}_layer_1.pkl"
            torch.save(history, history_path)
            print(f"Saving history to {history_path}")
    elif args.part == "2":
        # Part 1-2: Different number of training data with 2 hidden layers of 100 neurons
        print("Part 1-2: Different number of training data with 2 hidden layers of 100 neurons")
        train_data_size_list = [10000, 20000, 30000, 60000]
        for train_data_size in train_data_size_list:
            avoid_randomness(random_seed)
            print(f"Start training with train_data_size={train_data_size}")
            cropped_trainset = torch.utils.data.Subset(trainset, list(range(train_data_size)))
            cropped_trainloader = torch.utils.data.DataLoader(cropped_trainset, batch_size=train_batch_size, shuffle=True)
            model = DNN(784, 10, 100, 2, False).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            history = train(model, cropped_trainloader, testloader, criterion, optimizer, n_epochs, device, 10)
            history_path = f"./history/train_data_size_{train_data_size}.pkl"
            torch.save(history, history_path)
            print(f"Saving history to {history_path}")
    elif args.part == "3":
        # Part 1-3: Different number of hidden layers with 100 neurons
        print("Part 1-3: Different number of hidden layers with 100 neurons")
        num_hidden_layers_list = [1, 2, 3, 4, 5]
        for num_hidden_layers in num_hidden_layers_list:
            avoid_randomness(random_seed)
            print(f"Start training with num_hidden_layers={num_hidden_layers}")
            model = DNN(784, 10, 100, num_hidden_layers, True).to(device)
            optimizer = torch.optim.Adam(model.parameters())
            history = train(model, trainloader, testloader, criterion, optimizer, n_epochs, device, 10)
            history_path = f"./history/num_hidden_layers_{num_hidden_layers}.pkl"
            torch.save(history, history_path)
            print(f"Saving history to {history_path}")
    elif args.part == "4":
        # Part 2: Train on the dataset of HW2
        print("Part 2: Train on the dataset of HW2")
        num_hidden_layers_list = [1, 2, 3, 4, 5, 6]
        n_epochs = 50
        for num_hidden_layers in num_hidden_layers_list:
            avoid_randomness(random_seed)
            print(f"Start training with num_hidden_layers={num_hidden_layers}")
            model = DNN(2, 4, 100, num_hidden_layers, True).to(device) # batch_norm=True
            optimizer = torch.optim.Adam(model.parameters())
            history = train(model, trainloader, testloader, criterion, optimizer, n_epochs, device, 4, f"./models/hw2_num_hidden_layers_{num_hidden_layers}.pth")
            history_path = f"./history/hw2_num_hidden_layers_{num_hidden_layers}.pkl"
            torch.save(history, history_path)
            print(f"Saving history to {history_path}")
    else:
        raise ValueError("Invalid part number")

if __name__ == "__main__":
    main()
