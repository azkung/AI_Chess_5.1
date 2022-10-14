import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
from training.model import Net
import sys


def run_train(modelPath, xPath, yPath, epochs, prevModelPath = None, useCUDA = False, batchSize = 10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not useCUDA:
        device = torch.device('cpu')
    print("Loading numpy")
    train_x = np.load(xPath)
    train_y = np.load(yPath)
    print(train_x.shape)
    print(train_y.shape)

    print("Loading Device")
    
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.Tensor(train_y)

    net = Net()
    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    net.to(device)


    trainSet = TensorDataset(tensor_x, tensor_y)
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)

    print("Device loaded")

    if(prevModelPath != None):
        net.load_state_dict(torch.load(prevModelPath))

    optimizer = optim.Adam(net.parameters(), lr =0.0001)

    criterion = nn.MSELoss()

    print("Epochs Started")

    for epoch in range(epochs):
        steps = 0
        for idx, data in enumerate(trainLoader):
            X, y = data
            net.zero_grad()
            output = net(X)
            loss = criterion(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 10000 == 0:
                print("Epoch:", epoch + 1, "| Epoch Steps:", steps)
        print("Completed Epoch", epoch + 1, "| Latest Loss:", loss,)

    torch.save(net.state_dict(), modelPath)
    del tensor_x
    del tensor_y
    torch.cuda.empty_cache()
