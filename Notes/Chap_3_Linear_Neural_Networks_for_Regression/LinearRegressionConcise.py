import numpy as np
import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def init_model():
    # Define the model
    layer_1 = nn.Linear(2, 1)   # Define a single layer, input size is 2, output size is 1
    net = nn.Sequential(        # Combine all layers
        layer_1,
    )

    # Initialize weights and bias
    # net[0] is layer_1
    net[0].weight.data.normal_(0, 0.01)  
    net[0].bias.data.fill_(0)            

    # Define loss function
    loss = nn.MSELoss()     # Mean Squared Error, 均方误差

    # Define optimization algorithm
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # Stochastic Gradient Descent, 随机梯度下降

    return net, loss, trainer

def train(net, loss, trainer, num_epochs, batch_size, data_iter, features, labels):
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)     # Evaluate
        print(f'epoch {epoch + 1}, loss {l:f}')
