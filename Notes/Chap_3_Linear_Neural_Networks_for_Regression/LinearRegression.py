import random

import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples): #@save
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w))) # torch.normal(mean, std, size)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # add noise
    return X, y.reshape((-1, 1))    # -1 表示行数由数据总数和列数推算得出: num_examples / num_columns = num_rows

def data_iter(batch_size, features, labels):
    """Iterate through a dataset."""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)    # 将序列的所有元素随机排序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])    # min() 确保在num_example不整除batch_size时, 最后一次迭代不会超出范围
        yield features[batch_indices], labels[batch_indices]   # yield 生成器: 在运行时生成值，而不是一开始就生成所有的值

def init_params():
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b

def linreg(X, w, b):    #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y): #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):    #@save
    """Minibatch stochastic gradient descent."""
    # 禁用自动梯度计算以提高计算效率. 在计算loss时已经计算了梯度, 这里不需要再计算
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_linreg(net, loss, optimizer, 
          lr, num_epochs, batch_size
          , features, labels, w, b):
    """Train a linear regression model."""
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # Minibatch loss in X and y
            # Compute gradient on l with respect to [w, b]
            l.sum().backward()                 # Compute gradient on loss of all parameters
            optimizer([w, b], lr, batch_size)  # Update parameters using their gradient
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

