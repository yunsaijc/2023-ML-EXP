import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from IPython import display
from d2l import torch as d2l

num_inputs = 784    # 输入图片尺寸28*28
num_outputs = 10    # 10个类别
W = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
lr = 0.1
num_epochs = 10


def load_data_fashion_mnist(batch_size, resize=None):   #@save
    trans = [transforms.ToTensor()]
    if resize: trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="../../data/Chap3/FashionMNIST", 
                                                    train=True,
                                                    transform=trans,
                                                    download=False,
                                                    )
    mnist_test = torchvision.datasets.FashionMNIST(root="../../data/Chap3/FashionMNIST", 
                                                    train=False,
                                                    transform=trans,
                                                    download=False,
                                                    )
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)    
            )

def soft_max(X):
    """定义softmax函数"""
    X_exp = X.exp() # 对每个元素求指数
    partition = X_exp.sum(dim=1, keepdim=True)  # 对每行求和，维度变成(num_rows, 1)
    return X_exp / partition    # 此处应用了广播机制

def net(X):
    """定义模型"""
    reshaped_X = X.reshape(-1, W.shape[0])  # 为了能够矩阵相乘，按照W的第一维度reshape X的第二维度，-1表示自动推断
    return soft_max(torch.matmul(reshaped_X, W) + b)

def cross_entropy(y_hat, y):
    """
    定义交叉熵损失函数
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    y_hat[[0, 1], y] = tensor([0.1000, 0.5000])
    """
    return -torch.log(y_hat[range(len(y_hat)), y])  # 通过索引y来获取y_hat中对应于正确标签的预测概率

def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:   # 维度和列数（类别数）大于1
        y_hat = y_hat.argmax(axis=1)    # 假定在第一个维度上是样本，第二个维度是每个类别的预测概率，那么沿着第二个维度取最大值的索引
    cmp = y_hat.type(y.dtype) == y  # 将y_hat转换为y的数据类型，然后比较是否相等
    return float(cmp.type(y.dtype).sum()) # 将比较结果转换为y的数据类型，然后求和
# / len(y) # 将比较结果转换为y的数据类型，然后求和并除以y的长度

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)    # 正确预测数，预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())   # 正确预测数，预测总数
    return metric[0] / metric[1]

class Accumulator:  #@save
    """在Accumulator类中定义了add方法, 用于将两个变量的值相加, 并将结果存储在第一个变量中。"""
    def __init__(self, n):
        self.data = [0.0] * n   # 一个长度为n的列表，每个元素初始化为0.0

    def add(self, *args):
        """
        self.data = [0.0, 0.0]
        args = [1.0, 2.0]
        for a, b in zip(self.data, args):
            iter1: a: 0.0, b: 1.0
            iter2: a: 0.0, b: 2.0
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)] # 将两个变量的值相加, 并将结果存储在第一个变量中

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train() # 将模型设置为训练模式
    metric = Accumulator(3)   # 训练损失总和，训练准确度总和，样本数
    for X, y in train_iter: # 计算梯度并更新参数
        y_hat = net(X)  
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):  # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:   # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练损失和训练准确率

class Animator:  #@save
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))    # 训练损失，训练准确率，测试准确率
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)



def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
