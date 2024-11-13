import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)  # 第一层全连接层，输入特征数为784（28*28），输出特征数为512
        self.fc2 = nn.Linear(512, 256)    # 第二层全连接层，输入特征数为512，输出特征数为256
        self.fc3 = nn.Linear(256, 10)     # 第三层全连接层，输入特征数为256，输出特征数为10（类别数）

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将输入数据展平成一维向量
        x = torch.relu(self.fc1(x))  # 通过第一层全连接层和ReLU激活函数
        x = torch.relu(self.fc2(x))  # 通过第二层全连接层和ReLU激活函数
        x = self.fc3(x)  # 通过第三层全连接层
        return x

model = Net()
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
# train_dataset = datasets.MNIST('./data', train=True, download=False)
# test_dataset = datasets.MNIST('./data', train=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

# 训练模型
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 测试模型
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 运行训练和测试
for epoch in range(1, 2):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)



