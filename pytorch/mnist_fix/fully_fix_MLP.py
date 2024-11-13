import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from train_MLP_fix import fixed_fwd_op, fix_train_op
import pdb
import numpy as np
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F
# from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score
# from tqdm import tqdm
from typing import Tuple, List, Any

# from utils import tensor_to_txt
from fully_fix_linear import fix_x, fully_fix_linear


class NonBN_MLP_Fullfix(torch.nn.Module):
    """ 
    This module operates fixlization on several critical computations:
    * fixed model parameter
    * fixed activation after linear/relu
    * upper overflow and underflow of weights and activation are supported
    
    However, the data format computed in linear operations remains fp32, which
    is different from hardware fix8 computation.

    The improved version is in fully_fix_MLP.py
    
    """
    def __init__(self):
        super(NonBN_MLP_Fullfix, self).__init__()

        self.linear1 = torch.nn.Linear(28*28, 512)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(512, 256)
        self.relu2 = torch.nn.ReLU()
        self.output = nn.Linear(256, 10)

    def fp_forward(self, x):
        x = x.view(-1, 28*28)  # 将输入数据展平成一维向量
        x = self.linear1(x)
        x = self.relu1(x) 
        x = self.linear2(x)
        x = self.relu2(x)
        x= self.output(x)  
        return x

    def load_fullfix_model(self):
        self.linear1 = fully_fix_linear(self.linear1)
        self.linear2 = fully_fix_linear(self.linear2)
        self.output = fully_fix_linear(self.output)
        param = torch.load('./winter_weights/FullFloorWFix_8_32_64_32.pt')
        # param = torch.load('./AllFullFix8_32_64_32.pt')
        # param = torch.load('./FullFix8_32_64_32.pt')
        self.load_state_dict(param)
        self.fullfix_model()

    def fullfix_model(self):
        tmp = torch.round(self.linear1.weight.data*32)
        self.linear1.weight.data = tmp / 32
        tmp = torch.round(self.linear1.bias.data*32)
        self.linear1.bias.data = tmp / 32

        tmp = torch.round(self.linear2.weight.data*32)
        self.linear2.weight.data = tmp / 32
        tmp = torch.round(self.linear2.bias.data*32)
        self.linear2.bias.data = tmp / 32

        tmp = torch.round(self.output.weight.data*32)
        self.output.weight.data = tmp / 32
        tmp = torch.round(self.output.bias.data*32)
        self.output.bias.data = tmp / 32

        dim_i, dim_j = self.linear1.weight.data.shape
        for i in range(dim_i):
            for j in range(dim_j):
                if(self.linear1.weight.data[i, j] > 3.96875):
                    self.linear1.weight.data[i, j] = 3.96875
                elif(self.linear1.weight.data[i, j] < -3.96875):
                    self.linear1.weight.data[i, j] = -3.96875
        
        dim_i, dim_j = self.linear2.weight.data.shape
        for i in range(dim_i):
            for j in range(dim_j):
                if(self.linear2.weight.data[i, j] > 3.96875):
                    self.linear2.weight.data[i, j] = 3.96875
                elif(self.linear2.weight.data[i, j] < -3.96875):
                    self.linear2.weight.data[i, j] = -3.96875

        dim_i, dim_j = self.output.weight.data.shape
        for i in range(dim_i):
            for j in range(dim_j):
                if(self.output.weight.data[i, j] > 3.96875):
                    self.output.weight.data[i, j] = 3.96875
                elif(self.output.weight.data[i, j] < -3.96875):
                    self.output.weight.data[i, j] = -3.96875

        dim_i = self.linear1.bias.data.shape[0]
        for i in range(dim_i):
            if(self.linear1.bias.data[i] > 3.96875):
                    self.linear1.bias.data[i] = 3.96875
            elif(self.linear1.bias.data[i] < -3.96875):
                self.linear1.bias.data[i] = -3.96875

        dim_i = self.linear2.bias.data.shape[0]
        for i in range(dim_i):
            if(self.linear2.bias.data[i] > 3.96875):
                    self.linear2.bias.data[i] = 3.96875
            elif(self.linear2.bias.data[i] < -3.96875):
                self.linear2.bias.data[i] = -3.96875

        dim_i = self.output.bias.data.shape[0]
        for i in range(dim_i):
            if(self.output.bias.data[i] > 3.96875):
                    self.output.bias.data[i] = 3.96875
            elif(self.output.bias.data[i] < -3.96875):
                self.output.bias.data[i] = -3.96875
    
    def fixed_fwd(self, x):
        x = x.view(-1, 28*28)  # 将输入数据展平成一维向量
        M = 3.96875
        with torch.no_grad():
            x = fix_x(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)        
        x= self.output(x)
        # pdb.set_trace()
        return x

    def weight_split(self):
        # self.fixed_model(overflow=True)
        layer1_wchunk_1 = self.linear1.weight.T[0:8]
        layer1_wchunk_2 = self.linear1.weight.T[8:16]
        layer1_wchunk_3 = self.linear1.weight.T[16:24]
        layer1_wchunk_4 = self.linear1.weight.T[24:32]

        layer2_wchunk_1 = self.linear2.weight.T[0:8]
        layer2_wchunk_2 = self.linear2.weight.T[8:16]
        layer2_wchunk_3 = self.linear2.weight.T[16:24]
        layer2_wchunk_4 = self.linear2.weight.T[24:32]
        layer2_wchunk_1 = torch.cat((layer2_wchunk_1, self.linear2.weight.T[32:40]), dim=0)
        layer2_wchunk_2 = torch.cat((layer2_wchunk_2, self.linear2.weight.T[40:48]), dim=0)
        layer2_wchunk_3 = torch.cat((layer2_wchunk_3, self.linear2.weight.T[48:56]), dim=0)
        layer2_wchunk_4 = torch.cat((layer2_wchunk_4, self.linear2.weight.T[56:64]), dim=0)

        output_wchunk_1 = self.output.weight.T[0:8]
        output_wchunk_2 = self.output.weight.T[8:16]
        output_wchunk_3 = self.output.weight.T[16:24]
        output_wchunk_4 = self.output.weight.T[24:32]
        output_wchunk_1 = torch.cat((output_wchunk_1, self.output.weight.T[32:40]), dim=0)
        output_wchunk_2 = torch.cat((output_wchunk_2, self.output.weight.T[40:48]), dim=0)
        output_wchunk_3 = torch.cat((output_wchunk_3, self.output.weight.T[48:56]), dim=0)
        output_wchunk_4 = torch.cat((output_wchunk_4, self.output.weight.T[56:64]), dim=0)
        
        pad_tensor = torch.zeros(8, 5)
        output_wchunk_1 = torch.cat((output_wchunk_1, pad_tensor), dim=1)
        output_wchunk_2 = torch.cat((output_wchunk_2, pad_tensor), dim=1)
        output_wchunk_3 = torch.cat((output_wchunk_3, pad_tensor), dim=1)
        output_wchunk_4 = torch.cat((output_wchunk_4, pad_tensor), dim=1)

        tensor_to_txt(layer1_wchunk_1, './MLP_weight_chunk/layer1_wchunk_1.txt', n_dim=2)
        tensor_to_txt(layer1_wchunk_2, './MLP_weight_chunk/layer1_wchunk_2.txt', n_dim=2)
        tensor_to_txt(layer1_wchunk_3, './MLP_weight_chunk/layer1_wchunk_3.txt', n_dim=2)
        tensor_to_txt(layer1_wchunk_4, './MLP_weight_chunk/layer1_wchunk_4.txt', n_dim=2)
        tensor_to_txt(layer2_wchunk_1, './MLP_weight_chunk/layer2_wchunk_1.txt', n_dim=2)
        tensor_to_txt(layer2_wchunk_2, './MLP_weight_chunk/layer2_wchunk_2.txt', n_dim=2)
        tensor_to_txt(layer2_wchunk_3, './MLP_weight_chunk/layer2_wchunk_3.txt', n_dim=2)
        tensor_to_txt(layer2_wchunk_4, './MLP_weight_chunk/layer2_wchunk_4.txt', n_dim=2)
        tensor_to_txt(output_wchunk_1, './MLP_weight_chunk/output_wchunk_1.txt', n_dim=2)
        tensor_to_txt(output_wchunk_2, './MLP_weight_chunk/output_wchunk_2.txt', n_dim=2)
        tensor_to_txt(output_wchunk_3, './MLP_weight_chunk/output_wchunk_3.txt', n_dim=2)
        tensor_to_txt(output_wchunk_4, './MLP_weight_chunk/output_wchunk_4.txt', n_dim=2)


    def bias_extract(self):
        # self.fixed_model(overflow=True)
        layer1_bias = self.linear1.bias
        layer2_bias = self.linear2.bias
        output_bias = self.output.bias

        tensor_to_txt(layer1_bias, './MLP_weight_chunk/layer1_bias.txt', n_dim=1)
        tensor_to_txt(layer2_bias, './MLP_weight_chunk/layer2_bias.txt', n_dim=1)
        tensor_to_txt(output_bias, './MLP_weight_chunk/output_bias.txt', n_dim=1)

    def print_param(self):
        print(self.output.weight)
        print(self.output.bias)
        # for name, param in self.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# 训练模型
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model.fixed_fwd(data)
        # output = model.fp_forward(data)
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
            output = model.fixed_fwd(data)
            # output = model.fp_forward(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if (__name__ == '__main__'):
    model_fix8 = NonBN_MLP_Fullfix()
    # model_fix8.load_fullfix_model()
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(model_fix8.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST('../data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # 运行训练和测试
    for epoch in range(1, 2):
        train(model_fix8, train_loader, optimizer, epoch)
        model_fix8.fullfix_model()
        print("fullfix_model")
        test(model_fix8, test_loader)
    model_fix8.print_param()


    # model_fix8.fixed_model()

    # _, val_data_rows, _ = load_data()
    # val_dataset = get_dataset(val_data_rows)
    # val_data_loader = DataLoader(
    #     val_dataset, 
    #     # batch_size=1, 
    #     batch_size=102400, 
    #     shuffle=False, 
    #     drop_last=True
    # )
    # fixed_fwd_op(model=model_fix8, data_loader=val_data_loader)
    
    # fix_train_op(model=model_fix8, n_epochs=25)

    # model_fix8.weight_split()
    # model_fix8.bias_extract()


