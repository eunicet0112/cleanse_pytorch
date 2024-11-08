#%%

from __future__ import division, print_function

import argparse
import os

import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
from cnn import *
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

from PIL import Image

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

train_losses = []
train_acces = []
test_losses = []
test_acces = []
attack_losses = []
attack_acces = []





class Average(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count

    def __str__(self):
        return '{:.6f}'.format(self.average)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)


class Trainer(object):
    def __init__(self, net, optimizer, train_loader, test_loader_utility, test_loader_integrity, device, scheduler):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader_utility = test_loader_utility
        self.test_loader_integrity = test_loader_integrity
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.scheduler = scheduler

    def fit(self, epochs):
        
        the_last_loss = 100
        early_stop_times = 0
        patience = 5
       
        for epoch in range(1, epochs + 1):
            self.scheduler.step()
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()
            attack_loss, attack_acc = self.attack()
            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
                'attack loss: {}, attack acc: {}.'.format(attack_loss, attack_acc)
            )
            # append 
            train_losses.append(float(format(train_loss)))
            test_losses.append(float(format(test_loss)))
            attack_losses.append(float(format(attack_loss)))

            train_acces.append(float(format(train_acc).split('%')[0]))
            test_acces.append(float(format(test_acc).split('%')[0]))
            attack_acces.append(float(format(attack_acc).split('%')[0]))


            # # Early stopping 
            the_current_loss = float(format(train_loss))
            if the_current_loss > the_last_loss:
                early_stop_times += 1
                print('Early stop times: ', early_stop_times)
                if early_stop_times >= patience:
                    print('Early stopping!')
                    return
            else:
                early_stop_times = 0
            the_last_loss = the_current_loss

    def train(self):
        train_loss = Average()
        train_acc = Accuracy()
        print("_______")
        self.net.train()
        a = 0
        for data, label in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)
            a += 1
            output, _ = self.net(data)
            #print("check output: ", type(output), output.dtype, output.shape)
            #print("check label: ", type(label), label.dtype, label.shape, label)
            loss = self.loss_func(output, label)
            # print(a, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, label)
            # print(a, train_acc)
        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()

        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader_utility:
                data = data.to(self.device)
                label = label.to(self.device)

                # print("data:", data.shape)  # torch.Size([64, 3, 32, 32])

                output, _ = self.net(data)
                loss = self.loss_func(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)
                # print("check output: ", type(output), output.dtype, output.shape)
                # print("check label: ", type(label), label.dtype, label.shape, label)

        # img = Image.open('/home/sense/cleanse_pytorch/data-2/24_raw.jpg')
        # img_ = transform_valid(img).unsqueeze(0) 
        # img_ = img_.to(DEVICE)
        # outputs = model(img_)
        # print("out:", outputs[0])
        # out = torch.tensor(outputs[0])
        # _, indices = torch.max(out,1)
        # print("id: ", indices)

        return test_loss, test_acc

    def attack(self):
        attack_loss = Average()
        attack_acc = Accuracy()

        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader_integrity:
                data = data.to(self.device)
                label = label.to(self.device)

                output, _ = self.net(data)
                loss = self.loss_func(output, label)

                attack_loss.update(loss.item(), data.size(0))
                attack_acc.update(output, label)

        return attack_loss, attack_acc


def get_dataloader(train_gen, test_utility, test_adv_gen):
    #print("check numpy dtyep: ", train_gen[1].dtype)
    train_tensor_X, train_tensor_Y = torch.Tensor(train_gen[0]), torch.from_numpy(train_gen[1])
    #train_tensor_Y = train_tensor_Y.type(torch.LongTensor)
    #print("check tensor dtype: ", train_tensor_Y.dtype)
    trainset = torch.utils.data.TensorDataset(train_tensor_X, train_tensor_Y)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    test_u_x_tensor, test_u_y_tensor = torch.Tensor(test_utility[0]), torch.from_numpy(test_utility[1])
    testset_utility = torch.utils.data.TensorDataset(test_u_x_tensor, test_u_y_tensor)
    test_loader_utility = torch.utils.data.DataLoader(testset_utility, batch_size=64, shuffle=False)

    p_X_test, p_Y_test = torch.Tensor(test_adv_gen[0]), torch.from_numpy(test_adv_gen[1])
    testset_integrity = torch.utils.data.TensorDataset(p_X_test, p_Y_test)
    test_loader_integrity = torch.utils.data.DataLoader(testset_integrity, batch_size=64, shuffle=False)

    return train_loader, test_loader_utility, test_loader_integrity


def run(train_gen, test_utility, test_adv_gen, MODEL_FILEPATH):
    learning_rate = 0.001
    epochs = 100

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = c6f2().to(device)
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1 * 10e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    train_loader, test_loader_utility, test_loader_integrity = get_dataloader(train_gen, test_utility, test_adv_gen)
    trainer = Trainer(net, optimizer, train_loader, test_loader_utility, test_loader_integrity, device, scheduler)
    trainer.fit(epochs)

    # print('min training loss: ', min(train_losses))
    # print('min test loss: ', min(test_losses))
    # print('min attack loss: ', min(attack_losses))
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(test_losses, label='Test loss')
    # plt.plot(attack_losses, label='Attack loss')
    # plt.legend()
    # plt.savefig('loss.png')

    # plt.cla()
    # print('Max training acc: ', max(train_acces))
    # print('Max test acc: ', max(test_acces))
    # print('Max attack acc: ', max(attack_acces))
   
    # plt.plot(train_acces, label='Training acc')
    # plt.plot(test_acces, label='Test acc')
    # plt.plot(attack_acces, label='Attack acc')
    # plt.legend()
    # plt.savefig('acc.png')
    # plt.show()
    
    

    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    torch.save(net, MODEL_FILEPATH)
