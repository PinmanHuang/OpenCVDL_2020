"""
Author  : Joyce
Date    : 2020-10-28
"""
from UI.hw1_05_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import numpy as np
from matplotlib import pyplot as plt
# Q5
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# Q5_1
import pickle
import random
# Q5_3
# from torchvision import models
from torchsummary import summary

class PyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_10.clicked.connect(self.show_train_images)
        self.pushButton_11.clicked.connect(self.show_hyperparameters)
        self.pushButton_12.clicked.connect(self.show_model_strucuture)
        self.pushButton_13.clicked.connect(self.show_accuracy)
        self.pushButton_14.clicked.connect(self.test)

    def show_train_images(self):
        print('Show train images')
        model = Model()
        model.Q5_1()

    def show_hyperparameters(self):
        print('Show hyperparameters')
        model = Model()
        model.Q5_2()
    
    def show_model_strucuture(self):
        print('Show model strucuture')
        model = Model()
        model.Q5_3()
    
    def show_accuracy(self):
        print('Show accuracy')
        model = Model()
        model.Q5_4()

    def test(self):
        print('Test')
        model = Model()
        model.Q5_5()
    
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        # out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class Model(object):
    def __init__(self):
        super(Model, self).__init__()
        self.batch_size = 256
        self.learning_rate = 1e-2
        self.num_epoches = 50
        self.optimizer = 'SGD'
        
    '''
    batches.meta: size is 10,
                  0 to 9 class,
                  each value is a label to an image
    data_batch_1 ~ data_batch_5: size is 10000*3072 (32*32*3 RGB),
                                 uint8 ndarray,
                                 each row is an image(32*32)
    '''
    def Q5_1(self):
        print('Q5_1')

        # Load data
        with open('./cifar-10-batches-py/batches.meta', 'rb') as fo:
            label_dict = pickle.load(fo, encoding='bytes')
        labels = label_dict[b'label_names']
        with open('./cifar-10-batches-py/data_batch_'+str(random.randint(1, 5)), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        X = data_dict[b'data']
        Y = data_dict[b'labels']
        # Generate random sample image index
        select_img_idx = random.sample(range(10000), 10)

        # Show images
        fig = plt.figure()
        ax = []
        for i in range(10):
            img_data = X[select_img_idx[i], :]                      # size is 3072
            img = img_data.reshape(3, 32, 32).transpose([1, 2, 0])  # reshape image to 3 tunnel(RGB)
            ax.append(fig.add_subplot(2, 5, i + 1))
            ax[-1].set_title(labels[Y[select_img_idx[i]]].decode())
            plt.imshow(img)
        plt.show()

    
    def Q5_2(self):
        print('Q5_2')
        print('hyperparameters:\nbatch size: '+str(self.batch_size)+'\nlearning rate: '+str(self.learning_rate)+'\noptimizer: '+self.optimizer)
    
    def Q5_3(self):
        print('Q5_3')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG16().to(device)
        # model = models.vgg16()
        summary(model, (3, 32, 32))
    
    def Q5_4(self):
        print('Q5_4')
        # Load data
        train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=torchvision.transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # Create VGG16 model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VGG16().to(device)
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        # Train the model
        for epoch in range(self.num_epoches):
            running_loss = 0.0
            correct_pred = 0
            for i, data in enumerate(train_loader):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                # Forwarding
                y_pred = model(img)
                _, pred = torch.max(y_pred, 1)          # prediction the index of the maximum value location
                correct_pred += (pred == label).sum()
                # Backpropagation
                loss = criterion(y_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
            print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%'.format(epoch + 1, self.num_epoches, running_loss / (i + 1), correct_pred.item() / (self.batch_size * (i + 1)) * 100))


    def Q5_5(self):
        print('Q5_5')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())