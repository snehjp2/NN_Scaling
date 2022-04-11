import profile
import matplotlib
from sklearn.utils import shuffle
import torch
import os
import sys
from sys import argv
import setuptools
import torch.nn as nn
import pandas as pd
from torch.nn import Linear, Conv2d, CrossEntropyLoss, BatchNorm2d
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW, Adam, SGD, RMSprop
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from tqdm.autonotebook import tqdm
import seaborn as sns
from sklearn.metrics import classification_report
import warnings
from IPython.display import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchsummary import summary
from skmultilearn.model_selection import iterative_train_test_split
from pytorch_lightning.loggers import TensorBoardLogger

sns.set()
warnings.filterwarnings('ignore')
# torch.manual_seed(0)

# ### argv[0] = script name
# ### argv[1] = # epochs
# ### argv[2] = k

ckpt_path=False

k = int(sys.argv[2])
script_name = str(sys.argv[0])
logs_dir = script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2])
logger = TensorBoardLogger('tb_logs', name=logs_dir)

def get_prediction(x, model: pl.LightningModule):
  model.freeze() # prepares model for predicting
  probabilities = torch.softmax(model(x), dim=1)
  predicted_class = torch.argmax(probabilities, dim=1)
  return predicted_class, probabilities

def visualize_data(dataloader):
  examples = enumerate(dataloader)
  batch_idx, (example_data, example_targets) = next(examples)
  print(example_data.shape)
  for i in range(example_data.shape[0]):
    plt.subplot(example_data.shape[0]//3 + 1, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()
  
def visualize_performance(model, dataloader):
  examples = enumerate(dataloader)
  batch_idx, (example_data, example_targets) = next(examples)
  print(example_data.shape)
  output = model(example_data)
  for i in range(example_data.shape[0]):
    plt.subplot(example_data.shape[0]//3 + 1, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
  plt.show()
  
def subset_data(dataset, frac: float): 
    subset = torch.utils.data.Subset(dataset, range(0, frac))
    return subset

def analyze_subset(dataset):
    labels = []
    counts = {}
    for i in range(0,len(dataset)):
        labels.append(dataset[i][1])

    for i in range(0,10):
        counts[i] = labels.count(i)
    plt.bar(counts.keys(), counts.values(), tick_label=range(0,10))
    plt.xlabel('Integers')
    plt.ylabel('Frequency')
    plt.title(f'Total # of Digits: {len(dataset)}')
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def combine_dataset(train_ds, test_ds):
    full_train_ds = ConcatDataset([train_ds, test_ds])
    return full_train_ds

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = CIFAR10("./data", train=True, download=True, transform = transform)
test_ds = CIFAR10("./data", train=False, download=True, transform = transform)

train_dl = DataLoader(train_ds, batch_size=16, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=16, num_workers=0)

# full_train_dl = DataLoader(full_train_ds, batch_size=16, num_workers=0) # size = 157

class ResNet50(pl.LightningModule):
  def __init__(self, k):
    super(ResNet50, self).__init__()
    self.save_hyperparameters()
    self.k = k
    self.scale =2**(-self.k) 
    self.model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
    self.accuracy = Accuracy()
    
    self.model.conv1 = Conv2d(3, int(64*self.scale), kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
    self.model.bn1 = BatchNorm2d(int(self.scale*64), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[0].conv1 = Conv2d(int(self.scale*64), int(self.scale*128), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[0].bn1 = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[0].conv2 = Conv2d(int(self.scale*128), int(self.scale*128), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer1[0].bn2 = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[0].conv3 = Conv2d(int(self.scale*128), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[0].bn3 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[0].downsample[0] = Conv2d(int(self.scale*64), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[0].downsample[1] = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer1[1].conv1 = Conv2d(int(self.scale*256), int(self.scale*128), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[1].bn1 = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[1].conv2 = Conv2d(int(self.scale*128), int(self.scale*128), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer1[1].bn2 = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[1].conv3 = Conv2d(int(self.scale*128), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[1].bn3 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                       
    self.model.layer1[2].conv1 = Conv2d(int(self.scale*256), int(self.scale*128), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[2].bn1 = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[2].conv2 = Conv2d(int(self.scale*128), int(self.scale*128), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer1[2].bn2 = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer1[2].conv3 = Conv2d(int(self.scale*128), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer1[2].bn3 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer2[0].conv1 = Conv2d(int(self.scale*256), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[0].bn1 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[0].conv2 = Conv2d(int(self.scale*256), int(self.scale*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer2[0].bn2 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[0].conv3 = Conv2d(int(self.scale*256), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[0].bn3 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[0].downsample[0] = Conv2d(int(self.scale*256), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[0].downsample[1] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer2[1].conv1 = Conv2d(int(self.scale*512), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[1].bn1 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[1].conv2 = Conv2d(int(self.scale*256), int(self.scale*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer2[1].bn2 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[1].conv3 = Conv2d(int(self.scale*256), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[1].bn3 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
           
    self.model.layer2[2].conv1 = Conv2d(int(self.scale*512), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[2].bn1 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[2].conv2 = Conv2d(int(self.scale*256), int(self.scale*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer2[2].bn2 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[2].conv3 = Conv2d(int(self.scale*256), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[2].bn3 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer2[3].conv1 = Conv2d(int(self.scale*512), int(self.scale*256), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[3].bn1 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[3].conv2 = Conv2d(int(self.scale*256), int(self.scale*256), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer2[3].bn2 = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer2[3].conv3 = Conv2d(int(self.scale*256), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer2[3].bn3 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
           
    self.model.layer3[0].conv1 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[0].bn1 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[0].conv2 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer3[0].bn2 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[0].conv3 = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[0].bn3 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[0].downsample[0] = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[0].downsample[1] = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer3[1].conv1 = Conv2d(int(self.scale*1024), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[1].bn1 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[1].conv2 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer3[1].bn2 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[1].conv3 = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[1].bn3 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
           
    self.model.layer3[2].conv1 = Conv2d(int(self.scale*1024), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[2].bn1 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[2].conv2 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer3[2].bn2 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[2].conv3 = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[2].bn3 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
           
    self.model.layer3[3].conv1 = Conv2d(int(self.scale*1024), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[3].bn1 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[3].conv2 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer3[3].bn2 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[3].conv3 = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[3].bn3 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer3[4].conv1 = Conv2d(int(self.scale*1024), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[4].bn1 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[4].conv2 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer3[4].bn2 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[4].conv3 = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[4].bn3 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer3[5].conv1 = Conv2d(int(self.scale*1024), int(self.scale*512), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[5].bn1 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[5].conv2 = Conv2d(int(self.scale*512), int(self.scale*512), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer3[5].bn2 = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer3[5].conv3 = Conv2d(int(self.scale*512), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer3[5].bn3 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer4[0].conv1 = Conv2d(int(self.scale*1024), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[0].bn1 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[0].conv2 = Conv2d(int(self.scale*1024), int(self.scale*1024), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer4[0].bn2 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[0].conv3 = Conv2d(int(self.scale*1024), int(self.scale*2048), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[0].bn3 = BatchNorm2d(int(self.scale*2048), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[0].downsample[0] = Conv2d(int(self.scale*1024), int(self.scale*2048), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[0].downsample[1] = BatchNorm2d(int(self.scale*2048), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer4[1].conv1 = Conv2d(int(self.scale*2048), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[1].bn1 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[1].conv2 = Conv2d(int(self.scale*1024), int(self.scale*1024), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer4[1].bn2 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[1].conv3 = Conv2d(int(self.scale*1024), int(self.scale*2048), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[1].bn3 = BatchNorm2d(int(self.scale*2048), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.layer4[2].conv1 = Conv2d(int(self.scale*2048), int(self.scale*1024), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[2].bn1 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[2].conv2 = Conv2d(int(self.scale*1024), int(self.scale*1024), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.model.layer4[2].bn2 = BatchNorm2d(int(self.scale*1024), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.layer4[2].conv3 = Conv2d(int(self.scale*1024), int(self.scale*2048), kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.model.layer4[2].bn3 = BatchNorm2d(int(self.scale*2048), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    self.model.fc = Linear(in_features=int(self.scale*2048), out_features=10, bias=True)

    self.loss = CrossEntropyLoss()
    self.epoch = self.current_epoch

  def custom_histogram_adder(self):
    for name,params in self.named_parameters():
      self.logger.experiment.add_histogram(name,params,self.current_epoch)
      self.logger.experiment.add_histogram(f'{name}.grad',params.grad, self.current_epoch)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_no):
    x, y = batch
    logits, total = self(x), len(x)
    loss = self.loss(logits, y)
    pred = self.forward(x)
    correct = pred.argmax(dim=1).eq(y).sum().item()
    logs= {"train_loss:", loss}
    train_batch_dictionary = {"loss": loss, "log": logs, 'correct': correct, 'total': total}
    return train_batch_dictionary
  
  def validation_step(self, batch, batch_no):
    x, y = batch
    logits, total = self(x), len(x)
    loss = self.loss(logits, y)
    preds = self.forward(x)
    correct = preds.argmax(dim=1).eq(y).sum().item()
    logs = {"val_loss:", loss}
    val_batch_dictionary ={"loss": loss, "log": logs, 'correct': correct, 'total': total}
    return val_batch_dictionary
    
  def training_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    correct = sum([x["correct"] for  x in outputs])
    total = sum([x["total"] for  x in outputs])
    
    self.custom_histogram_adder()
    
    self.logger.experiment.add_scalar("Train_Loss/Epoch", avg_loss, self.current_epoch)
    self.logger.experiment.add_scalar("Train_ACC/Epoch", correct/total, self.current_epoch)
    
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    correct = sum([x["correct"] for  x in outputs])
    total = sum([x["total"] for  x in outputs])
      
    self.logger.experiment.add_scalar("Val_Loss/Epoch", avg_loss, self.current_epoch)
    self.logger.experiment.add_scalar("Val_ACC/Epoch", correct/total, self.current_epoch)
    
  def configure_optimizers(self):
    optimizer = SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    return [optimizer], [scheduler]

model = ResNet50(k=k)
trainer = pl.Trainer(max_epochs=int(sys.argv[1]), 
                     logger=logger, 
                     track_grad_norm=2, 
                     )
trainer.fit(model, train_dl, test_dl, ckpt_path=ckpt_path)
trainer.save_checkpoint(logs_dir + '/' + script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".pt")

inference_model = ResNet50(k=k).load_from_checkpoint(logs_dir + '/' + script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".pt")

true_y, pred_y = [], []

for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch
  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())

report = classification_report(true_y, pred_y, digits=3)
df = pd.DataFrame([report]).transpose()
df.to_csv(logs_dir + '/' + script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".txt")

# visualize_data(test_dl)
# visualize_performance(inference_model, test_dl)

