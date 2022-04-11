import torch
import os
import sys
from sys import argv
import setuptools
import torch.nn as nn
from torch.nn import Linear, Conv2d, CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW, Adam, SGD, RMSprop
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report
import warnings
from IPython.display import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchsummary import summary
from skmultilearn.model_selection import iterative_train_test_split
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')
torch.manual_seed(0)

# ### argv[0] = script name
# ### argv[1] = # epochs
# ### argv[2] = k

script_name = str(sys.argv[0])
logger = TensorBoardLogger ('tb_logs', name=script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Helper function to subset data
def subset_data(dataset,k): 
    subset = torch.utils.data.Subset(dataset, range(0, int(2**(-k)*(len(dataset)))))
    return subset

# Helper function to see distribution of data subset
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
    
# return number of total trainable parameters for model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# this is a very dirty way of combining the datasets ¯\_(ツ)_/¯
def combine_dataset(train_ds, test_ds):
    full_train_ds = ConcatDataset([train_ds, test_ds])
    return full_train_ds
  
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

transform = transforms.Compose([transforms.ToTensor()])

train_ds = MNIST("mnist", train=True, download=True, transform = transform) # size = ([60000, 28, 28], [60000])
test_ds = MNIST("mnist", train=True, download=True, transform = transform )# size = ([10000, 28, 28], [10000])
full_train_ds = combine_dataset(train_ds, test_ds)

subset_ds = subset_data(full_train_ds, int(sys.argv[2]))

train_dl = DataLoader(train_ds, batch_size=16, num_workers=0) # size = 157
test_dl = DataLoader(test_ds, batch_size=16, num_workers=0)
full_train_dl = DataLoader(full_train_ds, batch_size=16, num_workers=0) # size = 157
subset_dl = DataLoader(subset_ds, batch_size=16, shuffle=True, num_workers=0)

class ResNet50(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    self.model.fc = Linear(in_features=2048, out_features=10)
    self.model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.loss = CrossEntropyLoss()
    self.epoch = self.current_epoch

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_no):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    logs={"train_loss:", loss}
    batch_dictionary={"loss": loss, "log": logs}
    return batch_dictionary

  def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train_Loss/Epoch", avg_loss, self.current_epoch)
        epoch_dictionary={'loss': avg_loss}
#         return epoch_dictionary

  def configure_optimizers(self):
    optimizer = SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    return [optimizer], [scheduler]

model = ResNet50()
trainer = pl.Trainer(max_epochs=int(sys.argv[1]), logger=logger)
trainer.fit(model, full_train_dl)
trainer.save_checkpoint(script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".pt")

def get_prediction(x, model: pl.LightningModule):
  model.freeze() # prepares model for predicting
  probabilities = torch.softmax(model(x), dim=1)
  predicted_class = torch.argmax(probabilities, dim=1)
  return predicted_class, probabilities

inference_model = ResNet50.load_from_checkpoint(script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".pt", map_location=device)

true_y, pred_y = [], []
for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch
  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())

with open('./text_logs/' + script_name + '.txt', 'w') as f:
    f.write(classification_report(true_y, pred_y, digits=3))
