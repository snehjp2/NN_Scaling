from socket import ntohs
from typing import Any
import torch
import sys
from sys import argv
import torch.nn as nn
from torch import Tensor
import pandas as pd
from torch.nn import Linear, Conv2d, CrossEntropyLoss, BatchNorm2d, ReLU
from torch.optim import SGD
from torch.utils.data import  DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import seaborn as sns
from sklearn.metrics import classification_report
import warnings
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger

# ### argv[0] = script name
# ### argv[1] = # epochs
# ### argv[2] = k

ckpt_path=False

k = int(sys.argv[2])
epochs = int(sys.argv[1])
script_name = str(sys.argv[0])

logs_dir = script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2])
logger = TensorBoardLogger('tb_logs', name='logs_dir')

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
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = CIFAR10("./data", train=True, download=True, transform = transform)
test_ds = CIFAR10("./data", train=False, download=True, transform = transform)

train_dl = DataLoader(train_ds, batch_size=16, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=16, num_workers=0)

class BasicConv2d(nn.Module):
  def __init__(self, input_channels: int, output_channels: int, **kwargs: Any):
    super().__init__()
    
    self.layers = nn.Sequential(
      Conv2d(input_channels, output_channels, bias=False, **kwargs),
      BatchNorm2d(output_channels, eps=0.001),
      ReLU(inplace=True))
    
  def forward(self, x: Tensor):
    return self.layers(x)

class InceptionA(nn.Module):

    def __init__(self, input_channels: int, pool_features: int, k: int):
        super().__init__()
        self.k = k
        self.scale =2**(-self.k) 
        
        self.branch1x1 = BasicConv2d(input_channels, int(self.scale*64), kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, int(self.scale*48), kernel_size=1),
            BasicConv2d(int(self.scale*48), int(self.scale*64), kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, int(self.scale*64), kernel_size=1),
            BasicConv2d(int(self.scale*64), int(self.scale*96), kernel_size=3, padding=1),
            BasicConv2d(int(self.scale*96), int(self.scale*96), kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)
        branchpool = self.branchpool(x)
        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, input_channels: int, k:int):
      
        super().__init__()
        self.k = k
        self.scale =2**(-self.k)
        
        self.branch3x3 = BasicConv2d(input_channels, int(self.scale*384), kernel_size=3, stride=2)
        
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, int(self.scale*64), kernel_size=1),
            BasicConv2d(int(self.scale*64), int(self.scale*96), kernel_size=3, padding=1),
            BasicConv2d(int(self.scale*96), int(self.scale*96), kernel_size=3, stride=2)
        )
        
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3stack = self.branch3x3stack(x)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
  
    def __init__(self, input_channels: int, channels_7x7: int, k: int):
        super().__init__()
        self.k = k
        self.scale =2**(-self.k)
        
        self.branch1x1 = BasicConv2d(input_channels, int(self.scale*192), kernel_size=1)
        
        c7 = channels_7x7
        
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, int(self.scale*192), kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, int(self.scale*192), kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, int(self.scale*192), kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7stack = self.branch7x7stack(x)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]
        return torch.cat(outputs, 1)
      
class InceptionD(nn.Module):

    def __init__(self, input_channels: int, k: int):
        super().__init__()
        self.k = k
        self.scale = 2**(-self.k)
        
        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, int(self.scale*192), kernel_size=1),
            BasicConv2d(int(self.scale*192), int(self.scale*320), kernel_size=3, stride=2)
        )
        
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, int(self.scale*192), kernel_size=1),
            BasicConv2d(int(self.scale*192), int(self.scale*192), kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(int(self.scale*192), int(self.scale*192), kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(int(self.scale*192), int(self.scale*192), kernel_size=3, stride=2)
        )
        
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch7x7, branchpool]
        return torch.cat(outputs, 1)
    
class InceptionE(nn.Module):
  
    def __init__(self, input_channels: int, k: int):
        super().__init__()
        self.k = k
        self.scale =2**(-self.k)
        
        self.branch1x1 = BasicConv2d(input_channels, int(self.scale*320), kernel_size=1)

        self.branch3x3_1 = BasicConv2d(input_channels, int(self.scale*384), kernel_size=1)
        self.branch3x3_2a = BasicConv2d(int(self.scale*384), int(self.scale*384), kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(int(self.scale*384), int(self.scale*384), kernel_size=(3, 1), padding=(1, 0))
            
        self.branch3x3stack_1 = BasicConv2d(input_channels, int(self.scale*448), kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(int(self.scale*448), int(self.scale*384), kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(int(self.scale*384), int(self.scale*384), kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(int(self.scale*384), int(self.scale*384), kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, int(self.scale*192), kernel_size=1)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionV3(pl.LightningModule):
  
  def __init__(self, k):
    super(InceptionV3, self).__init__()
    self.save_hyperparameters()
    self.k = k
    self.scale = 2**(-self.k) 
    self.aux_logits = False
    
    self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    
    self.model.Conv2d_1a_3x3 = BasicConv2d(3, int(self.scale*32), kernel_size=3, stride=2)
    self.model.Conv2d_2a_3x3 = BasicConv2d(int(self.scale*32), int(self.scale*32), kernel_size=3)
    self.model.Conv2d_2b_3x3 = BasicConv2d(int(self.scale*32), int(self.scale*64), kernel_size=3, padding=1)
    
    self.model.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    self.model.Conv2d_3b_1x1 = BasicConv2d(int(self.scale*64), int(self.scale*80), kernel_size=1)
    self.model.Conv2d_4a_3x3 = BasicConv2d(int(self.scale*80), int(self.scale*192), kernel_size=3)
    
    self.model.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    
    self.model.Mixed_5b = InceptionA(int(self.scale*192), int(self.scale*32), k)
    self.model.Mixed_5c = InceptionA(int(self.scale*256), int(self.scale*64), k)
    self.model.Mixed_5d = InceptionA(int(self.scale*288), int(self.scale*64), k)
    
    self.model.Mixed_6a = InceptionB(int(self.scale*288), k)
    
    self.model.Mixed_6b = InceptionC(int(self.scale*768), int(self.scale*128), k)
    self.model.Mixed_6c = InceptionC(int(self.scale*768), int(self.scale*160), k)
    self.model.Mixed_6d = InceptionC(int(self.scale*768), int(self.scale*160), k)
    self.model.Mixed_6e = InceptionC(int(self.scale*768), int(self.scale*192), k)
    
    self.model.Mixed_7a = InceptionD(int(self.scale*768), k)
    self.model.Mixed_7b = InceptionE(int(self.scale*1280), k)
    self.model.Mixed_7c = InceptionE(int(self.scale*2048), k)
    
    self.model.fc = Linear(in_features=int(self.scale*2048), out_features=10)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.loss = CrossEntropyLoss()
    self.epoch = self.current_epoch

  def custom_histogram_adder(self):
    for name,params in self.named_parameters():
      self.logger.experiment.add_histogram(name,params,self.current_epoch)
      self.logger.experiment.add_histogram(f'{name}.grad',params.grad, self.current_epoch)

  def forward(self, x):
    x = self.model.Conv2d_1a_3x3(x)
    x = self.model.Conv2d_2a_3x3(x)
    x = self.model.Conv2d_2b_3x3(x)
    
    x = self.model.maxpool1(x)
    
    x = self.model.Conv2d_3b_1x1(x)
    x = self.model.Conv2d_4a_3x3(x)
    
    x = self.model.maxpool2(x)

    x = self.model.Mixed_5b(x)
    x = self.model.Mixed_5c(x)
    x = self.model.Mixed_5d(x)

    x = self.model.Mixed_6a(x)

    x = self.model.Mixed_6b(x)
    x = self.model.Mixed_6c(x)
    x = self.model.Mixed_6d(x)
    x = self.model.Mixed_6e(x)

    x = self.model.Mixed_7a(x)
    x = self.model.Mixed_7b(x)
    x = self.model.Mixed_7c(x)

    x = self.model.avgpool(x)
    
    x = self.model.dropout(x)
    
    x = torch.flatten(x, 1)
    out = self.model.fc(x)
    
    return out

  def training_step(self, batch, batch_no):
    x, y = batch
    logits, total = self(x), len(x)
    loss = self.loss(logits, y)
    pred = self.forward(x)
    correct = pred.argmax(dim=1).eq(y).sum().item()
    logs= {"train_loss:", loss}
    train_batch_dictionary ={"loss": loss, "log": logs, 'correct': correct, 'total': total}
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

model = InceptionV3(k=k)
trainer = pl.Trainer(max_epochs=epochs, 
                     logger=logger, 
                     track_grad_norm=2
                    #  gpus=-1,
                     )
trainer.fit(model, train_dl, test_dl, ckpt_path=ckpt_path)
trainer.save_checkpoint('./' + script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".pt")

inference_model = InceptionV3(k=k).load_from_checkpoint('./' + script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".pt")

true_y, pred_y = [], []

for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch
  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())

report = classification_report(true_y, pred_y, digits=3)
df = pd.DataFrame([report]).transpose()
df.to_csv('./' + script_name + '_epochs=' + str(argv[1]) + '_k=' + str(argv[2]) + ".txt")

# visualize_data(test_dl)
# visualize_performance(inference_model, test_dl)

