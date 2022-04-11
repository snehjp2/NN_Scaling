import torch
from sys import argv
from torch.nn import Linear, Conv2d, CrossEntropyLoss, BatchNorm2d, ReLU, MaxPool2d, Dropout
from torch.optim import SGD
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report
import warnings
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

warnings.filterwarnings('ignore')
torch.manual_seed(0)

# ### argv[0] = script name
# ### argv[1] = # epochs
# ### argv[2] = k-value

ckpt_path = False

script_name = str(argv[0])
epochs = int(argv[1])
k = int(argv[2])


logs_dir = script_name + '_epochs=' + str(epochs) + '_k=' + str(k)
logger = TensorBoardLogger ('tb_logs', name=logs_dir)

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
  
def get_prediction(x, model: pl.LightningModule):
  model.freeze() # prepares model for predicting
  probabilities = torch.softmax(model(x), dim=1)
  predicted_class = torch.argmax(probabilities, dim=1)
  return predicted_class, probabilities

  
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

class VGG16(pl.LightningModule):
  def __init__(self, k):
    super(VGG16, self).__init__()
    self.save_hyperparameters()
    self.k = k
    self.scale = 2**(-self.k)
    
    self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
  
    self.loss = CrossEntropyLoss()
    self.learning_rate = 0.01
    self.epoch = self.current_epoch
    
    self.model.features[0] = Conv2d(in_channels=3, out_channels=int(self.scale*64),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[1] = BatchNorm2d(int(self.scale*64), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[2] = ReLU(inplace=True)
    self.model.features[3] = Conv2d(in_channels=int(self.scale*64), out_channels=int(self.scale*64),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[4] = BatchNorm2d(int(self.scale*64), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[5] = ReLU(inplace=True)
    self.model.features[6] = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.model.features[7] = Conv2d(in_channels=int(self.scale*64), out_channels=int(self.scale*128),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[8] = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[9] = ReLU(inplace=True)
    self.model.features[10] = Conv2d(in_channels=int(self.scale*128), out_channels=int(self.scale*128),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[11] = BatchNorm2d(int(self.scale*128), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[12] = ReLU(inplace=True)
    self.model.features[13] = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.model.features[14] = Conv2d(in_channels=int(self.scale*128), out_channels=int(self.scale*256),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[15] = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[16] = ReLU(inplace=True)
    self.model.features[17] = Conv2d(in_channels=int(self.scale*256), out_channels=int(self.scale*256),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[18] = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[19] = ReLU(inplace=True)
    self.model.features[20] = Conv2d(in_channels=int(self.scale*256), out_channels=int(self.scale*256),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[21] = BatchNorm2d(int(self.scale*256), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[22] = ReLU(inplace=True) 
    self.model.features[23] = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.model.features[24] = Conv2d(in_channels=int(self.scale*256), out_channels=int(self.scale*512),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[25] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[26] = ReLU(inplace=True) 
    self.model.features[27] = Conv2d(in_channels=int(self.scale*512), out_channels=int(self.scale*512),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[28] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[29] = ReLU(inplace=True) 
    self.model.features[30] = Conv2d(in_channels=int(self.scale*512), out_channels=int(self.scale*512),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[31] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[32] = ReLU(inplace=True) 
    self.model.features[33] = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    self.model.features[34] = Conv2d(in_channels=int(self.scale*512), out_channels=int(self.scale*512),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[35] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[36] = ReLU(inplace=True)  
    self.model.features[37] = Conv2d(in_channels=int(self.scale*512), out_channels=int(self.scale*512),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[38] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[39] = ReLU(inplace=True) 
    self.model.features[40] = Conv2d(in_channels=int(self.scale*512), out_channels=int(self.scale*512),kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    self.model.features[41] = BatchNorm2d(int(self.scale*512), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.model.features[42] = ReLU(inplace=True) 
    self.model.features[43] = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    self.model.classifier[0] = Linear(in_features = int(self.scale*25088), out_features = int(4096*self.scale), bias=True)
    self.model.classifier[1] = ReLU(inplace=True) 
    self.model.classifier[2] = Dropout(p=0.5, inplace=False)
    self.model.classifier[3] = Linear(in_features = int(self.scale*4096), out_features = int(4096*self.scale), bias=True)
    self.model.classifier[4] = ReLU(inplace=True)
    self.model.classifier[5] = Dropout(p=0.5, inplace=False)
    self.model.classifier[6] = Linear(in_features = int(self.scale*4096), out_features = int(10), bias=True)
    
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
    
    # self.custom_histogram_adder()
    
    self.logger.experiment.add_scalar("Train_Loss/Epoch", avg_loss, self.current_epoch)
    self.logger.experiment.add_scalar("Train_ACC/Epoch", correct/total, self.current_epoch)
    
  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    correct = sum([x["correct"] for  x in outputs])
    total = sum([x["total"] for  x in outputs])
      
    self.logger.experiment.add_scalar("Val_Loss/Epoch", avg_loss, self.current_epoch)
    self.logger.experiment.add_scalar("Val_ACC/Epoch", correct/total, self.current_epoch)
    
  def configure_optimizers(self):
    optimizer = SGD(self.parameters(), lr=(self.learning_rate), momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    return [optimizer], [scheduler]


model = VGG16(k=k)
trainer = pl.Trainer(max_epochs=epochs, 
                    logger=logger,
                    track_grad_norm=2
                    # gpus=-1
                    )

trainer.fit(model, train_dl, test_dl, ckpt_path=ckpt_path)
trainer.save_checkpoint(logs_dir + '/' + script_name + '_epochs=' + str(epochs) + '_k=' + str(k) + ".pt")

inference_model = VGG16(k=k).load_from_checkpoint(logs_dir + '/' + script_name + '_epochs=' + str(epochs) + '_k=' + str(k) + ".pt")

true_y, pred_y = [], []

for batch in tqdm(iter(test_dl), total=len(test_dl)):
  x, y = batch
  true_y.extend(y)
  preds, probs = get_prediction(x, inference_model)
  pred_y.extend(preds.cpu())

report = classification_report(true_y, pred_y, digits=3)
df = pd.DataFrame([report]).transpose()
df.to_csv(logs_dir + '/' + script_name + '_epochs=' + str(epochs) + '_k=' + str(k) + ".txt")

                                                                                                                                                                                             