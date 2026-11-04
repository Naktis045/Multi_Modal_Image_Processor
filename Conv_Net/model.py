import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import pytorch_lightning as pl

import torchmetrics as metrics
from torchmetrics import Mertric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers, data_dir, accelerator, devices,learning_rate

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/ConvNet/tensorboard')
step = 0

class Model(pl.LightningModule):
    def __init__(self, input_size, learning_rate, hidden_size, output_size, num_classes):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.input = nn.Linear(input_size, hidden_size)
        self.Conv2D = nn.Conv2D(128, 3, padding = 'same', kernel_regularizer = 'l2')
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MaxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Conv2D = nn.Conv2D(64, 3, padding = 'same', kernel_regularizer = 'l2')
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MaxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Conv2D = nn.Conv2D(32, 3, padding = 'same', kernel_regularizer = 'l2')
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MaxPool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.Flatten = nn.Flatten()
        self.fc = nn.Lirear(32, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(output_size, hidden_size)


        self.accuracy = metrics.Accuracy(task='multiclass', num_classes=num_classes)
        #self.f1_score = metrics.F1Score(task = 'multiclass', num_classes=num_classes)
        #self.my_accuracy = My_Accuracy()

    def forward(self, x):
        input = self.input()
        x = self.Conv2D(input)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.MaxPool(x)
        x = self.Conv2D(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.MaxPool(x)
        x = self.Conv2D(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.MaxPool(x)
        x = self.Flatten(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.output(x)

    def train_step(self, batch, batch_idx):
        loss, scores , y = self.common_step(batch, batch_idx)
        accuracy = self.accuracy(scores,y)
        #my_accuracy = self.my_accuracy(scores,y)
        #self.log('train_loss', loss, prog_bar=True)
        #self.log('train_accuracy', accuracy, prog_bar=True)
        # OR -- same
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy},
                      on_step = True, on_epoch = True, prog_bar=True)
        
        writer.add_scalar('Training loss', loss, global_step=step)
        writer.add_scalar('Training Accuracy', accuracy, global_step=step)
        step = step + 1
         # Optimize tensorboard vis.
        losses = []
        losses.append(loss.item())
        accuracies = []
        accuracies.append(self.accuracy)
        
        writer.add_scalar({'lr':learning_rate, 'batch_size':batch_size},
                          {'accuracy': sum(accuracies)/len(accuracies),
                           'loss':sum(losses)/len(losses)})
                
        
        # For Tensorboard visualization, log every 100 steps
        if batch_idx % 100 == 0:
            print(f"Train Step: {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28), nrow=4)
            self.logger.experiment.add_image('train_images', grid, self.global_step)

        return {'loss': loss, 'accuracy': accuracy, 'y': y, 'scores': scores}# for train_epoch_end(dict)-average loss calculation

    # Very Cool func.
    def train_epoch_end(self, outputs):
        average_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', average_loss, prog_bar=True)
        #******scores = 

    def val_step(self, batch, batch_idx):
        loss, scores , y = self.common_step(batch, batch_idx)
        val_accuracy = self.accuracy(scores, y)
        self.log_dict({'val_loss': loss,'val_accuracy': val_accuracy}, prog_bar=True)

        writer.add_scalar('Validational loss', loss, global_step=step)
        writer.add_scalar('Validataional Accuracy', val_accuracy, global_step=step)
        step = step + 1
         # Optimize tensorboard vis.
        losses = []
        losses.append(loss.item())
        accuracies = []
        accuracies.append(self.accuracy)
        
        writer.add_scalar({'lr':learning_rate, 'batch_size':batch_size},
                          {'accuracy': sum(accuracies)/len(accuracies),
                          'loss':sum(losses)/len(losses)})
                
        
    
    def test_step(self, batch, batch_idx):
        loss, scores , y = self.common_step(batch, batch_idx)
        test_accuracy = self.accuracy(scores, y)
        self.log_dict({'test_loss': loss,'test_accuracy': test_accuracy}, prog_bar=True)

        writer.add_scalar('Test loss', loss, global_step=step)
        writer.add_scalar('Test Accuracy', test_accuracy, global_step=step)
        step = step + 1
    # Optimize tensorboard vis. *********
        losses = []
        losses.append(loss.item())
        accuracies = []
        accuracies.append(self.accuracy)

        writer.add_scalar({'lr':learning_rate, 'batch_size':batch_size},
                          {'accuracy': sum(accuracies)/len(accuracies),
                           'loss':sum(losses)/len(losses)})
        

    def common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = nn.CrossEntropyLoss()(scores, y)
        acc = (scores.argmax(dim=1) == y).float().mean()
        return loss, scores, y

    def config_optim(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def learning_rate_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def early_stopping(self, patience=3):
        return pl.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=True, mode='min')

        
        







