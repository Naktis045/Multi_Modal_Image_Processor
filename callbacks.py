import torch
import torch.optim as optim
import torch.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# more custom callbacks in lib ___bolts___

class CustomCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')# patience=3 means if val_loss does not improve for 3 consecutive epochs, stop training(standart)

        checkpoint = ModelCheckpoint(
                    dirpath="checkpoints/",
                    filename="model-{epoch:02d}-{val_loss:.f}",
                    save_top_k=1,
                    save_last=True,
                    monitor="val_loss",
                    mode="min",
                )
        
    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module):
        print("Training has ended!")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        return super().on_load_checkpoint(trainer, pl_module, checkpoint)
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)
        