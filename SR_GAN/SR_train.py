import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from SR_GAN.SR_dataset import SRDataModule
from SR_GAN.SR_model import Generator, Discriminator

from config import batch_size, num_workers, data_dir_train, data_dir_test, accelerator, devices,device, learning_rate, min_epochs, max_epochs


device = device

# matrix multiplication
torch.set_float32_matmul_precision('medium')


# GAN LightningModule Wrapper to manage Generator & Discriminator together
class SRModel(pl.LightningModule):
    def __init__(self, learning_rate=learning_rate):
        super().__init__()
        self.automatic_optimization = False  # Recommended for GANs with separate optimizers
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.lr = learning_rate

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        optim_g, optim_d = self.optimizers()

        # 1. Train Discriminator
        fake_imgs = self.generator(lr_imgs)
        loss_d = self.discriminator(hr_imgs).mean() - self.discriminator(fake_imgs.detach()).mean()
        
        optim_d.zero_grad()
        self.manual_backward(loss_d)
        optim_d.step()

        # 2. Train Generator
        loss_g = -self.discriminator(fake_imgs).mean() + self.vgg_loss(fake_imgs, hr_imgs)
        
        optim_g.zero_grad()
        self.manual_backward(loss_g)
        optim_g.step()

        # Log losses
        self.log_dict({"loss_d": loss_d, "loss_g": loss_g}, prog_bar=True)

    def configure_optimizers(self):
        optim_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        # Create a learning rate scheduler for each optimizer (e.g., decay LR by 10% every 30 epochs)
        scheduler_g = torch.optim.lr_scheduler.StepLR(optim_g, step_size=30, gamma=0.1)#gamma parameter - mult factor used to reduce __decay__ of learning_rate over time
        scheduler_d = torch.optim.lr_scheduler.StepLR(optim_d, step_size=30, gamma=0.1)
        
        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()


if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="sr_model")

    # Strategy for Multi-GPU training
    strategy = DeepSpeedStrategy()

    # Profiler configuration
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler_sr"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    # Instantiate the unified Lightning model
    model = SRModel(learning_rate=learning_rate)
    model.to(device)

    # Datamodule setup
    datamodule = SRDataModule(
        data_dir_train=data_dir_train, 
        data_dir_test=data_dir_test,
        batch_size=batch_size, 
        num_workers=num_workers
    )

    # Standard Callbacks
    early_stoping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/", monitor="val_loss", mode="min")

    # Trainer setup
    trainer = pl.Trainer(
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        strategy=strategy,
        devices=devices,
        accelerator=accelerator,
        logger=logger,
        profiler=profiler,
        callbacks=[early_stoping, checkpoint_callback]
    )

    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)