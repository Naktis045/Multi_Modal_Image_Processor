import sys
from torch.cuda import device
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from callbacks import CustomCallback

from Cycle_GAN import Cycle_config
from Cycle_GAN.utils import save_checkpoint, load_checkpoint
from Cycle_GAN.Cycle_config import min_epochs, max_epochs, batch_size, num_workers, lr, lambda_identity, load_model, save_model, checkpoint_gen_A, checkpoint_gen_B, data_dir_train, data_dir_test, device, devices, accelerator
from Cycle_GAN.Cycle_dataset import CycleDataModule
from Cycle_GAN.Cycle_Generator_model import Generator
from Cycle_GAN.Cycle_Discriminator_model import Discriminator

torch.set_float32_matmul_precision('medium')

def train_fn(loader, disc_A, disc_B, gen_A, gen_B, optimizer_generator, optimizer_discriminator, l1, mse, gen_scalars, disc_scalars, profiler=None):
    loop = tqdm(loader, leave=True)
   
    for idx, (A, B) in enumerate(loop):
        A = A.to(device)
        B = B.to(device)

        # Train with discriminators (A, B)
        with torch.cuda.amp.autocast():
            fake_B = gen_B(B)
            disc_B_real = disc_B(A)
            dics_B_fake = disc_B(fake_B.detach())
            dics_B_real_loss = mse(disc_B_real, torch.ones_like(disc_B_real))
            disc_B_fake_loss = mse(dics_B_fake, torch.zeros_like(dics_B_fake))
            disc_B_loss = dics_B_real_loss + disc_B_fake_loss

            fake_A = gen_A(A)
            disc_A_real = disc_A(B)
            dics_A_fake = disc_A(fake_A.detach())
            dics_A_real_loss = mse(disc_A_real, torch.ones_like(disc_A_real))
            disc_A_fake_loss = mse(dics_A_fake, torch.zeros_like(dics_A_fake))
            disc_A_loss = dics_A_real_loss + disc_A_fake_loss

            disc_loss = (disc_B_loss + disc_A_loss) / 2

        optimizer_discriminator.zero_grad()
        disc_scalars.scale(disc_loss).backward()
        disc_scalars.step(optimizer_discriminator)
        disc_scalars.update()

        # Train Generators
        with torch.cuda.amp.autocast():
            disc_B_fake = disc_B(fake_B)
            disc_A_fake = disc_A(fake_A)
            loss_gen_B = mse(disc_B_fake, torch.ones_like(disc_B_fake))
            loss_gen_A = mse(disc_A_fake, torch.ones_like(disc_A_fake))

            cycle_A = gen_A(fake_A)
            cycle_B = gen_B(fake_B)
            cycle_A_loss = l1(A, cycle_A)
            cycle_B_loss = l1(B, cycle_B)

            gen_loss = (
                loss_gen_A
                + loss_gen_B
                + cycle_A_loss * Cycle_config.lambda_cycle
                + cycle_B_loss * Cycle_config.lambda_cycle
            )

        optimizer_generator.zero_grad()
        gen_scalars.scale(gen_loss).backward()
        gen_scalars.step(optimizer_generator)
        gen_scalars.update()

        if idx % 200 == 0:
            save_image(fake_B * 0.5 + 0.5, f"generated_image/human_face_{idx}.png")
            save_image(fake_A * 0.5 + 0.5, f"generated_image/human_restyled_{idx}.png")

        # Step the profiler forward on every batch iteration
        if profiler is not None:
            profiler.step()


def main():
    disc_A = Discriminator(in_channels=3).to(device)
    disc_B = Discriminator(in_channels=3).to(device)
    gen_A = Generator(img_channels=3, num_residuals=9).to(device)
    gen_B = Generator(img_channels=3, num_residuals=9).to(device)

    optimizer_generator = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=lr,
        betas=(0.5, 0.999)
    )
    
    optimizer_discriminator = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()), 
        lr=lr, 
        betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Load checkpoints if configured
    if Cycle_config.load_model:
        load_checkpoint(gen_A, optimizer_generator, checkpoint_gen_A, lr)
        load_checkpoint(gen_B, optimizer_generator, checkpoint_gen_B, lr)
        load_checkpoint(disc_A, optimizer_discriminator, checkpoint_gen_A, lr)
        load_checkpoint(disc_B, optimizer_discriminator, checkpoint_gen_B, lr)

    # Initialize DataModule and extract the training dataloader
    cycle_datamodule = CycleDataModule(
        data_dir_train=data_dir_train,
        data_dir_test=data_dir_test,
        batch_size=batch_size,
        num_workers=num_workers
    )
    cycle_datamodule.setup(stage="fit")
    loader = cycle_datamodule.train_dataloader()

    gen_scalars = torch.cuda.amp.GradScaler()
    disc_scalars = torch.cuda.amp.GradScaler()

    # Training Epoch Loop
    # Native PyTorch Profiler Context Manager
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(skip_first=10, wait=1, warmup=1, active=20),
        on_trace_ready=tensorboard_trace_handler("tb_logs/profiler1"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for epoch in range(max_epochs):
            print(f"Epoch [{epoch+1}/{max_epochs}]")
            train_fn(
                loader, disc_A, disc_B, gen_A, gen_B, 
                optimizer_generator, optimizer_discriminator, 
                l1, mse, gen_scalars, disc_scalars, prof=prof
            )


if __name__ == "__main__":
    main()