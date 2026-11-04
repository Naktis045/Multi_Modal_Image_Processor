import torch
from torch.cuda import device
from tqdm import tqdm
import pytorch_lightning as pl

from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers,min_epochs, max_epochs, data_dir, accelerator, devices,learning_rate
from Conv_Net.model import Model
from Conv_Net.dataset import ConvDataModule
from callbacks import CustomCallback

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.strategies import DeepSpeedStrategy


torch.set_float32_matmul_precision('medium') #precision for matrix multiplication, can be 'high', 'medium', or 'low'
#This setting can help improve performance on certain hardware, 
#especially GPUs, by allowing for faster computations with a trade-off in numerical precision ******

if __name__ == '__main__':
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="mnist_model")

# Strategy for Mult. GPU training
    strategy = DeepSpeedStrategy # can very depending from the model
# Profiler - compute, time management;
    profiler = pl.profilers.PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler1"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, wramp=1, active=20),# also can add ,repeat=1 
    ),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,  # Tracks Python function names causing bottlenecks

    model = Model(input_size = input_size, 
                  learning_rate = learning_rate, 
                  hidden_size = hidden_size, 
                  output_size = output_size, 
                  num_classes = num_classes)
    model.to(device)

    datamodule = ConvDataModule(data_dir = data_dir, 
                         batch_size = batch_size, 
                         num_workers = num_workers)

    trainer = pl.Trainer(min_epochs = min_epochs,
                          max_epochs = max_epochs,
                          strategy=strategy ,
                          devices = devices, 
                          accelerator = accelerator, 
                          logger = logger,
                          profiler = profiler,
                          callbacks = [model.early_stopping()],
                          # OR
                          #callbacks = [CustomCallback()],
                          )
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
    # OR
    #trainer.save_checkpoint("model.ckpt")
    #trainer.eval()--for inference
    # Store Weights __only__
    # torch.save(model.state_dict(), "wieghts_only.pt")

    """
    Key Purposes of Using the Profiler

    Find Performance Bottlenecks: It tells you exactly what is slowing down your training. Is your GPU waiting around for data, or is the model computation itself too heavy?

    Step-Time Breakdown: It visually breaks down every training step into phases (e.g., data loading, data transfer to GPU, forward pass, backward pass, and optimizer updates) so you can see where time is being wasted.

    GPU Utilization Monitoring: It tracks how much of your GPU memory and compute power are actually being used, helping you spot memory leaks or underutilized hardware.

    Operator Tracking: It lists the slowest PyTorch operations (kernels) running on your GPU so you know which layers or functions need optimization.
    """


"""
Strategy :

In PyTorch Lightning, the strategy parameter inside the Trainer controls how your model and data are distributed across your hardware (CPUs, single GPU, multi-GPU, or clusters).
Choosing the right strategy depends on your model size and hardware setup.   Optimal Strategies by Task TypeStrategy ParameterBest Used ForWhy Use It"auto" (Default)Standard Single-Device or Basic Multi-GPUAutomatically detects your hardware and selects a sensible default (Single device or standard DDP).
"ddp"Standard multi-GPU / multi-node training where the model fits entirely in one GPU's VRAM.Replicates the full model on each GPU and synchronizes gradients. It is fast, robust, and the industry standard for computer vision and standard NLP models.
"fsdp"Large models (e.g., Large Language Models, massive Vision Transformers) that exceed a single GPU's memory.
Fully Sharded Data Parallel (Meta) splits model parameters, gradients, and optimizer states across multiple GPUs. 
It lets you train models up to several times larger than standard DDP."deepspeed"
Extremely large billion-parameter models requiring advanced memory offloading.
Microsoft's DeepSpeed (ZeRO stages) allows parameter/optimizer state partitioning and can offload memory to CPU RAM or NVMe disk if GPU memory runs out.
"""