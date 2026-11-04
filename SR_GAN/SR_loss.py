import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the modern weights API instead of to.device() from old versions
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:36].eval()# [:36] - 5th conv layer - before maxpool, afrer act
        #self.loss = nn.MSELoss()# if dont use Dropout - no .eval() is realy needed
        self.loss = nn.L1Loss() # L1 loss is often preferred over MSE for sharper perceptual results in SRGAN

# Don't change parameters for VGG net, Freezing parameters so gradients don't flow through VGG
        for param in self.vgg.parameters():
            param.requires_grad = False

    
    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)

        return self.loss(vgg_input_features, vgg_target_features)

"""
# Can do a test case
def test_vgg_loss():
    criterion = VGGLoss()
    
    # Create a dummy image tensor (Batch, Channels, Height, Width)
    img = torch.randn(1, 3, 256, 256)
    
    # Compute loss (comparing an image to itself should be ~0 loss)
    loss = criterion(img, img)
    print(f"Test Loss: {loss.item():.4f}")
"""
