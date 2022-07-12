import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v3_large(
    weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
model.eval()

scripted = torch.jit.script(model)
optimized = optimize_for_mobile(scripted)

optimized._save_for_lite_interpreter("model.ptl")  # beta functionality
