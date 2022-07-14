import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from external.superpoint.demo_superpoint import SuperPointNet


def save_model_for_mobile(model_class, state_dict_path, filename):
    model = model_class()
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    scripted = torch.jit.script(model)
    optimized = optimize_for_mobile(scripted)

    optimized._save_for_lite_interpreter(filename)  # beta functionality


# Save SuperPoint
save_model_for_mobile(SuperPointNet, "external/superpoint/superpoint_v1.pth",
                      "../app/src/main/assets/superpoint.ptl")
