import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from external.superpoint.decoder import SuperPointDecoder
from external.superpoint.demo_superpoint import SuperPointNet


def save_for_mobile(module_class, state_dict_path=None, filename="output.ptl"):
    module = module_class()
    if state_dict_path is not None:
        module.load_state_dict(torch.load(state_dict_path))
    module.eval()

    scripted = torch.jit.script(module)
    optimized = optimize_for_mobile(scripted)

    print(f"========== Optimized code for {module_class.__name__} ==========")
    print(optimized.code)

    optimized._save_for_lite_interpreter(filename)


# Save SuperPoint
save_for_mobile(SuperPointNet, "external/superpoint/superpoint_v1.pth", "../app/src/main/assets/superpoint.ptl")

# Save SuperPoint decoder
save_for_mobile(SuperPointDecoder, filename="../app/src/main/assets/superpoint_decoder.ptl")
