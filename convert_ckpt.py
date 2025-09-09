# convert_ckpt.py — make a pure state_dict .pth from TorchECG .pth.tar (PyTorch 2.6+)
import os
import torch
import torch_ecg

SRC = "/Users/steven/.cache/torch_ecg/saved_models/BestModel_ECG_UNET_LUDB_epoch122_09-08_19-34_metric_0.90.pth.tar"
DST = "outputs/checkpoints/unet1d_ludb_best.pth"
os.makedirs(os.path.dirname(DST), exist_ok=True)

# Allow-list the custom class used inside the checkpoint
torch.serialization.add_safe_globals([torch_ecg.cfg.CFG])

# Since this is your own checkpoint, it's trusted; load with weights_only=False
ckpt = torch.load(SRC, map_location="cpu", weights_only=False)

# TorchECG trainer usually stores model weights under "model_state_dict"
state_dict = ckpt.get("model_state_dict", ckpt)
if not isinstance(state_dict, dict):
    raise RuntimeError("Unexpected checkpoint structure: no model_state_dict found")

torch.save(state_dict, DST)
print(f"Saved pure state_dict -> {DST}  (keys: {len(state_dict)})")
