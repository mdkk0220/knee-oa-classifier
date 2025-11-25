# scripts/test_keys.py
import torch

path = "outputs/resnet50_finetune_combined/model_best.pth"
ckpt = torch.load(path, map_location="cpu")

state = ckpt.get("model_state_dict", ckpt)

print("\n===== MODEL STATE DICT KEYS =====")
for k in state.keys():
    print(k)
