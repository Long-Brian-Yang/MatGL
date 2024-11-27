import torch
import os
# model_path = "logs/checkpoints/model.pt"
# checkpoint = torch.load(model_path, map_location='cpu')
# print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Not a dict")

# import torch

# # load state dict
# state_file = "logs/checkpoints/state.pt"
# state_dict = torch.load(state_file, map_location='cpu')
# print("State.pt keys:", state_dict.keys() if isinstance(state_dict, dict) else "Not a dict")

print("Model path exists:", os.path.exists("./logs/checkpoints/model.pt"))
print("Absolute path:", os.path.abspath("./logs/checkpoints/model.pt"))