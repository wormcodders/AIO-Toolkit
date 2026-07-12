import torch
from safetensors.torch import load_file
import sys

if len(sys.argv) < 2:
    print("Usage: python check_nan.py <path_to_safetensors_file>")
    sys.exit(1)

file_path = sys.argv[1]
try:
    tensors = load_file(file_path)
    has_nan = False
    for name, tensor in tensors.items():
        if torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item():
            print(f"CORRUPTION FOUND in layer: {name}")
            has_nan = True
            
    if not has_nan:
        print("Success! No NaNs or Infs found. The file is mathematically healthy.")
    else:
        print("FAILED: The file contains corrupted mathematical values (NaN/Inf).")
except Exception as e:
    print(f"Error loading file: {e}")
