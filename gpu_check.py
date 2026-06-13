"""Tiny sanity check that training will actually run on the GPU.

    python gpu_check.py

Mirrors the device-selection logic used in trainer.py / ppo_agent.py
(`torch.cuda.is_available() and args.cuda`) and confirms a tensor + a real
project network actually land on the GPU.
"""
import torch
import torch.nn as nn

print(f"torch version      : {torch.__version__}")
print(f"CUDA available     : {torch.cuda.is_available()}")
print(f"CUDA build         : {torch.version.cuda}")
print(f"GPU device count   : {torch.cuda.device_count()}")

# This is exactly how trainer.py / ppo_agent.py choose the device (cuda defaults to True).
cuda_arg = True
device = torch.device("cuda" if torch.cuda.is_available() and cuda_arg else "cpu")
print(f"Selected device    : {device}")

if device.type != "cuda":
    print("\n==> Training would run on the CPU. GPU is NOT being used.")
    raise SystemExit(0)

print(f"GPU name           : {torch.cuda.get_device_name(0)}")

# Real op on the GPU to confirm the runtime works, not just that it's detected.
x = torch.randn(2048, 2048, device=device)
y = (x @ x).sum()
torch.cuda.synchronize()
print(f"Matmul on GPU OK   : result device = {y.device}")

# Put a small MLP (same shape as the project's actor/critic) on the GPU.
net = nn.Sequential(nn.Linear(17, 64), nn.Tanh(), nn.Linear(64, 6)).to(device)
out = net(torch.randn(4, 17, device=device))
print(f"Network params on  : {next(net.parameters()).device}")
print(f"Forward pass on    : {out.device}")

mem = torch.cuda.memory_allocated() / 1e6
print(f"GPU memory in use  : {mem:.1f} MB")
print("\n==> GPU is working and will be used for training.")
