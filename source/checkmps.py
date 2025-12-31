import torch
x = torch.randn(1, device="mps")
y = x * 2
print("MPS works:", y.device)
