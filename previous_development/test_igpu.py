#!/usr/bin/env python3
import torch

if torch.cuda.is_available():
    print("✅ ROCm/HIP is available!")
    device = torch.device("cuda")
    try:
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = x @ y
        print("✅ Tensor multiplication successful on iGPU!")
        print(z.cpu())
    except Exception as e:
        print(f"❌ iGPU test failed: {e}")
else:
    print("❌ ROCm/HIP is not available.")
