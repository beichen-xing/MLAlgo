import numpy as np


image = np.random.rand(224, 224)
patch_size = 16

h, w = image.shape
patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size)\
    .transpose(0, 2, 1, 3)\
    .reshape(-1, patch_size, patch_size)


print(f"Original image shape: {image.shape}")
print(f"Shape of patches: {patches.shape}")
# print(f"Example patch (patch 0):\n{patches[0]}")


import torch

image = torch.arange(224 * 224).view(1, 1, 224, 224).float()
b, c, h, w = image.shape

patches = image.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)\
    .permute(0, 2, 4, 1, 3, 5)\
    .reshape(-1, c, patch_size, patch_size)

print(f"Original image shape: {image.shape}")
print(f"Shape of patches: {patches.shape}")
# print(f"Example patch (patch 0):\n{patches[0, 0]}")